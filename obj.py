"""obj.py -- a minimal, from-scratch Wavefront OBJ loader for the
threejs-port project.

Loads a dropped ``.obj`` text file by name from the host registry on
``three.OBJLoader`` and builds a ``three.Group`` of static meshes:

  * ``v`` / ``vt`` / ``vn`` attribute pools,
  * ``f`` faces in all four corner forms (``v``, ``v/vt``, ``v//vn``,
    ``v/vt/vn``), with negative (relative) indices and fan
    triangulation of polygons with more than three corners,
  * ``o`` / ``g`` object/group boundaries and ``usemtl`` material
    switches split the output into one mesh per (group, material)
    run, mirroring the real OBJLoader,
  * ``mtllib`` libraries are resolved against the same name registry
    (drop the ``.mtl`` next to the ``.obj``); only the diffuse colour
    ``Kd`` is honoured, mapped onto ``MeshLambertMaterial``.  Faces
    without a usable normal pool get computed vertex normals.

Deliberately out of scope (never a crash; noted on the group's
``userData['obj_warnings']`` where applicable): free-form geometry
(curves/surfaces), point/line primitives, textures from MTL
(``map_Kd``), specular/ambient colours, external files that were not
dropped alongside.

Only whitelisted modules are used (numpy + the sibling ``three``
module), so the file passes the Code & Magic check_code rules like
the rest of the app.  The module is self-contained: ``three.py``
imports it lazily from the ``OBJLoader`` facade.
"""

import numpy as np

import three


class OBJError(Exception):
    """Raised when an OBJ asset cannot be parsed at all."""


# ---------------------------------------------------------------------------
# MTL parsing (diffuse colour only)
# ---------------------------------------------------------------------------

def parse_mtl(text):
    """Parse MTL text into ``{material_name: (r, g, b)}`` floats."""
    materials = {}
    current = None
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        keyword = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        if keyword == "newmtl":
            current = rest.strip()
            materials.setdefault(current, (0.8, 0.8, 0.8))
        elif keyword == "kd" and current is not None:
            try:
                values = [float(v) for v in rest.split()[:3]]
            except ValueError:
                continue
            if len(values) == 3:
                materials[current] = tuple(
                    min(1.0, max(0.0, v)) for v in values
                )
    return materials


def _color_hex(rgb):
    r, g, b = (int(round(min(1.0, max(0.0, c)) * 255.0)) for c in rgb)
    return (r << 16) | (g << 8) | b


# ---------------------------------------------------------------------------
# OBJ parsing
# ---------------------------------------------------------------------------

def _resolve_index(token, count, what):
    """Resolve a 1-based (or negative relative) OBJ index to 0-based."""
    idx = int(token)
    if idx == 0:
        raise OBJError("%s index 0 is invalid in OBJ" % what)
    resolved = idx - 1 if idx > 0 else count + idx
    if resolved < 0 or resolved >= count:
        raise OBJError(
            "%s index %d out of range (pool size %d)" % (what, idx, count)
        )
    return resolved


def _parse_corner(token, n_pos, n_uv, n_nrm):
    """Parse one face corner into a (v, vt, vn) key; vt/vn may be None."""
    fields = token.split("/")
    vi = _resolve_index(fields[0], n_pos, "vertex")
    ti = None
    ni = None
    if len(fields) > 1 and fields[1] != "":
        ti = _resolve_index(fields[1], n_uv, "uv")
    if len(fields) > 2 and fields[2] != "":
        ni = _resolve_index(fields[2], n_nrm, "normal")
    return (vi, ti, ni)


class _Segment:
    """A run of faces sharing one (group, material) pair."""

    __slots__ = ("group", "material", "corners")

    def __init__(self, group, material):
        self.group = group
        self.material = material
        self.corners = []  # flat list of (v, vt, vn) keys, 3 per triangle


def parse_obj(text):
    """Parse OBJ text into ``(segments, pools, mtl_names, warnings)``.

    ``pools`` is ``(positions, uvs, normals)`` as lists of tuples;
    segment corners index into them.
    """
    positions = []
    uvs = []
    normals = []
    segments = []
    mtl_names = []
    warnings = []
    group_name = ""
    material_name = None
    current = None

    def ensure_segment():
        nonlocal current
        if current is None:
            current = _Segment(group_name, material_name)
            segments.append(current)
        return current

    for raw in str(text).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        keyword = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        if keyword == "v":
            values = [float(v) for v in rest.split()[:3]]
            if len(values) != 3:
                raise OBJError("bad vertex line: %r" % raw)
            positions.append(tuple(values))
        elif keyword == "vt":
            values = [float(v) for v in rest.split()[:2]]
            if len(values) < 2:
                raise OBJError("bad uv line: %r" % raw)
            uvs.append((values[0], values[1]))
        elif keyword == "vn":
            values = [float(v) for v in rest.split()[:3]]
            if len(values) != 3:
                raise OBJError("bad normal line: %r" % raw)
            normals.append(tuple(values))
        elif keyword == "f":
            tokens = rest.split()
            if len(tokens) < 3:
                warnings.append("face with fewer than 3 corners skipped")
                continue
            corners = [
                _parse_corner(tok, len(positions), len(uvs), len(normals))
                for tok in tokens
            ]
            seg = ensure_segment()
            for i in range(1, len(corners) - 1):
                seg.corners.extend(
                    (corners[0], corners[i], corners[i + 1])
                )
        elif keyword in ("o", "g"):
            name = rest.strip()
            if name != group_name:
                group_name = name
                current = None
        elif keyword == "usemtl":
            name = rest.strip() or None
            if name != material_name:
                material_name = name
                current = None
        elif keyword == "mtllib":
            for name in rest.split():
                if name not in mtl_names:
                    mtl_names.append(name)
        elif keyword in ("s", "vp", "p", "l", "curv", "surf", "mg"):
            warnings.append("unsupported OBJ statement %r ignored" % keyword)
        # Anything else (vt extras, etc.) is ignored silently.
    segments = [s for s in segments if s.corners]
    return segments, (positions, uvs, normals), mtl_names, warnings


def _build_mesh(segment, pools, materials):
    positions, uvs, normals = pools
    cache = {}
    out_pos = []
    out_uv = []
    out_nrm = []
    indices = []
    have_uv = True
    have_nrm = True
    for key in segment.corners:
        new_index = cache.get(key)
        if new_index is None:
            vi, ti, ni = key
            new_index = len(out_pos)
            cache[key] = new_index
            out_pos.append(positions[vi])
            if ti is None:
                have_uv = False
                out_uv.append((0.0, 0.0))
            else:
                out_uv.append(uvs[ti])
            if ni is None:
                have_nrm = False
                out_nrm.append((0.0, 0.0, 0.0))
            else:
                out_nrm.append(normals[ni])
        indices.append(new_index)

    geometry = three.BufferGeometry()
    geometry.attributes["position"] = np.array(out_pos, dtype=np.float32)
    if have_uv:
        geometry.attributes["uv"] = np.array(out_uv, dtype=np.float32)
    if have_nrm:
        geometry.attributes["normal"] = np.array(out_nrm, dtype=np.float32)
    geometry.index = np.array(indices, dtype=np.int64)
    if not have_nrm:
        geometry.computeVertexNormals()

    rgb = materials.get(segment.material) if segment.material else None
    if rgb is not None:
        material = three.MeshLambertMaterial({"color": _color_hex(rgb)})
    else:
        material = three.MeshLambertMaterial()
    name = segment.group or "mesh"
    if segment.material:
        name = "%s_%s" % (name, segment.material) if segment.group else segment.material
    mesh = three.Mesh(geometry, material, name=name)
    return mesh


def build_obj(text, materials=None):
    """Build a ``three.Group`` from OBJ text.

    ``materials`` maps material names to (r, g, b) diffuse colours
    (from :func:`parse_mtl`); missing materials fall back to the
    default Lambert grey-white.
    """
    segments, pools, _mtl_names, warnings = parse_obj(text)
    if not segments:
        raise OBJError("no faces found in OBJ text")
    group = three.Group()
    materials = materials or {}
    for segment in segments:
        group.add(_build_mesh(segment, pools, materials))
    if warnings:
        group.userData["obj_warnings"] = list(warnings)
    return group


# ---------------------------------------------------------------------------
# Entry points used by the three.OBJLoader facade
# ---------------------------------------------------------------------------

def load_text(text, name="", materials=None):
    """Parse OBJ text (plus any registered MTL it references).

    ``materials`` (name -> (r, g, b)), when given -- the OBJLoader's
    ``setMaterials`` override -- takes precedence over the ``mtllib``
    libraries the .obj itself names; names it does not cover still
    fall back to those libraries."""
    registry = three.OBJLoader._registry
    _segments, _pools, mtl_names, _warnings = parse_obj(text)
    resolved = {}
    for mtl_name in mtl_names:
        mtl_text = _registry_lookup(registry, mtl_name)
        if mtl_text is not None:
            resolved.update(parse_mtl(mtl_text))
    if materials:
        resolved.update(materials)
    group = build_obj(text, resolved)
    if name:
        group.name = str(name).rsplit(".", 1)[0]
    return group


def _registry_lookup(registry, name):
    key = str(name)
    data = registry.get(key)
    if data is None and ("/" in key or "\\" in key):
        data = registry.get(key.replace("\\", "/").split("/")[-1])
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data).decode("utf-8", errors="replace")
    return data


def load_named(name, materials=None):
    """Resolve ``name`` against the OBJLoader registry and load it."""
    registry = three.OBJLoader._registry
    text = _registry_lookup(registry, name)
    if text is None:
        raise OBJError(
            "no dropped OBJ file named %r (registered: %s)"
            % (str(name), ", ".join(sorted(registry.keys())) or "none")
        )
    return load_text(text, name=str(name), materials=materials)

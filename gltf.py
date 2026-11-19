"""gltf.py -- a minimal, from-scratch glTF 2.0 loader for the threejs-port
project.

Loads a dropped ``.glb`` (binary glTF) or ``.gltf`` (JSON with data-URI
embedded buffers) by name from the host registry on ``three.GLTFLoader``
and builds a static scene graph out of ``three.py`` objects:

  * node hierarchy with ``matrix`` or TRS transforms,
  * mesh primitives (mode 4 / triangles only) with position / normal /
    uv / index accessors (interleaved bufferViews and normalised integer
    attributes are supported),
  * PBR base colour: ``baseColorFactor`` becomes the material colour and
    ``baseColorTexture`` becomes the material map when the image can be
    resolved (embedded image bytes decoded with PIL, or an image ``uri``
    found in the ``TextureLoader`` registry).

Deliberately out of scope (skipped with a warning recorded on the
result's ``warnings`` list, never a crash): skinning, animation, morph
targets, cameras, external buffer/image references, Draco and other
extensions, non-triangle primitive modes, sparse accessors.

Only whitelisted modules are used (json / struct / base64 / math +
numpy + PIL when available), so the file passes the Code & Magic
check_code rules like the rest of the app.
"""

import base64
import json
import struct

import numpy as np

import three


class GLTFError(Exception):
    """Raised when a glTF asset cannot be parsed at all."""


class _MissingBuffer(Exception):
    """Internal: an accessor points at an unavailable (external) buffer."""


# ---------------------------------------------------------------------------
# Container parsing
# ---------------------------------------------------------------------------

_GLB_MAGIC = 0x46546C67  # 'glTF'
_GLB_JSON = 0x4E4F534A   # 'JSON'
_GLB_BIN = 0x004E4942    # 'BIN\0'


def parse_container(data):
    """Split raw bytes into (document dict, list of buffer byte strings).

    Buffers that cannot be obtained (external references) are represented
    by ``None`` so the rest of the asset can still be read.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise GLTFError("glTF data must be bytes")
    data = bytes(data)
    if len(data) >= 4 and struct.unpack_from("<I", data, 0)[0] == _GLB_MAGIC:
        return _parse_glb(data)
    return _parse_gltf_json(data)


def _parse_glb(data):
    if len(data) < 12:
        raise GLTFError("GLB too short")
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise GLTFError("bad GLB magic")
    if version != 2:
        raise GLTFError("only glTF 2.0 GLB is supported")
    doc = None
    bin_chunk = b""
    offset = 12
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_len]
        offset += chunk_len
        if chunk_type == _GLB_JSON:
            try:
                doc = json.loads(chunk.decode("utf-8"))
            except Exception as exc:
                raise GLTFError("GLB JSON chunk is invalid: %s" % exc)
        elif chunk_type == _GLB_BIN:
            bin_chunk = chunk
    if doc is None:
        raise GLTFError("GLB has no JSON chunk")
    buffers = []
    for i, buf in enumerate(doc.get("buffers", [])):
        if buf.get("uri"):
            buffers.append(_resolve_buffer_uri(buf["uri"], i))
        elif i == 0:
            buffers.append(bin_chunk)
        else:
            buffers.append(None)
    if not buffers and doc.get("buffers"):
        buffers = [bin_chunk]
    return doc, buffers


def _parse_gltf_json(data):
    try:
        text = data.decode("utf-8")
    except Exception as exc:
        raise GLTFError("not a GLB and not UTF-8 JSON: %s" % exc)
    try:
        doc = json.loads(text)
    except Exception as exc:
        raise GLTFError("invalid glTF JSON: %s" % exc)
    if not isinstance(doc, dict):
        raise GLTFError("glTF document must be a JSON object")
    buffers = []
    for i, buf in enumerate(doc.get("buffers", [])):
        buffers.append(_resolve_buffer_uri(buf.get("uri"), i))
    return doc, buffers


def _resolve_buffer_uri(uri, index):
    """Return the bytes for a buffer: a data URI, or an external
    reference resolved against the GLTFLoader registry (a .bin
    dropped / staged alongside its .gltf, by exact name then by
    basename).  Unresolvable references stay None so the rest of the
    asset can still be read."""
    if not uri:
        return None
    if isinstance(uri, str) and uri.startswith("data:"):
        comma = uri.find(",")
        if comma >= 0:
            payload = uri[comma + 1:]
            if ";base64" in uri[:comma]:
                try:
                    return base64.b64decode(payload)
                except Exception:
                    return None
            return payload.encode("utf-8", "replace")
    if isinstance(uri, str):
        registry = three.GLTFLoader._registry
        data = registry.get(uri)
        if data is None:
            base = uri.replace("\\", "/").split("/")[-1]
            data = registry.get(base)
        if data is not None:
            return data
    return None  # external reference: not registered


# ---------------------------------------------------------------------------
# Accessor reading
# ---------------------------------------------------------------------------

_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _read_accessor(doc, buffers, index):
    """Read one accessor as a numpy array shaped (count, components).

    Integer accessors keep their integer dtype; ``normalized`` integer
    accessors are converted to float32 in [0, 1] / [-1, 1] per the spec.
    """
    accessors = doc.get("accessors", [])
    if index is None or index < 0 or index >= len(accessors):
        raise GLTFError("accessor index out of range: %r" % (index,))
    acc = accessors[index]
    comp = acc.get("componentType")
    dtype = _COMPONENT_DTYPES.get(comp)
    if dtype is None:
        raise GLTFError("unsupported componentType: %r" % (comp,))
    ncomp = _TYPE_COUNTS.get(acc.get("type"), 1)
    count = int(acc.get("count", 0))
    if acc.get("sparse") is not None:
        raise _MissingBuffer("sparse accessors are not supported")
    view_index = acc.get("bufferView")
    if view_index is None:
        # An accessor without a bufferView is initialised to zero.
        return np.zeros((count, ncomp), dtype=dtype)
    views = doc.get("bufferViews", [])
    if view_index < 0 or view_index >= len(views):
        raise GLTFError("bufferView index out of range: %r" % (view_index,))
    view = views[view_index]
    buf_index = int(view.get("buffer", 0))
    if buf_index < 0 or buf_index >= len(buffers) or buffers[buf_index] is None:
        raise _MissingBuffer("buffer %d is unavailable" % buf_index)
    data = buffers[buf_index]
    item = np.dtype(dtype).itemsize * ncomp
    stride = int(view.get("byteStride", item))
    start = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    if count == 0:
        return np.zeros((0, ncomp), dtype=dtype)
    if stride == item:
        flat = np.frombuffer(data, dtype=dtype, count=count * ncomp,
                             offset=start)
        arr = np.array(flat, dtype=dtype).reshape(count, ncomp)
    else:
        raw = np.ndarray(
            shape=(count, ncomp), dtype=dtype, buffer=data, offset=start,
            strides=(stride, np.dtype(dtype).itemsize),
        )
        arr = np.array(raw)
    if acc.get("normalized"):
        arr = _denormalize(arr, dtype)
    return arr


def _denormalize(arr, dtype):
    info = np.iinfo(dtype) if np.dtype(dtype).kind in "iu" else None
    if info is None:
        return arr.astype(np.float32)
    out = arr.astype(np.float32) / float(info.max)
    if info.min < 0:
        out = np.maximum(out, -1.0)
    return out


def _read_index(doc, buffers, index):
    arr = _read_accessor(doc, buffers, index)
    return np.asarray(arr).reshape(-1).astype(np.int64)


def _read_float_attr(doc, buffers, index, ncomp):
    arr = _read_accessor(doc, buffers, index)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, ncomp)
    return arr.reshape(-1, ncomp).astype(np.float32)


# ---------------------------------------------------------------------------
# Images / textures
# ---------------------------------------------------------------------------

def _decode_image_bytes(data):
    """Decode PNG/JPEG bytes to an (h, w, 3) numpy array via PIL.

    Returns None when PIL is unavailable or the bytes do not decode."""
    if data is None:
        return None
    try:
        import io

        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(io.BytesIO(bytes(data))) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:
        return None


def _resolve_texture(doc, buffers, tex_index, warnings):
    """Resolve a glTF texture to a three.Texture (or None)."""
    textures = doc.get("textures", [])
    if tex_index is None or tex_index < 0 or tex_index >= len(textures):
        return None
    tex = textures[tex_index]
    if tex.get("extensions"):
        warnings.append("texture %d uses extensions; ignored" % tex_index)
    source = tex.get("source")
    images = doc.get("images", [])
    if source is None or source < 0 or source >= len(images):
        return None
    image = images[source]
    uri = image.get("uri")
    if uri and not uri.startswith("data:"):
        # A named image may match a dropped image registered with the
        # host (TextureLoader registry), e.g. a file dropped alongside.
        loader = three.TextureLoader()
        for key in (uri, uri.replace("\\", "/").split("/")[-1]):
            if key in three.TextureLoader.registered_names():
                return loader.load(key)
        warnings.append(
            "image %d is an external reference (%r); texture skipped"
            % (source, uri)
        )
        return None
    if uri and uri.startswith("data:"):
        comma = uri.find(",")
        raw = None
        if comma >= 0 and ";base64" in uri[:comma]:
            try:
                raw = base64.b64decode(uri[comma + 1:])
            except Exception:
                raw = None
        arr = _decode_image_bytes(raw)
        if arr is None:
            warnings.append("image %d (data URI) could not be decoded" % source)
            return None
        return three.Texture(arr, name="gltf-image-%d" % source)
    view_index = image.get("bufferView")
    if view_index is not None:
        views = doc.get("bufferViews", [])
        if 0 <= view_index < len(views):
            view = views[view_index]
            buf_index = int(view.get("buffer", 0))
            if 0 <= buf_index < len(buffers) and buffers[buf_index] is not None:
                data = buffers[buf_index]
                start = int(view.get("byteOffset", 0))
                length = int(view.get("byteLength", 0))
                arr = _decode_image_bytes(data[start:start + length])
                if arr is not None:
                    return three.Texture(arr, name="gltf-image-%d" % source)
        warnings.append("image %d (embedded) could not be decoded" % source)
        return None
    warnings.append("image %d has no usable source; texture skipped" % source)
    return None


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def _build_material(doc, buffers, mat_index, warnings):
    if mat_index is None:
        return three.MeshLambertMaterial()
    materials = doc.get("materials", [])
    if mat_index < 0 or mat_index >= len(materials):
        return three.MeshLambertMaterial()
    mat = materials[mat_index]
    if mat.get("extensions"):
        warnings.append(
            "material %d uses extensions %s; ignored"
            % (mat_index, sorted(mat["extensions"].keys()))
        )
    pbr = mat.get("pbrMetallicRoughness", {}) or {}
    factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
    rgb = [float(factor[i]) if i < len(factor) else 1.0 for i in range(3)]
    alpha = float(factor[3]) if len(factor) > 3 else 1.0
    params = {"color": three.Color(rgb[0], rgb[1], rgb[2])}
    tex = None
    base_tex = pbr.get("baseColorTexture")
    if isinstance(base_tex, dict):
        tex = _resolve_texture(doc, buffers, base_tex.get("index"), warnings)
        if tex is not None:
            params["map"] = tex
    material = three.MeshLambertMaterial(params)
    emissive = mat.get("emissiveFactor")
    if emissive:
        material.emissive = three.Color(
            float(emissive[0]), float(emissive[1]), float(emissive[2])
        )
    if mat.get("doubleSided"):
        material.side = 2  # DoubleSide
    if alpha < 1.0:
        material.opacity = alpha
        if mat.get("alphaMode") == "BLEND":
            material.transparent = True
    if mat.get("name"):
        material.name = str(mat["name"])
    return material


# ---------------------------------------------------------------------------
# Geometry / meshes
# ---------------------------------------------------------------------------

def _build_primitive(doc, buffers, prim, warnings):
    """Build one Mesh for a primitive, or None when it is out of scope."""
    if prim.get("extensions"):
        warnings.append(
            "primitive uses extensions %s; skipped"
            % sorted(prim["extensions"].keys())
        )
        return None
    mode = prim.get("mode", 4)
    if mode != 4:
        warnings.append("primitive mode %r is not triangles; skipped" % mode)
        return None
    attrs = prim.get("attributes", {}) or {}
    if "POSITION" not in attrs:
        warnings.append("primitive without POSITION; skipped")
        return None
    geometry = three.BufferGeometry()
    try:
        position = _read_float_attr(doc, buffers, attrs["POSITION"], 3)
        geometry.setAttribute("position", position)
        if "NORMAL" in attrs:
            geometry.setAttribute(
                "normal", _read_float_attr(doc, buffers, attrs["NORMAL"], 3)
            )
        if "TEXCOORD_0" in attrs:
            geometry.setAttribute(
                "uv", _read_float_attr(doc, buffers, attrs["TEXCOORD_0"], 2)
            )
        if prim.get("indices") is not None:
            geometry.setIndex(_read_index(doc, buffers, prim["indices"]))
        else:
            geometry.setIndex(
                np.arange(position.shape[0], dtype=np.int64)
            )
    except _MissingBuffer as exc:
        warnings.append("primitive skipped: %s" % exc)
        return None
    if "NORMAL" not in attrs:
        geometry.computeVertexNormals()
    material = _build_material(doc, buffers, prim.get("material"), warnings)
    return three.Mesh(geometry, material)


def _build_mesh_object(doc, buffers, mesh_index, warnings):
    meshes = doc.get("meshes", [])
    mesh = meshes[mesh_index]
    built = []
    for prim in mesh.get("primitives", []):
        obj = _build_primitive(doc, buffers, prim, warnings)
        if obj is not None:
            built.append(obj)
    if not built:
        return three.Group()
    if len(built) == 1:
        obj = built[0]
    else:
        obj = three.Group()
        for child in built:
            obj.add(child)
    if mesh.get("name"):
        obj.name = str(mesh["name"])
    if mesh.get("weights") is not None:
        warnings.append(
            "mesh %d has morph target weights; ignored" % mesh_index
        )
    return obj


# ---------------------------------------------------------------------------
# Nodes / scenes
# ---------------------------------------------------------------------------

def _apply_node_transform(obj, node):
    matrix = node.get("matrix")
    if matrix is not None and len(matrix) >= 16:
        m = three.Matrix4()
        m.fromArray([float(v) for v in matrix[:16]])
        m.decompose(obj.position, obj.quaternion, obj.scale)
        return
    translation = node.get("translation")
    if translation:
        obj.position.set(
            float(translation[0]), float(translation[1]), float(translation[2])
        )
    rotation = node.get("rotation")
    if rotation:
        obj.quaternion.set(
            float(rotation[0]), float(rotation[1]),
            float(rotation[2]), float(rotation[3]),
        )
    scale = node.get("scale")
    if scale:
        obj.scale.set(float(scale[0]), float(scale[1]), float(scale[2]))


def _build_node(doc, buffers, node_index, warnings, built):
    if node_index in built:
        return built[node_index]
    nodes = doc.get("nodes", [])
    node = nodes[node_index]
    if node.get("mesh") is not None:
        obj = _build_mesh_object(doc, buffers, int(node["mesh"]), warnings)
        if not isinstance(obj, three.Object3D):
            obj = three.Group()
    else:
        obj = three.Group()
    built[node_index] = obj
    if node.get("name"):
        obj.name = str(node["name"])
    _apply_node_transform(obj, node)
    if node.get("camera") is not None:
        warnings.append(
            "node %d holds a camera; cameras are not loaded" % node_index
        )
    if node.get("skin") is not None:
        warnings.append(
            "node %d is skinned; skinning is not supported "
            "(bind-pose geometry is used)" % node_index
        )
    if node.get("extensions"):
        warnings.append(
            "node %d uses extensions %s; ignored"
            % (node_index, sorted(node["extensions"].keys()))
        )
    for child_index in node.get("children", []) or []:
        child = _build_node(doc, buffers, int(child_index), warnings, built)
        obj.add(child)
    return obj


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

class GLTFResult:
    """What GLTFLoader hands to onLoad (three.js calls it ``gltf``)."""

    isGLTF = True

    def __init__(self, scene, scenes, warnings, asset):
        self.scene = scene
        self.scenes = scenes
        self.animations = []
        self.cameras = []
        self.warnings = warnings
        self.asset = asset
        self.userData = {}

    def __repr__(self):
        return "GLTFResult(scene=%r, warnings=%d)" % (
            self.scene, len(self.warnings),
        )


def build_gltf(doc, buffers):
    """Build a GLTFResult from a parsed document + buffer list."""
    warnings = []
    for i, buf in enumerate(doc.get("buffers", [])):
        if buf.get("uri") and not str(buf["uri"]).startswith("data:"):
            warnings.append(
                "buffer %d is an external reference (%r); "
                "external buffers are not supported" % (i, buf["uri"])
            )
        elif i < len(buffers) and buffers[i] is None and buf.get("uri"):
            warnings.append("buffer %d could not be decoded" % i)
    if doc.get("skins"):
        warnings.append(
            "document has %d skin(s); skinning is not supported"
            % len(doc["skins"])
        )
    if doc.get("animations"):
        warnings.append(
            "document has %d animation(s); animations are not loaded"
            % len(doc["animations"])
        )
    if doc.get("cameras"):
        warnings.append(
            "document has %d camera(s); cameras are not loaded"
            % len(doc["cameras"])
        )
    for ext in doc.get("extensionsUsed", []) or []:
        warnings.append("extension used by document: %s (not supported)" % ext)

    scenes = []
    built_nodes = {}
    for scene_def in doc.get("scenes", []) or []:
        group = three.Group()
        if scene_def.get("name"):
            group.name = str(scene_def["name"])
        for root_index in scene_def.get("nodes", []) or []:
            group.add(
                _build_node(doc, buffers, int(root_index), warnings,
                            built_nodes)
            )
        scenes.append(group)
    if scenes:
        default = int(doc.get("scene", 0))
        if default < 0 or default >= len(scenes):
            default = 0
        scene = scenes[default]
    else:
        scene = three.Group()
        scenes = [scene]
    asset = doc.get("asset", {}) or {}
    return GLTFResult(scene, scenes, warnings, asset)


def load_bytes(data, name=""):
    """Parse raw .glb / .gltf bytes and build the scene."""
    doc, buffers = parse_container(data)
    result = build_gltf(doc, buffers)
    result.name = str(name)
    return result


def load_named(name):
    """Resolve ``name`` against the GLTFLoader registry and load it."""
    registry = three.GLTFLoader._registry
    key = str(name)
    data = registry.get(key)
    if data is None and ("/" in key or "\\" in key):
        data = registry.get(key.replace("\\", "/").split("/")[-1])
    if data is None:
        raise GLTFError(
            "no dropped glTF file named %r (registered: %s)"
            % (key, ", ".join(sorted(registry.keys())) or "none")
        )
    return load_bytes(data, name=key)

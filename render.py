# render.py -- numpy software rasterizer for the three.js port.
#
# Interface:
#   render_scene(scene, camera, width, height)
#       -> np.ndarray (height, width) uint8 of palette indices (16 UI colours +
#          a 210-colour cube, see cube_colours)
#
# - Walks the scene graph, collects Meshes and lights.
# - Transforms vertices by matrixWorld, then by the camera view/projection
#   (perspective projection), clips triangles against the near/far planes in
#   view space, and fills each triangle inside its screen bounding box with
#   numpy.  A z-buffer resolves overlap.  Per-mesh work that used to be
#   per-triangle Python (face normals, flat shading, projection, clip
#   classification) is vectorised over the whole mesh; triangles that miss
#   both clip planes take a fast path, and only the rare clipped ones go
#   through the Sutherland-Hodgman polygon path.
# - Material.side: FrontSide (the default) culls back faces,
#   BackSide culls front faces, DoubleSide draws both.  Facing is tested
#   with the world-space geometric normal against the view direction.
# - Fog: when scene.fog is a THREE.Fog, each fragment's RGB is
#   blended linearly toward the fog colour by view depth (three.js:
#   factor = (depth - near) / (far - near), clamped) BEFORE palette
#   quantisation.  Points and lines are fogged too; wireframe edges use
#   the face-centre depth (flat shading).
# - Flat per-face shading:
#     MeshBasicMaterial   -> the material colour as-is
#     MeshLambertMaterial -> colour * (sum of AmbientLight contributions
#                            + sum over DirectionalLights of
#                              light colour * intensity * max(dot(N, L), 0))
#     MeshPhongMaterial   -> the Lambert terms plus a Blinn-Phong specular
#                            term per DirectionalLight (flat per face, the
#                            half vector taken at the face centre)
#   HemisphereLight adds an environment term mixing its ground and sky
#   colours by the face normal along the light's axis (its normalised
#   position, straight up by default).  PointLight and SpotLight
#   contribute per face from the face centre: the three.js
#   physical distance attenuation (1 / max(d ** decay, 0.01), windowed
#   to zero at the cutoff distance when one is set), and for spots a
#   smoothstep cone factor between cos(angle) and cos(angle *
#   (1 - penumbra)).  Both feed the diffuse sum and the Blinn-Phong
#   specular of Phong/Standard.  Only DirectionalLight casts shadows.
#     MeshNormalMaterial  -> the view-space face normal mapped from
#                            [-1, 1] to RGB [0, 1], three.js style
#     MeshStandardMaterial (approximation -- not GGX PBR) ->
#                            the Lambert diffuse scaled by (1 - metalness)
#                            plus a Blinn-Phong specular whose exponent
#                            comes from roughness (0 = tight, 1 = broad
#                            and faint) and whose colour is the
#                            conductor F0 mix(0.04 grey, base colour,
#                            metalness)
#   When scene.environment (or a material's envMap) holds a CubeTexture,
#   its reflection -- sampled along the per-face reflected view
#   direction -- is mixed into the Phong/Standard specular (Schlick
#   Fresnel with the same F0 for Standard; the specular colour for
#   Phong), scaled by envMapIntensity.
#   Any material with wireframe=True draws only the triangle edges
#   (z-tested line segments in the flat face colour).
#   where N is the world-space geometric face normal (oriented toward the
#   viewer, because both sides are rendered) and L points from the surface
#   toward the light (DirectionalLight.position relative to its target at
#   the origin, three.js style).
# - The shaded RGB is mapped to the nearest Pyxel palette colour (Euclidean
#   distance in RGB).  One deliberate exception, inherited from the approved
#   mock (mock_v4): a Lambert face whose shaded colour would quantise to
#   black even though it is clearly not black is drawn as a 2x2-style
#   checker dither between black and its second-nearest palette colour, so
#   a dark-but-lit face stays visible against the black background.
# - If a Lambert scene contains no lights at all, a built-in fallback light
#   rig (ambient + one directional, the mock's setup) is used so the mesh
#   does not collapse to black; scenes that do carry lights are shaded
#   strictly by those lights.
# - Background: scene.background (a Color) or black (palette 0).
# - Points / Line objects are drawn after the mesh pass,
#   sharing its z-buffer: points are plotted as size x size blocks (with
#   three.js-style size attenuation), lines as z-tested segments.
# - LineSegments and the line helpers are drawn in the same
#   pass as independent segment pairs.  Materials with vertexColors use
#   the geometry's 'color' attribute (per-point colours, segment colours
#   interpolated along the segment).  Sprites are drawn last as
#   view-plane billboards (map x color, material.rotation, center
#   anchor, sizeAttenuation), also sharing the z-buffer.
# - Shadows: when the renderer passes shadow_enabled and one
#   DirectionalLight has castShadow set, a fixed 128x128 depth map is
#   rendered from that light's viewpoint (orthographic frustum fitted to
#   the casters and receivers).  Lambert/Phong meshes with receiveShadow
#   set are then shadow-tested per fragment: shadowed fragments lose
#   exactly that light's diffuse (and Phong specular) contribution, with
#   a small depth bias against self-shadow acne.  Hard shadows only (no
#   PCF).  With shadows off -- or no casting light / casters / receivers
#   -- the render is pixel-identical to the unshadowed path.
# - Textures: when a mesh has a uv attribute and its
#   material has a map, texels are sampled with perspective-correct UV
#   interpolation (nearest texel, repeat wrap, flipY).  The sampled
#   texel colour is multiplied by the same flat face multiplier as the
#   untextured path (material colour x irradiance) before palette
#   quantisation, via a per-(texture, shade) cache.
#
# Everything here is duck-typed: geometry attributes may be
# plain numpy arrays or BufferAttribute-like objects (.array), matrices may
# be numpy (4,4) arrays or Matrix4-like objects (.elements, column-major as
# in three.js), colours may be Color-like (.r/.g/.b), hex ints, or sequences.

import math

import numpy as np

__all__ = ["render_scene", "PALETTE", "cube_colours"]

CUBE_BASE = 16
CUBE_LEVELS = (6, 7, 5)  # 6 * 7 * 5 = 210 colours at indices 16..225
_CUBE_LV = np.array([CUBE_LEVELS[0] - 1, CUBE_LEVELS[1] - 1, CUBE_LEVELS[2] - 1], dtype=np.float64)
_BAYER = (
    np.array(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
        dtype=np.float64,
    )
    / 16.0
    - 0.5
)


def cube_colours():
    """Return 210 0xRRGGBB ints for pyxel.colors[16:226] (6x7x5 RGB cube)."""
    out = []
    lr, lg, lb = CUBE_LEVELS
    for r in range(lr):
        rv = int(round(255 * r / (lr - 1)))
        for g in range(lg):
            gv = int(round(255 * g / (lg - 1)))
            for b in range(lb):
                bv = int(round(255 * b / (lb - 1)))
                out.append((rv << 16) | (gv << 8) | bv)
    return out


def _cube_palette_rgb():
    out = []
    lr, lg, lb = CUBE_LEVELS
    for r in range(lr):
        rv = round(255 * r / (lr - 1))
        for g in range(lg):
            gv = round(255 * g / (lg - 1))
            for b in range(lb):
                bv = round(255 * b / (lb - 1))
                out.append((rv, gv, bv))
    return out


# Pyxel default 16-colour UI palette + 210-colour RGB cube (226 colours total).
PALETTE16 = np.array(
    [
        (0, 0, 0),
        (29, 43, 83),
        (126, 37, 83),
        (0, 135, 81),
        (171, 82, 54),
        (95, 87, 79),
        (194, 195, 199),
        (255, 241, 232),
        (255, 0, 77),
        (255, 163, 0),
        (255, 236, 39),
        (0, 228, 54),
        (41, 173, 255),
        (131, 118, 156),
        (255, 119, 168),
        (255, 204, 170),
    ],
    dtype=np.float64,
)
PALETTE = np.vstack([PALETTE16, np.array(_cube_palette_rgb(), dtype=np.float64)])

_IDENTITY = np.eye(4, dtype=np.float64)

# Fallback rig used only when a Lambert mesh is rendered in a scene with no
# lights at all (direction matches the approved mock's light).
_FALLBACK_AMBIENT = (np.array([1.0, 1.0, 1.0]), 0.35)
_FALLBACK_DIRECTIONAL = (
    np.array([1.0, 1.0, 1.0]),
    0.65,
    np.array([0.55, 0.9, 0.35]),
)

# Shadow map: fixed low resolution, speed first.  The bias is
# in normalised light-depth units ([0, 1] across the fitted frustum) and
# absorbs the texel-snapping mismatch between the map pass and the main
# pass; too small shows self-shadow acne, too large leaks light under
# thin casters.
_SHADOW_MAP_SIZE = 128
_SHADOW_BIAS = 0.004


# --------------------------------------------------------------------------
# small duck-typing helpers
# --------------------------------------------------------------------------

def _get(obj, name, default=None):
    """Attribute or dict access that never raises."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_mat4(m, default=None):
    """Convert a Matrix4-like / ndarray / flat sequence to a row-major
    (4,4) float64 numpy array.  three.js `.elements` are column-major."""
    if m is None:
        return default
    # Matrix4-like object with .elements (three.js convention, column-major)
    elements = getattr(m, "elements", None)
    if elements is not None and not isinstance(m, np.ndarray):
        try:
            arr = np.asarray(elements, dtype=np.float64).ravel()
            if arr.size == 16:
                return arr.reshape(4, 4, order="F").copy()
        except Exception:
            pass
    # object exposing toArray()
    to_array = getattr(m, "toArray", None)
    if callable(to_array) and not isinstance(m, np.ndarray):
        try:
            arr = np.asarray(to_array(), dtype=np.float64).ravel()
            if arr.size == 16:
                return arr.reshape(4, 4, order="F").copy()
        except Exception:
            pass
    try:
        arr = np.asarray(m, dtype=np.float64)
    except Exception:
        return default
    if arr.shape == (4, 4):
        return arr.copy()
    if arr.size == 16:
        # A bare flat array: assume row-major numpy style.
        return arr.reshape(4, 4).copy()
    return default


def _vec3(v, default=(0.0, 0.0, 0.0)):
    """Vector3-like / sequence -> np.array shape (3,)."""
    if v is None:
        return np.array(default, dtype=np.float64)
    x = getattr(v, "x", None)
    if x is not None:
        try:
            return np.array(
                [float(x), float(getattr(v, "y")), float(getattr(v, "z"))],
                dtype=np.float64,
            )
        except Exception:
            pass
    try:
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.size >= 3:
            return arr[:3].copy()
    except Exception:
        pass
    return np.array(default, dtype=np.float64)


def _color_rgb(color, default=(1.0, 1.0, 1.0)):
    """Color-like / hex int / sequence -> np.array (3,) of floats 0..1."""
    if color is None:
        return np.array(default, dtype=np.float64)
    if isinstance(color, (int, np.integer)):
        n = int(color)
        return np.array(
            [((n >> 16) & 255) / 255.0, ((n >> 8) & 255) / 255.0, (n & 255) / 255.0]
        )
    r = getattr(color, "r", None)
    if r is not None:
        try:
            return np.array(
                [float(r), float(getattr(color, "g")), float(getattr(color, "b"))],
                dtype=np.float64,
            )
        except Exception:
            pass
    if isinstance(color, dict):
        try:
            return np.array(
                [float(color["r"]), float(color["g"]), float(color["b"])],
                dtype=np.float64,
            )
        except Exception:
            pass
    try:
        arr = np.asarray(color, dtype=np.float64).ravel()
        if arr.size >= 3:
            arr = arr[:3].copy()
            if arr.max() > 1.0:  # 0..255 style
                arr /= 255.0
            return arr
    except Exception:
        pass
    return np.array(default, dtype=np.float64)


def _attr_array(geometry, name):
    """Fetch a BufferGeometry attribute as an (N,3) float64 array or None."""
    attrs = _get(geometry, "attributes")
    raw = None
    if isinstance(attrs, dict):
        raw = attrs.get(name)
    elif attrs is not None:
        raw = getattr(attrs, name, None)
    if raw is None:
        # some geometries expose the attribute directly on the geometry
        raw = _get(geometry, name)
    if raw is None:
        return None
    arr = getattr(raw, "array", None)
    if arr is None:
        arr = raw
    try:
        arr = np.asarray(arr, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0 or arr.size % 3 != 0:
        return None
    return arr.reshape(-1, 3)


def _uv_array(geometry):
    """Fetch the 'uv' attribute as an (N,2) float64 array or None."""
    attrs = _get(geometry, "attributes")
    raw = None
    if isinstance(attrs, dict):
        raw = attrs.get("uv")
    elif attrs is not None:
        raw = getattr(attrs, "uv", None)
    if raw is None:
        raw = _get(geometry, "uv")
    if raw is None:
        return None
    arr = getattr(raw, "array", None)
    if arr is None:
        arr = raw
    try:
        arr = np.asarray(arr, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0 or arr.size % 2 != 0:
        return None
    return arr.reshape(-1, 2)


def _texture_rgb(material):
    """The material's map image as an (h, w, 3) float64 array in 0..1,
    or None when the material has no usable texture.

    The conversion is memoised on the texture object so the returned
    array identity (and the quantisation caches keyed by it) is stable
    across frames."""
    tex = _get(material, "map")
    if tex is None:
        return None
    img = _get(tex, "image")
    if img is None:
        img = tex  # a bare array was assigned as the map
    cache = _get(tex, "_rgb_cache")
    if cache is not None and cache[0] is img:
        return cache[1]
    try:
        arr = np.asarray(img, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return None
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.max() > 1.0:  # 0..255 style
        arr = arr / 255.0
    arr = np.ascontiguousarray(arr)
    try:
        tex._rgb_cache = (img, arr)
    except Exception:
        pass
    return arr


# Cache of quantised texture images: (id(image), shade bucket) ->
# (image ref, (h, w) uint8 palette-index array).  Shading is flat per
# face, so one quantised copy per (texture, shade) serves whole faces.
_TEX_CACHE = {}

# Unique-colour decomposition per texture: id(image) -> (image ref,
# unique colours, inverse indices).  Kept apart from _TEX_CACHE so a new
# shade bucket only re-shades the unique colours, not the full image.
_TEX_UNIQ = {}


def _tex_index_image(tex_rgb, mult, emis=None):
    """Palette-index image of ``tex_rgb`` multiplied by ``mult`` (the
    face's colour x irradiance, three.js map * color * light), plus
    the additive emissive term when given."""
    bucket = tuple(int(round(float(c) * 24.0)) for c in mult)
    ebucket = None
    if emis is not None:
        ebucket = tuple(int(round(float(c) * 24.0)) for c in emis)
    key = (id(tex_rgb), bucket, ebucket)
    hit = _TEX_CACHE.get(key)
    if hit is not None and hit[0] is tex_rgb:
        return hit[1]
    h, w = tex_rgb.shape[:2]
    shaded = tex_rgb * np.asarray(mult, dtype=np.float64)[None, None, :]
    if emis is not None:
        shaded = shaded + np.asarray(emis, dtype=np.float64)[None, None, :]
    shaded = np.clip(shaded, 0.0, 1.0)
    q = np.clip(np.floor(shaded * _CUBE_LV + 0.5), 0.0, _CUBE_LV).astype(np.int32)
    img_idx = (CUBE_BASE + (q[:, :, 0] * 7 + q[:, :, 1]) * 5 + q[:, :, 2]).astype(np.uint8)
    if len(_TEX_CACHE) > 512:
        _TEX_CACHE.clear()
    _TEX_CACHE[key] = (tex_rgb, img_idx)
    return img_idx


def _clip_with_uv(verts, uvs, near, far):
    """Like _clip_triangle but carries per-vertex uv through the clip.
    Returns (positions list, uvs list)."""
    poly = [(verts[i], uvs[i]) for i in range(3)]

    def clip(poly, dist_fn):
        if not poly:
            return poly
        out = []
        prev_p, prev_a = poly[-1]
        d_prev = dist_fn(prev_p)
        for cur_p, cur_a in poly:
            d_cur = dist_fn(cur_p)
            if d_cur >= 0.0:
                if d_prev < 0.0:
                    t = d_prev / (d_prev - d_cur)
                    out.append((prev_p + (cur_p - prev_p) * t,
                                prev_a + (cur_a - prev_a) * t))
                out.append((cur_p, cur_a))
            elif d_prev >= 0.0:
                t = d_prev / (d_prev - d_cur)
                out.append((prev_p + (cur_p - prev_p) * t,
                            prev_a + (cur_a - prev_a) * t))
            prev_p, prev_a, d_prev = cur_p, cur_a, d_cur
        return out

    poly = clip(poly, lambda p: -p[2] - near)
    if far and far > near:
        poly = clip(poly, lambda p: p[2] + far)
    return [p for p, _ in poly], [a for _, a in poly]


def _index_array(geometry, n_verts):
    """Fetch the triangle index as a flat int array (sequential if
    none), with the geometry's drawRange window applied."""
    raw = _get(geometry, "index")
    idx = None
    if raw is not None:
        arr = getattr(raw, "array", None)
        if arr is None:
            arr = raw
        try:
            cand = np.asarray(arr, dtype=np.int32).ravel()
            if cand.size >= 3:
                idx = cand
        except Exception:
            pass
    if idx is None:
        idx = np.arange(n_verts, dtype=np.int32)
    draw_range = _get(geometry, "drawRange")
    if isinstance(draw_range, dict):
        start = int(draw_range.get("start", 0) or 0)
        count = draw_range.get("count", float("inf"))
        if start > 0:
            idx = idx[start:]
        if count is not None and count != float("inf"):
            idx = idx[:max(0, int(count))]
    return idx[: (idx.size // 3) * 3]


def _children(obj):
    kids = _get(obj, "children", None)
    if kids is None:
        return []
    try:
        return list(kids)
    except TypeError:
        return []


def _walk(root):
    """Yield every object in the scene graph (depth-first, cycle-safe)."""
    seen = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        stack.extend(_children(obj))


def _type_name(obj):
    return type(obj).__name__ if obj is not None else ""


def _is_mesh(obj):
    if _get(obj, "isMesh", False):
        return True
    if _type_name(obj) == "Mesh":
        return True
    return _get(obj, "geometry") is not None and _get(obj, "material") is not None


def _is_points(obj):
    if _get(obj, "isPoints", False):
        return True
    return _type_name(obj) == "Points"


def _is_line(obj):
    if _get(obj, "isLineSegments", False):
        return False
    if _get(obj, "isLine", False):
        return True
    return _type_name(obj) == "Line"


def _is_line_segments(obj):
    if _get(obj, "isLineSegments", False):
        return True
    return _type_name(obj) in (
        "LineSegments", "GridHelper", "AxesHelper", "BoxHelper",
    )


def _is_sprite(obj):
    if _get(obj, "isSprite", False):
        return True
    return _type_name(obj) == "Sprite"


def _light_kind(obj):
    """'ambient' / 'directional' / 'hemisphere' / 'point' / 'spot' / None."""
    name = _type_name(obj)
    typ = str(_get(obj, "type", "") or "")
    if (
        _get(obj, "isAmbientLight", False)
        or name == "AmbientLight"
        or typ == "AmbientLight"
    ):
        return "ambient"
    if (
        _get(obj, "isDirectionalLight", False)
        or name == "DirectionalLight"
        or typ == "DirectionalLight"
    ):
        return "directional"
    if (
        _get(obj, "isHemisphereLight", False)
        or name == "HemisphereLight"
        or typ == "HemisphereLight"
    ):
        return "hemisphere"
    if (
        _get(obj, "isSpotLight", False)
        or name == "SpotLight"
        or typ == "SpotLight"
    ):
        return "spot"
    if (
        _get(obj, "isPointLight", False)
        or name == "PointLight"
        or typ == "PointLight"
    ):
        return "point"
    return None


def _distance_attenuation(dist, cutoff, decay, legacy=False):
    """three.js getDistanceAttenuation: inverse power falloff, with a
    smooth window to zero at ``cutoff`` when a cutoff distance is set
    (distance == 0 means no cutoff).

    ``legacy`` selects the pre-r155 (legacy lights) branch of the
    same shader chunk, as bundled e.g. in the manual's r132 build:
    with no cutoff distance the attenuation is 1.0 (no falloff at
    all), and with a cutoff it is pow(saturate(1 - d / cutoff),
    decay).  Sample code of that generation (PointLight intensity
    1-3 at scene scales of 5-15 units) is written against it; under
    the physical branch alone it quantises to black on a 16-colour
    palette."""
    if legacy:
        if cutoff > 0.0 and decay > 0.0:
            return np.power(
                np.clip(1.0 - dist / cutoff, 0.0, 1.0), decay)
        return np.ones_like(np.asarray(dist, dtype=np.float64))
    atten = 1.0 / np.maximum(np.power(dist, decay), 0.01)
    if cutoff > 0.0:
        window = np.clip(1.0 - np.power(dist / cutoff, 4.0), 0.0, 1.0)
        atten = atten * window * window
    return atten


def _spot_factor(angle_cos, cos_outer, cos_inner):
    """three.js getSpotAttenuation: smoothstep from the outer cone edge
    (cos(angle)) to the inner edge (cos(angle * (1 - penumbra))).  With
    penumbra == 0 the edges coincide and the factor is a hard step."""
    if cos_inner <= cos_outer:
        return np.where(angle_cos >= cos_outer, 1.0, 0.0)
    t = np.clip(
        (angle_cos - cos_outer) / (cos_inner - cos_outer), 0.0, 1.0
    )
    return t * t * (3.0 - 2.0 * t)


def _material_kind(obj):
    """'standard' / 'phong' / 'normal' / 'lambert' / 'basic'
    (unknown -> basic)."""
    name = _type_name(obj)
    typ = str(_get(obj, "type", "") or "")
    if (
        _get(obj, "isMeshStandardMaterial", False)
        or "standard" in name.lower()
        or "standard" in typ.lower()
    ):
        return "standard"
    if (
        _get(obj, "isMeshPhongMaterial", False)
        or "phong" in name.lower()
        or "phong" in typ.lower()
    ):
        return "phong"
    if (
        _get(obj, "isMeshNormalMaterial", False)
        or "normal" in name.lower()
        or "normal" in typ.lower()
    ):
        return "normal"
    if (
        _get(obj, "isMeshLambertMaterial", False)
        or "lambert" in name.lower()
        or "lambert" in typ.lower()
    ):
        return "lambert"
    return "basic"


# --------------------------------------------------------------------------
# matrices
# --------------------------------------------------------------------------

def _local_matrix(obj):
    """Compose T * R * S from .position/.rotation/.scale if present."""
    pos = _get(obj, "position")
    rot = _get(obj, "rotation")
    scl = _get(obj, "scale")
    if pos is None and rot is None and scl is None:
        return None
    p = _vec3(pos, (0.0, 0.0, 0.0))
    s = _vec3(scl, (1.0, 1.0, 1.0))
    if rot is None:
        rx = ry = rz = 0.0
    elif hasattr(rot, "x"):
        rx, ry, rz = float(rot.x), float(rot.y), float(rot.z)
    else:
        e = _vec3(rot, (0.0, 0.0, 0.0))
        rx, ry, rz = float(e[0]), float(e[1]), float(e[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    # three.js Euler order 'XYZ'
    r = np.array(
        [
            [cy * cz, -cy * sz, sy],
            [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
            [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy],
        ],
        dtype=np.float64,
    )
    m = np.eye(4)
    m[:3, :3] = r * s  # column scaling
    m[:3, 3] = p
    return m


def _world_matrix(obj):
    """Prefer .matrixWorld; fall back to composing the parent chain."""
    mw = _as_mat4(_get(obj, "matrixWorld"))
    parent = _get(obj, "parent")
    if mw is not None and parent is None:
        return mw
    # If a parent chain exists, composing locals is the reliable route
    # (matrixWorld may be stale or identity in duck-typed objects).
    chain = []
    node = obj
    composed_ok = True
    seen = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        local = _local_matrix(node)
        if local is None:
            node_mw = _as_mat4(_get(node, "matrixWorld"))
            if node_mw is None:
                composed_ok = False
                break
            local = node_mw
            chain.append(local)
            break  # matrixWorld already includes its own parents
        chain.append(local)
        node = _get(node, "parent")
    if composed_ok and chain:
        m = _IDENTITY.copy()
        for local in reversed(chain):
            m = m @ local
        return m
    if mw is not None:
        return mw
    return _IDENTITY.copy()


def _perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(float(fov_deg)) / 2.0)
    near = float(near) if near else 0.1
    far = float(far) if far else 2000.0
    if far <= near:
        far = near + 1.0
    m = np.zeros((4, 4))
    m[0, 0] = f / float(aspect)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _camera_matrices(camera, width, height):
    """Return (view, projection, camera_world_position)."""
    aspect = _get(camera, "aspect")
    if not aspect:
        aspect = width / height if height else 1.0
    near = _get(camera, "near", 0.1)
    far = _get(camera, "far", 2000.0)
    fov = _get(camera, "fov", 50.0)

    proj = _as_mat4(_get(camera, "projectionMatrix"))
    if proj is None or np.allclose(proj, _IDENTITY):
        proj = _perspective(fov, aspect, near, far)

    view = _as_mat4(_get(camera, "matrixWorldInverse"))
    if view is None or np.allclose(view, _IDENTITY):
        cam_world = _as_mat4(_get(camera, "matrixWorld"))
        pos = _vec3(_get(camera, "position"), (0.0, 0.0, 0.0))
        if cam_world is not None and not np.allclose(cam_world, _IDENTITY):
            view = np.linalg.inv(cam_world)
        elif np.any(pos != 0.0):
            # no world matrix: the camera looks toward the origin
            forward = -pos
            n = np.linalg.norm(forward)
            if n < 1e-12:
                view = _IDENTITY.copy()
                view[:3, 3] = -pos
            else:
                fwd = forward / n
                up = np.array([0.0, 1.0, 0.0])
                if abs(float(fwd @ up)) > 0.999:
                    up = np.array([1.0, 0.0, 0.0])
                right = np.cross(up, -fwd)
                right /= np.linalg.norm(right) + 1e-12
                true_up = np.cross(-fwd, right)
                view = np.eye(4)
                view[0, :3] = right
                view[1, :3] = true_up
                view[2, :3] = -fwd
                view[:3, 3] = -view[:3, :3] @ pos
        else:
            view = _IDENTITY.copy()

    cam_world = _as_mat4(_get(camera, "matrixWorld"))
    if cam_world is not None and not np.allclose(cam_world, _IDENTITY):
        cam_pos = cam_world[:3, 3].copy()
    else:
        cam_pos = _vec3(_get(camera, "position"), (0.0, 0.0, 0.0))
    return view, proj, cam_pos


# --------------------------------------------------------------------------
# shadow map
# --------------------------------------------------------------------------

def _look_at_view(eye, center):
    """View matrix looking from ``eye`` toward ``center`` (world -> view,
    camera down -Z), or None when the two points coincide."""
    forward = np.asarray(center, dtype=np.float64) - np.asarray(
        eye, dtype=np.float64
    )
    n = np.linalg.norm(forward)
    if n < 1e-12:
        return None
    fwd = forward / n
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(fwd @ up)) > 0.999:
        up = np.array([1.0, 0.0, 0.0])
    right = np.cross(up, -fwd)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(-fwd, right)
    view = np.eye(4)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -fwd
    view[:3, 3] = -view[:3, :3] @ np.asarray(eye, dtype=np.float64)
    return view


def _ortho_proj(left, right, bottom, top, near, far):
    """OpenGL-style orthographic projection (NDC in [-1, 1])."""
    m = np.zeros((4, 4))
    m[0, 0] = 2.0 / (right - left)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


def _mesh_world_tris(mesh):
    """(world positions (N,3), triangles (M,3)) for one mesh, applying
    the same validity filtering as the main pass; (None, None) when the
    mesh has nothing drawable."""
    geometry = _get(mesh, "geometry")
    if geometry is None:
        return None, None
    positions = _attr_array(geometry, "position")
    if positions is None or len(positions) == 0:
        return None, None
    indices = _index_array(geometry, len(positions))
    if indices.size < 3:
        return None, None
    tris = indices.reshape(-1, 3)
    valid = np.all((tris >= 0) & (tris < len(positions)), axis=1)
    tris = tris[valid]
    if len(tris) == 0:
        return None, None
    world = _world_matrix(mesh)
    world_pos = positions @ world[:3, :3].T + world[:3, 3]
    return world_pos, tris


def _build_shadow_map(caster_data, receiver_data, light_pos, light_target):
    """Render the depth map for one directional light.

    caster_data: list of (world_pos, tris, side); receiver_data: list
    of (world_pos, tris).  Returns (depth_map, light_vp) where
    depth_map is a (_SHADOW_MAP_SIZE,)^2 float array of normalised
    depths (1.0 = nothing stored) and light_vp maps world positions to
    light clip space, or None when nothing can cast or receive.

    FrontSide casters skip triangles facing away from the light: for a
    closed mesh the surface nearest the light is always light-facing,
    so the stored minima are unchanged."""
    if not caster_data or not receiver_data:
        return None
    view = _look_at_view(light_pos, light_target)
    if view is None:
        return None
    all_pos = np.concatenate(
        [wp for wp, _, _ in caster_data] + [wp for wp, _ in receiver_data]
    )
    lv = all_pos @ view[:3, :3].T + view[:3, 3]
    lo = lv.min(axis=0)
    hi = lv.max(axis=0)
    pad_xy = 0.05 * max(hi[0] - lo[0], hi[1] - lo[1], 1e-6) + 1e-4
    depth_lo = -hi[2]  # nearest surface distance from the light
    depth_hi = -lo[2]  # farthest
    if depth_hi <= 0.0:
        return None
    near = max(depth_lo - 0.05 * max(depth_hi - depth_lo, 1e-6) - 1e-4, 1e-3)
    far = depth_hi + 0.05 * max(depth_hi - depth_lo, 1e-6) + 1e-4
    proj = _ortho_proj(
        lo[0] - pad_xy, hi[0] + pad_xy,
        lo[1] - pad_xy, hi[1] + pad_xy,
        near, far,
    )
    vp = proj @ view
    size = _SHADOW_MAP_SIZE
    smap = np.ones((size, size), dtype=np.float64)
    light_pos = np.asarray(light_pos, dtype=np.float64)
    for world_pos, tris, side in caster_data:
        if side == 0 and len(tris):
            wp_tris = world_pos[tris]
            e1 = wp_tris[:, 1] - wp_tris[:, 0]
            e2 = wp_tris[:, 2] - wp_tris[:, 0]
            nrm = np.stack(
                [
                    e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1],
                    e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2],
                    e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0],
                ],
                axis=1,
            )
            facing = np.sum(
                nrm * (light_pos[None, :] - wp_tris.mean(axis=1)), axis=1
            )
            tris = tris[facing > 0.0]
            if len(tris) == 0:
                continue
        ones = np.ones((len(world_pos), 1))
        clip = np.concatenate([world_pos, ones], axis=1) @ vp.T
        ndc = clip[:, :3]  # orthographic: w stays 1
        scr = np.stack(
            [
                (ndc[:, 0] * 0.5 + 0.5) * size,
                (-ndc[:, 1] * 0.5 + 0.5) * size,
            ],
            axis=1,
        )
        dep = ndc[:, 2] * 0.5 + 0.5
        tri_scr = scr[tris]
        tri_dep = dep[tris]
        for t in range(len(tris)):
            A, B, C = tri_scr[t]
            x0 = max(int(np.floor(min(A[0], B[0], C[0]))), 0)
            x1 = min(int(np.ceil(max(A[0], B[0], C[0]))), size - 1)
            y0 = max(int(np.floor(min(A[1], B[1], C[1]))), 0)
            y1 = min(int(np.ceil(max(A[1], B[1], C[1]))), size - 1)
            if x0 > x1 or y0 > y1:
                continue
            denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (
                A[1] - C[1]
            )
            if abs(denom) < 1e-12:
                continue
            xs = np.arange(x0, x1 + 1, dtype=np.float64)[None, :] + 0.5
            ys = np.arange(y0, y1 + 1, dtype=np.float64)[:, None] + 0.5
            w0 = ((B[1] - C[1]) * (xs - C[0])
                  + (C[0] - B[0]) * (ys - C[1])) / denom
            w1 = ((C[1] - A[1]) * (xs - C[0])
                  + (A[0] - C[0]) * (ys - C[1])) / denom
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            if not np.any(inside):
                continue
            d = (w0 * tri_dep[t, 0] + w1 * tri_dep[t, 1]
                 + w2 * tri_dep[t, 2])
            sub = smap[y0 : y1 + 1, x0 : x1 + 1]
            upd = inside & (d < sub)
            sub[upd] = d[upd]
    return smap, vp


def _shadow_mask(w0, w1, w2, iw, iw3, sh3, smap):
    """Boolean grid: True where the fragment is shadowed.

    sh3: the triangle's vertices in light NDC (x, y, z).  The light NDC
    is affine in world position, so perspective-correct interpolation
    against the camera's 1/w (iw3 / iw) reproduces it exactly."""
    a = sh3 * iw3[:, None]  # per-vertex light NDC weighted by 1/w
    l = (w0[:, :, None] * a[0] + w1[:, :, None] * a[1]
         + w2[:, :, None] * a[2]) / iw[:, :, None]
    size = smap.shape[0]
    mx = (l[:, :, 0] * 0.5 + 0.5) * size
    my = (-l[:, :, 1] * 0.5 + 0.5) * size
    lz01 = l[:, :, 2] * 0.5 + 0.5
    tx = np.floor(mx).astype(np.int32)
    ty = np.floor(my).astype(np.int32)
    valid = (
        (tx >= 0) & (tx < size) & (ty >= 0) & (ty < size)
        & (lz01 >= 0.0) & (lz01 <= 1.0)
    )
    stored = smap[
        np.clip(ty, 0, size - 1), np.clip(tx, 0, size - 1)
    ]
    return valid & (lz01 > stored + _SHADOW_BIAS)


# --------------------------------------------------------------------------
# palette mapping
# --------------------------------------------------------------------------

def _nearest_palette(rgb255):
    """rgb (3,) in 0..255 -> (nearest index, second-nearest index)."""
    d2 = np.sum((PALETTE - rgb255) ** 2, axis=1)
    order = np.argsort(d2)
    return int(order[0]), int(order[1])


def _palette_index(rgb01, dither_xy=None):
    """Map a 0..1 RGB colour to a 210-colour cube index (4x4 Bayer optional)."""
    c = np.clip(np.asarray(rgb01, dtype=np.float64), 0.0, 1.0)
    dz = _BAYER[dither_xy[1] & 3, dither_xy[0] & 3] if dither_xy is not None else 0.0
    q = np.clip(np.floor(c * _CUBE_LV + 0.5 + dz), 0.0, _CUBE_LV).astype(np.int32)
    return int(CUBE_BASE + (q[0] * 7 + q[1]) * 5 + q[2])


# --------------------------------------------------------------------------
# fog / side helpers and the shared triangle fill
# --------------------------------------------------------------------------

def _float_or(value, default):
    """float(value) with a fallback when value is None / unparseable."""
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _material_side(material):
    """Material.side as an int: 0=FrontSide, 1=BackSide, 2=DoubleSide.
    Missing / unrecognised values fall back to FrontSide (the three.js
    default)."""
    raw = _get(material, "side", 0)
    try:
        side = int(raw)
    except Exception:
        return 0
    return side if side in (0, 1, 2) else 0


def _fog_context(scene):
    """(fog_rgb, near, far) when scene.fog looks like a THREE.Fog or
    THREE.FogExp2, else None."""
    fog = _get(scene, "fog")
    if fog is None:
        return None
    if _get(fog, "isFogExp2", False):
        density = max(_float_or(_get(fog, "density"), 0.00025), 1e-6)
        return (_color_rgb(_get(fog, "color")), 0.0, 2.0 / density)
    near = _float_or(_get(fog, "near"), 1.0)
    far = _float_or(_get(fog, "far"), 2000.0)
    if far <= near:
        return None
    return (_color_rgb(_get(fog, "color")), near, far)


def _env_sampler(env):
    """A direction sampler for an environment map, or None.

    The primary path is the object's own ``sample_rgb`` (our
    CubeTexture).  Duck-typed fallback: a ``faces`` list of six solid
    colours (three.js order [+x, -x, +y, -y, +z, -z]; image faces are
    averaged to their mean colour) sampled by major axis.  The returned
    callable takes an (N, 3) array of directions and returns (N, 3)
    float RGB in 0..1 (or None on failure)."""
    if env is None:
        return None
    sample = _get(env, "sample_rgb")
    if callable(sample):
        def _via_method(dirs):
            try:
                out = np.asarray(sample(dirs), dtype=np.float64)
                return out.reshape(dirs.shape)
            except Exception:
                return None
        return _via_method
    faces = _get(env, "faces")
    if faces is not None:
        try:
            face_list = list(faces)
        except Exception:
            return None
        if len(face_list) != 6:
            return None
        cols = []
        for face in face_list:
            arr = None
            if face is not None and not isinstance(
                face, (int, np.integer)
            ) and getattr(face, "r", None) is None:
                try:
                    cand = np.asarray(face, dtype=np.float64)
                    if cand.ndim == 3 and cand.size:
                        arr = cand.reshape(-1, cand.shape[2])[:, :3].mean(axis=0)
                        if arr.max() > 1.0:
                            arr = arr / 255.0
                except Exception:
                    arr = None
            cols.append(_color_rgb(face) if arr is None else arr)
        face_rgb = np.array(cols, dtype=np.float64)

        def _via_faces(dirs):
            ax = np.argmax(np.abs(dirs), axis=1)
            comp = dirs[np.arange(len(dirs)), ax]
            face_i = 2 * ax + (comp < 0.0)
            return face_rgb[face_i]
        return _via_faces
    return None


def _flat_fill_info(face_rgb):
    """(representative index, 4x4 Bayer tile) for one flat face in the
    6x7x5 colour cube."""
    c = np.clip(np.asarray(face_rgb, dtype=np.float64), 0.0, 1.0)
    q = np.clip(
        np.floor(c[None, None, :] * _CUBE_LV + 0.5 + _BAYER[:, :, None]),
        0.0,
        _CUBE_LV,
    ).astype(np.int32)
    tile = (CUBE_BASE + (q[:, :, 0] * 7 + q[:, :, 1]) * 5 + q[:, :, 2]).astype(np.uint8)
    return int(tile[0, 0]), tile


def _quantize_into(sub_img, rgb01, upd, x0, y0):
    """Per-pixel 4x4 Bayer cube quantisation of an (h, w, 3) RGB array (0..1)."""
    bh, bw = rgb01.shape[:2]
    ys = np.arange(y0, y0 + bh, dtype=np.int32)[:, None] & 3
    xs = np.arange(x0, x0 + bw, dtype=np.int32)[None, :] & 3
    dz = _BAYER[ys, xs][..., None]
    c = np.clip(rgb01, 0.0, 1.0)
    q = np.clip(np.floor(c * _CUBE_LV + 0.5 + dz), 0.0, _CUBE_LV).astype(np.int32)
    out = (CUBE_BASE + (q[:, :, 0] * 7 + q[:, :, 1]) * 5 + q[:, :, 2]).astype(np.uint8)
    sub_img[upd] = out[upd]


def _fill_flat(sub_img, mask, index, second, x0, x1, y0, y1):
    """Write one flat face over ``mask`` using its 4x4 Bayer tile when provided."""
    if not np.any(mask):
        return
    if second is None:
        sub_img[mask] = index
    elif isinstance(second, np.ndarray):
        ys = np.arange(y0, y1 + 1, dtype=np.int32)[:, None] & 3
        xs = np.arange(x0, x1 + 1, dtype=np.int32)[None, :] & 3
        sub_img[mask] = second[ys, xs][mask]
    else:
        parity = (
            (np.arange(x0, x1 + 1, dtype=np.int32)[None, :]
             + np.arange(y0, y1 + 1, dtype=np.int32)[:, None]) % 2
        ) == 0
        sub_img[mask & parity] = second
        sub_img[mask & ~parity] = 0


def _raster_fill(img, zbuf, scr3, dep3, iw3, fd3, uv3,
                 tex_rgb, face_rgb, flat_index, dither_second,
                 fog_ctx, width, height,
                 sh3=None, smap=None, face_rgb_sh=None, emis_rgb=None):
    """Fill one screen-space triangle.

    scr3/dep3/iw3/fd3: the three vertices' screen positions, NDC depths,
    1/w values and view fog depths (-z in view space).  uv3: (3,2) uvs or
    None.  Colour source: a texture (tex_rgb, with face_rgb as the
    colour x irradiance multiplier) when tex_rgb is not None, else the
    flat face_rgb via (flat_index, dither_second).  fog_ctx blends the
    per-pixel RGB toward the fog colour before quantisation.

    Shadow: when sh3 (the vertices' light-space NDC) and smap
    (the shadow depth map) are given, fragments failing the shadow test
    are drawn with the *_sh colour set instead (the same face with the
    shadow-casting light's contribution removed).
    """
    A = scr3[0]
    B = scr3[1]
    C = scr3[2]
    x0 = max(int(np.floor(min(A[0], B[0], C[0]))), 0)
    x1 = min(int(np.ceil(max(A[0], B[0], C[0]))), width - 1)
    y0 = max(int(np.floor(min(A[1], B[1], C[1]))), 0)
    y1 = min(int(np.ceil(max(A[1], B[1], C[1]))), height - 1)
    if x0 > x1 or y0 > y1:
        return
    denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    if abs(denom) < 1e-12:
        return
    xs = np.arange(x0, x1 + 1, dtype=np.float64)[None, :] + 0.5
    ys = np.arange(y0, y1 + 1, dtype=np.float64)[:, None] + 0.5
    w0 = ((B[1] - C[1]) * (xs - C[0]) + (C[0] - B[0]) * (ys - C[1])) / denom
    w1 = ((C[1] - A[1]) * (xs - C[0]) + (A[0] - C[0]) * (ys - C[1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    if not np.any(inside):
        return
    # NDC depth: affine in screen space (clip z and clip w are both
    # view-affine, and their ratio over one triangle is a plane in
    # screen coordinates), so it interpolates with the plain screen
    # barycentrics.  The 1/w-weighted formula -- correct for the
    # view-affine attributes below (uv, fog depth, light NDC) --
    # systematically distorts depth and flips the overlap order
    # between triangles with very different depth ranges (a small mesh
    # standing on a huge ground plane was drawn underneath it).
    iw = w0 * iw3[0] + w1 * iw3[1] + w2 * iw3[2]
    z = w0 * dep3[0] + w1 * dep3[1] + w2 * dep3[2]
    sub_z = zbuf[y0 : y1 + 1, x0 : x1 + 1]
    upd = inside & (z < sub_z)
    if not np.any(upd):
        return
    sub_z[upd] = z[upd]
    sub_img = img[y0 : y1 + 1, x0 : x1 + 1]

    shadowed = None
    if sh3 is not None and smap is not None:
        shadowed = _shadow_mask(w0, w1, w2, iw, iw3, sh3, smap)

    if fog_ctx is None and tex_rgb is None:
        # flat colour (flat path, incl. the checker dither rule)
        if shadowed is None:
            _fill_flat(sub_img, upd, flat_index, dither_second,
                       x0, x1, y0, y1)
        else:
            _fill_flat(sub_img, upd & ~shadowed, flat_index,
                       dither_second, x0, x1, y0, y1)
            shd = upd & shadowed
            if np.any(shd):
                # lazily: fully lit triangles never pay for the
                # shadow variant's palette info
                sh_rgb = face_rgb_sh
                if emis_rgb is not None:
                    # emissive is not shadowed in three.js either
                    sh_rgb = sh_rgb + emis_rgb
                fi_sh, ds_sh = _flat_fill_info(sh_rgb)
                _fill_flat(sub_img, shd, fi_sh, ds_sh, x0, x1, y0, y1)
        return

    if tex_rgb is not None:
        # Perspective-correct UV sampling (nearest texel, repeat wrap,
        # flipY).  Without fog the pre-quantised index image is used
        # (textured path); with fog the raw texel RGB is needed.
        ua, ub, uc = uv3[0], uv3[1], uv3[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (w0 * ua[0] * iw3[0] + w1 * ub[0] * iw3[1]
                 + w2 * uc[0] * iw3[2]) / iw
            v = (w0 * ua[1] * iw3[0] + w1 * ub[1] * iw3[1]
                 + w2 * uc[1] * iw3[2]) / iw
        uu = u - np.floor(u)  # repeat wrap
        vv = v - np.floor(v)
        if fog_ctx is None:
            idx_img = _tex_index_image(tex_rgb, face_rgb, emis_rgb)
            th, tw = idx_img.shape
            uu = np.nan_to_num(uu, nan=0.0, posinf=0.0, neginf=0.0)
            vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
            cols = (uu * tw).astype(np.int32)
            np.clip(cols, 0, tw - 1, out=cols)
            rows = (th - 1) - (vv * th).astype(np.int32)
            np.clip(rows, 0, th - 1, out=rows)
            sampled = idx_img[rows, cols]
            if shadowed is None:
                sub_img[upd] = sampled[upd]
            else:
                idx_img_sh = _tex_index_image(
                    tex_rgb, face_rgb_sh, emis_rgb)
                sampled_sh = idx_img_sh[rows, cols]
                lit = upd & ~shadowed
                shd = upd & shadowed
                sub_img[lit] = sampled[lit]
                sub_img[shd] = sampled_sh[shd]
            return
        th, tw = tex_rgb.shape[:2]
        uu = np.nan_to_num(uu, nan=0.0, posinf=0.0, neginf=0.0)
        vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
        cols = (uu * tw).astype(np.int32)
        np.clip(cols, 0, tw - 1, out=cols)
        rows = (th - 1) - (vv * th).astype(np.int32)
        np.clip(rows, 0, th - 1, out=rows)
        if shadowed is None:
            rgb = tex_rgb[rows, cols] * np.asarray(face_rgb)[None, None, :]
        else:
            mult = np.where(
                shadowed[:, :, None],
                np.asarray(face_rgb_sh)[None, None, :],
                np.asarray(face_rgb)[None, None, :],
            )
            rgb = tex_rgb[rows, cols] * mult
        if emis_rgb is not None:
            # emissive is additive, after the texel multiply
            rgb = rgb + np.asarray(emis_rgb)[None, None, :]
    else:
        if shadowed is None:
            rgb = np.broadcast_to(np.asarray(face_rgb, dtype=np.float64),
                                  (y1 - y0 + 1, x1 - x0 + 1, 3))
        else:
            sh_rgb = np.asarray(face_rgb_sh, dtype=np.float64)
            if emis_rgb is not None:
                # face_rgb already carries the emissive term on this
                # path; the shadow variant does not yet
                sh_rgb = sh_rgb + emis_rgb
            rgb = np.where(
                shadowed[:, :, None],
                sh_rgb[None, None, :],
                np.asarray(face_rgb, dtype=np.float64)[None, None, :],
            )

    # fog: blend toward the fog colour by view depth, then quantise
    fog_rgb, fog_near, fog_far = fog_ctx
    with np.errstate(divide="ignore", invalid="ignore"):
        fdepth = (
            w0 * fd3[0] * iw3[0]
            + w1 * fd3[1] * iw3[1]
            + w2 * fd3[2] * iw3[2]
        ) / iw
    f = np.clip((fdepth - fog_near) / (fog_far - fog_near), 0.0, 1.0)
    rgb = rgb * (1.0 - f)[:, :, None] + fog_rgb[None, None, :] * f[:, :, None]
    _quantize_into(sub_img, rgb, upd, x0, y0)


# --------------------------------------------------------------------------
# triangle clipping (view space, camera looks down -Z)
# --------------------------------------------------------------------------

def _clip_triangle(verts, near, far):
    """Sutherland-Hodgman against near (z <= -near) and far (z >= -far)
    planes in view space.  verts: (3,3).  Returns a list of (3,) points
    forming a convex polygon (possibly empty)."""
    poly = [verts[0], verts[1], verts[2]]

    def clip(poly, dist_fn):
        if not poly:
            return poly
        out = []
        prev = poly[-1]
        d_prev = dist_fn(prev)
        for cur in poly:
            d_cur = dist_fn(cur)
            if d_cur >= 0.0:
                if d_prev < 0.0:
                    t = d_prev / (d_prev - d_cur)
                    out.append(prev + (cur - prev) * t)
                out.append(cur)
            elif d_prev >= 0.0:
                t = d_prev / (d_prev - d_cur)
                out.append(prev + (cur - prev) * t)
            prev, d_prev = cur, d_cur
        return out

    # inside near plane: -z - near >= 0  (points have z <= -near)
    poly = clip(poly, lambda p: -p[2] - near)
    if far and far > near:
        # inside far plane: z + far >= 0  (points have z >= -far)
        poly = clip(poly, lambda p: p[2] + far)
    return poly


# --------------------------------------------------------------------------
# points / lines
# --------------------------------------------------------------------------

def _raw_index(geometry, n_verts):
    """Index array for points/lines (no multiple-of-3 truncation, unlike
    the triangle helper).  Falls back to the sequential order."""
    raw = _get(geometry, "index")
    if raw is not None:
        arr = getattr(raw, "array", None)
        if arr is None:
            arr = raw
        try:
            idx = np.asarray(arr, dtype=np.int32).ravel()
            if idx.size >= 1:
                return idx
        except Exception:
            pass
    return np.arange(n_verts, dtype=np.int32)


def _project_points(view_pos, proj, width, height):
    """(N,3) view-space points -> (screen (N,2), ndc z (N,), inv w (N,))."""
    ones = np.ones((len(view_pos), 1))
    clip = np.concatenate([view_pos, ones], axis=1) @ proj.T
    w = clip[:, 3]
    safe_w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    ndc = clip[:, :3] / safe_w[:, None]
    scr = np.stack(
        [
            (ndc[:, 0] * 0.5 + 0.5) * width,
            (-ndc[:, 1] * 0.5 + 0.5) * height,
        ],
        axis=1,
    )
    return scr, ndc[:, 2], 1.0 / safe_w


def _draw_segment(img, zbuf, a_v, b_v, proj, width, height, near, far, colour,
                  fog_ctx=None, base_rgb=None, base_rgb_b=None):
    """Rasterise one view-space segment: parametric clip against the
    near/far z planes, then a per-pixel walk with the NDC depth
    interpolated plainly in the screen parameter (it is affine there,
    see _raster_fill), tested against (and written into) the shared
    z-buffer.  With
    fog_ctx, the palette index is recomputed per pixel from base_rgb
    blended toward the fog colour by view depth.  With base_rgb_b
    (vertex colours) the colour is interpolated between the
    two endpoint RGBs along the segment before quantisation (and
    before the fog blend when both apply)."""
    a = np.asarray(a_v, dtype=np.float64)
    b = np.asarray(b_v, dtype=np.float64)
    t0, t1 = 0.0, 1.0
    dz = b[2] - a[2]
    # keep -far <= z <= -near
    for bound, is_upper in ((-near, True), (-far, False)):
        if abs(dz) < 1e-12:
            outside = (a[2] > bound) if is_upper else (a[2] < bound)
            if outside:
                return
            continue
        t_hit = (bound - a[2]) / dz
        if is_upper:  # keep z <= bound
            if dz > 0:
                t1 = min(t1, t_hit)
            else:
                t0 = max(t0, t_hit)
        else:  # keep z >= bound
            if dz > 0:
                t0 = max(t0, t_hit)
            else:
                t1 = min(t1, t_hit)
    if t0 > t1:
        return
    ends = np.stack([a + (b - a) * t0, a + (b - a) * t1])
    scr, depth, inv_w = _project_points(ends, proj, width, height)
    (x0, y0), (x1, y1) = scr
    steps = int(max(abs(x1 - x0), abs(y1 - y0)))
    if steps < 1:
        steps = 1
    ez0, ez1 = float(ends[0][2]), float(ends[1][2])
    for s in range(steps + 1):
        t = s / steps
        iw = inv_w[0] + (inv_w[1] - inv_w[0]) * t
        if abs(iw) < 1e-12:
            continue
        # NDC depth is affine in the screen parameter (see _raster_fill)
        z = depth[0] + (depth[1] - depth[0]) * t
        x = int(np.floor(x0 + (x1 - x0) * t))
        y = int(np.floor(y0 + (y1 - y0) * t))
        if 0 <= x < width and 0 <= y < height and z < zbuf[y, x]:
            zbuf[y, x] = z
            if base_rgb is not None and (
                fog_ctx is not None or base_rgb_b is not None
            ):
                rgb = base_rgb
                if base_rgb_b is not None:
                    rgb = base_rgb * (1.0 - t) + base_rgb_b * t
                if fog_ctx is not None:
                    fog_rgb, fog_near, fog_far = fog_ctx
                    fdepth = -(ez0 + (ez1 - ez0) * t)
                    f = (fdepth - fog_near) / (fog_far - fog_near)
                    f = 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)
                    rgb = rgb * (1.0 - f) + fog_rgb * f
                img[y, x] = _palette_index(rgb)
            else:
                img[y, x] = colour


def _draw_primitives(primitives, img, zbuf, view, proj, width, height, near, far,
                     fog_ctx=None):
    """Draw collected Points / Line / LineSegments objects into ``img``
    using the mesh pass's z-buffer (they test and write depth, so
    meshes occlude them and they occlude each other in scene order).
    Materials with vertexColors use the geometry's 'color' attribute:
    per-point colours for Points, interpolated endpoint colours for
    lines and segments."""
    for kind, obj in primitives:
        geometry = _get(obj, "geometry")
        material = _get(obj, "material")
        if geometry is None or material is None:
            continue
        if _get(material, "visible", True) is False:
            continue
        positions = _attr_array(geometry, "position")
        if positions is None or len(positions) == 0:
            continue
        order = _raw_index(geometry, len(positions))
        order = order[(order >= 0) & (order < len(positions))]
        draw_range = _get(geometry, "drawRange")
        if isinstance(draw_range, dict):
            start = int(draw_range.get("start", 0) or 0)
            count = draw_range.get("count", float("inf"))
            if start > 0:
                order = order[start:]
            if count is not None and count != float("inf"):
                order = order[:max(0, int(count))]
        if len(order) == 0:
            continue
        pts = positions[order]
        world = _world_matrix(obj)
        world_pos = pts @ world[:3, :3].T + world[:3, 3]
        view_pos = world_pos @ view[:3, :3].T + view[:3, 3]
        base_rgb = _color_rgb(_get(material, "color"))
        colour = _palette_index(base_rgb)
        # Vertex colours: the geometry's 'color' attribute,
        # used when the material opts in with vertexColors and expanded
        # through the same index order as the positions.
        cols = None
        if _get(material, "vertexColors", False):
            colors_attr = _attr_array(geometry, "color")
            if colors_attr is not None and len(colors_attr) == len(positions):
                cols = colors_attr[order]

        if kind == "points":
            try:
                size = float(_get(material, "size", 1.0))
            except Exception:
                size = 1.0
            attenuate = _get(material, "sizeAttenuation", True)
            scr, depth, _inv_w = _project_points(view_pos, proj, width, height)
            for i in range(len(view_pos)):
                vz = view_pos[i, 2]
                if vz > -near or vz < -far:
                    continue
                if attenuate:
                    # three.js: gl_PointSize = size * (height/2 / -mvZ)
                    px = size * (height * 0.5) / max(-vz, 1e-3)
                    px = int(min(16, max(1, round(px))))
                else:
                    px = int(min(16, max(1, round(size))))
                cx = int(np.floor(scr[i, 0]))
                cy = int(np.floor(scr[i, 1]))
                xa = max(cx - px // 2, 0)
                xb = min(cx - px // 2 + px, width)
                ya = max(cy - px // 2, 0)
                yb = min(cy - px // 2 + px, height)
                if xa >= xb or ya >= yb:
                    continue
                sub_z = zbuf[ya:yb, xa:xb]
                upd = depth[i] < sub_z
                if np.any(upd):
                    sub_z[upd] = depth[i]
                    sub_img = img[ya:yb, xa:xb]
                    src_rgb = cols[i] if cols is not None else base_rgb
                    if fog_ctx is not None:
                        fog_rgb, fog_near, fog_far = fog_ctx
                        f = (-vz - fog_near) / (fog_far - fog_near)
                        f = 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)
                        sub_img[upd] = _palette_index(
                            src_rgb * (1.0 - f) + fog_rgb * f
                        )
                    elif cols is not None:
                        sub_img[upd] = _palette_index(src_rgb)
                    else:
                        sub_img[upd] = colour
        elif kind == "line_segments":
            # independent pairs (0, 1), (2, 3), ...
            for i in range(0, len(view_pos) - 1, 2):
                _draw_segment(
                    img,
                    zbuf,
                    view_pos[i],
                    view_pos[i + 1],
                    proj,
                    width,
                    height,
                    near,
                    far,
                    _palette_index(cols[i]) if cols is not None else colour,
                    fog_ctx,
                    cols[i] if cols is not None else base_rgb,
                    cols[i + 1] if cols is not None else None,
                )
        else:  # "line": a strip through the (possibly indexed) vertices
            for i in range(len(view_pos) - 1):
                _draw_segment(
                    img,
                    zbuf,
                    view_pos[i],
                    view_pos[i + 1],
                    proj,
                    width,
                    height,
                    near,
                    far,
                    _palette_index(cols[i]) if cols is not None else colour,
                    fog_ctx,
                    cols[i] if cols is not None else base_rgb,
                    cols[i + 1] if cols is not None else None,
                )


# --------------------------------------------------------------------------
# sprites
# --------------------------------------------------------------------------

def _sprite_center(obj):
    """The sprite's ``center`` anchor as (x, y) floats (default 0.5, 0.5)."""
    center = _get(obj, "center")
    if center is None:
        return 0.5, 0.5
    cx = getattr(center, "x", None)
    if cx is not None:
        try:
            return float(cx), float(getattr(center, "y"))
        except Exception:
            pass
    try:
        arr = np.asarray(center, dtype=np.float64).ravel()
        if arr.size >= 2:
            return float(arr[0]), float(arr[1])
    except Exception:
        pass
    return 0.5, 0.5


def _draw_sprites(sprites, img, zbuf, view, proj, width, height, near, far,
                  fog_ctx=None):
    """Draw Sprite objects as camera-facing billboards (three.js Sprite
    semantics): each sprite is a quad parallel to the view plane through
    its world position, anchored so that ``center`` sits on the
    position, scaled by the world scale (x/y basis lengths), rotated in
    the view plane by ``material.rotation`` and textured by
    ``material.map`` tinted by ``material.color`` when a map is present
    (flat colour otherwise).  With ``sizeAttenuation`` False the scale
    is multiplied by the view depth so the on-screen size is constant.
    Shares the mesh pass's z-buffer (tests and writes depth)."""
    for obj in sprites:
        material = _get(obj, "material")
        if material is None:
            continue
        if _get(material, "visible", True) is False:
            continue
        world = _world_matrix(obj)
        anchor_v = view[:3, :3] @ world[:3, 3] + view[:3, 3]
        vz = float(anchor_v[2])
        if vz > -near or vz < -far:
            continue
        sx = float(np.linalg.norm(world[:3, 0]))
        sy = float(np.linalg.norm(world[:3, 1]))
        if sx <= 0.0 or sy <= 0.0:
            continue
        if not _get(material, "sizeAttenuation", True):
            sx *= -vz
            sy *= -vz
        cx, cy = _sprite_center(obj)
        rot = _float_or(_get(material, "rotation", 0.0), 0.0)
        cos_r = float(np.cos(rot))
        sin_r = float(np.sin(rot))
        # unit-quad corners (x right, y up); uv equals the corner, so
        # uv (0, 0) is the texture's bottom-left, as in three.js.
        quad = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        offs = quad - np.array([cx, cy], dtype=np.float64)
        offs[:, 0] *= sx
        offs[:, 1] *= sy
        rot_offs = np.stack(
            [
                offs[:, 0] * cos_r - offs[:, 1] * sin_r,
                offs[:, 0] * sin_r + offs[:, 1] * cos_r,
            ],
            axis=1,
        )
        view_pts = np.concatenate(
            [anchor_v[None, :2] + rot_offs, np.full((4, 1), vz)], axis=1
        )
        ones = np.ones((4, 1))
        clip = np.concatenate([view_pts, ones], axis=1) @ proj.T
        w = clip[:, 3]
        if np.any(np.abs(w) < 1e-9):
            continue
        ndc = clip[:, :3] / w[:, None]
        scr = np.stack(
            [
                (ndc[:, 0] * 0.5 + 0.5) * width,
                (-ndc[:, 1] * 0.5 + 0.5) * height,
            ],
            axis=1,
        )
        dep = ndc[:, 2]
        iw = 1.0 / w
        fd = np.full(4, -vz)
        tex_rgb = _texture_rgb(material)
        base_rgb = _color_rgb(_get(material, "color"))
        opacity = _get(material, "opacity")
        if opacity is not None:
            try:
                base_rgb = base_rgb * float(opacity)
            except Exception:
                pass
        if tex_rgb is not None:
            flat_index, dither_second = None, None
            uv = quad.copy()
        else:
            flat_index, dither_second = _palette_index(base_rgb), None
            uv = None
        for tri in ((0, 1, 2), (0, 2, 3)):
            sel = list(tri)
            _raster_fill(
                img, zbuf,
                scr[sel], dep[sel], iw[sel], fd[sel],
                uv[sel] if uv is not None else None,
                tex_rgb, base_rgb, flat_index, dither_second,
                fog_ctx, width, height,
            )


# --------------------------------------------------------------------------
# wireframe edges
# --------------------------------------------------------------------------

def _wire_edge(img, zbuf, p, q, dp, dq, index, width, height):
    """Draw one wireframe edge between screen-space points ``p`` and
    ``q``, testing and writing the shared z-buffer.  ``dp`` / ``dq``
    are the endpoints' NDC depths, interpolated plainly in the screen
    parameter (NDC depth is affine in screen space, see _raster_fill)."""
    dx = float(q[0] - p[0])
    dy = float(q[1] - p[1])
    steps = int(max(abs(dx), abs(dy)) * 2.0) + 1
    ts = np.linspace(0.0, 1.0, steps + 1)
    xs = p[0] + dx * ts
    ys = p[1] + dy * ts
    z = dp + (dq - dp) * ts
    xi = np.floor(xs).astype(np.int32)
    yi = np.floor(ys).astype(np.int32)
    ok = (
        (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        & np.isfinite(z)
    )
    if not np.any(ok):
        return
    xi = xi[ok]
    yi = yi[ok]
    # Sit a hair in front so an edge survives a coplanar fill.
    z = z[ok] - 1e-4
    upd = z < zbuf[yi, xi]
    if np.any(upd):
        zbuf[yi[upd], xi[upd]] = z[upd]
        img[yi[upd], xi[upd]] = index


def _facing_sign(nrm, centres, cam_pos, view):
    """(M,) front-facing measure for the (unnormalised) triangle
    normals ``nrm``: dot(normal, cam_pos - centre), > 0 when the
    front face looks toward the camera.

    Degenerate case: when the camera sits (nearly) at a triangle's
    centre -- the classic fullscreen-quad setup, an orthographic
    camera at the origin rendering a plane centred on the origin
    (the manual's shadertoy lesson) -- the camera-to-centre vector
    vanishes, the plain dot is 0, and the triangle would be culled
    even though it faces the camera head-on.  Real three.js culls
    by clip-space winding, which stays well-defined there; for
    (near-)degenerate triangles this falls back to the normal
    against the camera's backward axis (the view matrix's third
    row, pointing out of the screen toward the viewer)."""
    to_cam = cam_pos[None, :] - centres
    facing = np.sum(nrm * to_cam, axis=1)
    scale = np.sqrt(np.sum(nrm * nrm, axis=1)) * np.maximum(
        np.sqrt(np.sum(to_cam * to_cam, axis=1)), 1.0)
    deg = np.abs(facing) <= 1e-9 * np.maximum(scale, 1e-30)
    if np.any(deg):
        back = view[2, :3]
        facing = np.where(
            deg, np.sum(nrm * back[None, :], axis=1), facing)
    return facing


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def render_scene(scene, camera, width, height, shadow_enabled=False,
                 legacy_lights=False):
    # legacy_lights: punctual (point/spot) distance attenuation uses
    # the legacy-lights branch of getDistanceAttenuation instead of
    # the physical one.  Direct Python callers keep the physical
    # default; three.py's WebGLRenderer passes its own lights mode
    # (legacy unless the page opted into physically correct lights).
    width = int(width)
    height = int(height)
    img = np.zeros((height, width), dtype=np.uint8)

    if scene is None or camera is None or width <= 0 or height <= 0:
        return img

    # Let a real three.py scene refresh its world matrices if it can.
    update = _get(scene, "updateMatrixWorld")
    if callable(update):
        try:
            update()
        except Exception:
            pass
    cam_update = _get(camera, "updateMatrixWorld")
    if callable(cam_update):
        try:
            cam_update()
        except Exception:
            pass

    # background: a Color fills the frame; a Texture is drawn as a
    # full-frame backdrop (three.js stretches a plain background
    # texture over the viewport).  A CubeTexture backdrop is not
    # modelled (the frame stays black).
    bg = _get(scene, "background")
    if bg is not None:
        bg_rgb = None
        if _get(bg, "isTexture", False):
            bg_rgb = _texture_rgb({"map": bg})
        if bg_rgb is not None:
            th, tw = bg_rgb.shape[:2]
            rows = np.arange(height, dtype=np.int32) * th // height
            cols = np.arange(width, dtype=np.int32) * tw // width
            sub_bg = bg_rgb[rows][:, cols]
            _quantize_into(img, sub_bg, np.ones((height, width), dtype=bool), 0, 0)
        elif not _get(bg, "isCubeTexture", False):
            img[:, :] = _palette_index(_color_rgb(bg))

    # ---- collect meshes, points, lines and lights -----------------------
    meshes = []
    primitives = []   # ("points" | "line" | "line_segments", obj), scene order
    sprites = []      # Sprite objects, drawn last
    ambient = []      # (rgb, intensity)
    directional = []  # (rgb, intensity, direction-toward-light)
    hemisphere = []   # (sky rgb, ground rgb, intensity, axis)
    points = []       # (rgb, intensity, position, distance, decay)
    spots = []        # (rgb, intensity, position, axis, cos_outer,
    #                    cos_inner, distance, decay)
    # The first castShadow directional (when the renderer has
    # shadows enabled) -- (index into directional, light pos, target pos)
    shadow_light = None
    for obj in _walk(scene):
        kind = _light_kind(obj)
        if kind == "ambient":
            ambient.append(
                (_color_rgb(_get(obj, "color")), float(_get(obj, "intensity", 1.0) or 0.0))
            )
            continue
        if kind == "directional":
            pos = _vec3(_get(obj, "position"), (1.0, 1.0, 1.0))
            target_obj = _get(obj, "target")
            tgt = (
                _vec3(_get(target_obj, "position"), (0.0, 0.0, 0.0))
                if target_obj is not None
                else np.zeros(3)
            )
            direction = pos - tgt
            n = np.linalg.norm(direction)
            if n < 1e-12:
                continue
            if (
                shadow_enabled
                and shadow_light is None
                and _get(obj, "castShadow", False)
            ):
                shadow_light = (len(directional), pos.copy(), tgt.copy())
            directional.append(
                (
                    _color_rgb(_get(obj, "color")),
                    float(_get(obj, "intensity", 1.0) or 0.0),
                    direction / n,
                )
            )
            continue
        if kind == "hemisphere":
            # three.js derives the hemisphere axis from the light's
            # position (default (0, 1, 0)); a zero position falls back
            # to straight up.
            axis = _vec3(_get(obj, "position"), (0.0, 1.0, 0.0))
            an = np.linalg.norm(axis)
            axis = axis / an if an >= 1e-12 else np.array([0.0, 1.0, 0.0])
            hemisphere.append(
                (
                    _color_rgb(_get(obj, "color")),
                    _color_rgb(_get(obj, "groundColor")),
                    float(_get(obj, "intensity", 1.0) or 0.0),
                    axis,
                )
            )
            continue
        if kind == "point":
            points.append(
                (
                    _color_rgb(_get(obj, "color")),
                    float(_get(obj, "intensity", 1.0) or 0.0),
                    _vec3(_get(obj, "position"), (0.0, 0.0, 0.0)),
                    float(_get(obj, "distance", 0.0) or 0.0),
                    float(_get(obj, "decay", 2.0) or 0.0),
                )
            )
            continue
        if kind == "spot":
            pos = _vec3(_get(obj, "position"), (0.0, 0.0, 0.0))
            target_obj = _get(obj, "target")
            tgt = (
                _vec3(_get(target_obj, "position"), (0.0, 0.0, 0.0))
                if target_obj is not None
                else np.zeros(3)
            )
            axis = tgt - pos
            an = np.linalg.norm(axis)
            if an < 1e-12:
                continue
            try:
                angle = float(_get(obj, "angle", math.pi / 3.0))
            except Exception:
                angle = math.pi / 3.0
            angle = min(max(angle, 1e-6), math.pi / 2.0)
            try:
                penumbra = float(_get(obj, "penumbra", 0.0))
            except Exception:
                penumbra = 0.0
            penumbra = min(max(penumbra, 0.0), 1.0)
            spots.append(
                (
                    _color_rgb(_get(obj, "color")),
                    float(_get(obj, "intensity", 1.0) or 0.0),
                    pos,
                    axis / an,
                    math.cos(angle),
                    math.cos(angle * (1.0 - penumbra)),
                    float(_get(obj, "distance", 0.0) or 0.0),
                    float(_get(obj, "decay", 2.0) or 0.0),
                )
            )
            continue
        if _is_points(obj):
            if _get(obj, "visible", True) is False:
                continue
            primitives.append(("points", obj))
            continue
        if _is_line_segments(obj):
            if _get(obj, "visible", True) is False:
                continue
            primitives.append(("line_segments", obj))
            continue
        if _is_line(obj):
            if _get(obj, "visible", True) is False:
                continue
            primitives.append(("line", obj))
            continue
        if _is_sprite(obj):
            if _get(obj, "visible", True) is False:
                continue
            sprites.append(obj)
            continue
        if _is_mesh(obj):
            if _get(obj, "visible", True) is False:
                continue
            if _get(obj, "isInstancedMesh", False):
                inst_count = int(_get(obj, "count", 0) or 0)
                inst_mat_attr = _get(obj, "instanceMatrix")
                if inst_count > 0 and inst_mat_attr is not None:
                    mat_arr = np.asarray(
                        _get(inst_mat_attr, "array", inst_mat_attr),
                        dtype=np.float64,
                    ).reshape(-1)
                    max_i = min(inst_count, int(mat_arr.size // 16))
                    inst_col_attr = _get(obj, "instanceColor")
                    col_arr = None
                    if inst_col_attr is not None:
                        ca = np.asarray(
                            _get(inst_col_attr, "array", inst_col_attr),
                            dtype=np.float64,
                        ).reshape(-1)
                        if ca.size >= max_i * 3:
                            col_arr = ca
                    parent_world = _world_matrix(obj)
                    geom = _get(obj, "geometry")
                    mat = _get(obj, "material")
                    cast_s = bool(_get(obj, "castShadow", False))
                    recv_s = bool(_get(obj, "receiveShadow", False))
                    for ii in range(max_i):
                        m_inst = mat_arr[ii * 16 : (ii + 1) * 16].reshape(4, 4).T
                        ic = col_arr[ii * 3 : (ii + 1) * 3] if col_arr is not None else None
                        meshes.append({
                            "isMesh": True,
                            "geometry": geom,
                            "material": mat,
                            "matrixWorld": parent_world @ m_inst,
                            "parent": None,
                            "castShadow": cast_s,
                            "receiveShadow": recv_s,
                            "_instance_color": ic,
                        })
                continue
            meshes.append(obj)

    if not meshes and not primitives and not sprites:
        return img

    view, proj, cam_pos = _camera_matrices(camera, width, height)
    near = float(_get(camera, "near", 0.1) or 0.1)
    far = float(_get(camera, "far", 2000.0) or 2000.0)

    zbuf = np.full((height, width), np.inf, dtype=np.float64)

    fog_ctx = _fog_context(scene)

    # Environment map: scene.environment, sampled per face by
    # the Phong/Standard shading path (a material envMap overrides it).
    env_ctx = _env_sampler(_get(scene, "environment"))

    # ---- shadow map -------------------------------------------
    shadow_ctx = None  # (depth map, light view-projection matrix)
    view_inv = None
    if shadow_light is not None:
        caster_data = []
        receiver_data = []
        for mesh in meshes:
            cast = bool(_get(mesh, "castShadow", False))
            recv = bool(_get(mesh, "receiveShadow", False))
            if not (cast or recv):
                continue
            wp, tr = _mesh_world_tris(mesh)
            if wp is None:
                continue
            if cast:
                caster_data.append(
                    (wp, tr, _material_side(_get(mesh, "material")))
                )
            if recv:
                receiver_data.append((wp, tr))
        built = _build_shadow_map(
            caster_data, receiver_data, shadow_light[1], shadow_light[2]
        )
        if built is not None:
            shadow_ctx = built
            view_inv = np.linalg.inv(view)

    # ---- per mesh --------------------------------------------------------
    for mesh in meshes:
        geometry = _get(mesh, "geometry")
        material = _get(mesh, "material")
        if geometry is None or material is None:
            continue
        if _get(geometry, "isGeometry", False):
            sync_fn = _get(geometry, "_sync_buffers")
            if callable(sync_fn):
                sync_fn()
        if _get(material, "visible", True) is False:
            continue
        positions = _attr_array(geometry, "position")
        if positions is None or len(positions) == 0:
            continue
        # ShaderMaterial: both GLSL stages run on the
        # vector evaluator; the fixed shading paths below do not
        # apply.  (Points/Line/Sprite objects keep the fixed paths --
        # shader materials are only wired for meshes.)
        if _get(material, "isShaderMaterial", False):
            _draw_shader_mesh(
                img, zbuf, mesh, view, proj, cam_pos,
                width, height, near, far)
            continue
        # Texture support: both a uv attribute and a material
        # map are required; otherwise the flat-colour path below is used.
        uv_attr = _uv_array(geometry)
        tex_rgb = _texture_rgb(material)
        if uv_attr is None or len(uv_attr) != len(positions):
            tex_rgb = None
        if tex_rgb is not None:
            # three.js uvTransform for the map (Matrix3.setUvTransform
            # composed from offset / repeat / rotation / center).
            # Applied via a fresh array -- uv_attr may alias the
            # geometry's own attribute storage.
            _map = _get(material, "map")
            _rep = _get(_map, "repeat")
            _off = _get(_map, "offset")
            _cen = _get(_map, "center")
            _rx = float(_get(_rep, "x", 1.0)) if _rep is not None else 1.0
            _ry = float(_get(_rep, "y", 1.0)) if _rep is not None else 1.0
            _ox = float(_get(_off, "x", 0.0)) if _off is not None else 0.0
            _oy = float(_get(_off, "y", 0.0)) if _off is not None else 0.0
            _rot = _get(_map, "rotation")
            _rot = float(_rot) if _rot is not None else 0.0
            _cx = float(_get(_cen, "x", 0.0)) if _cen is not None else 0.0
            _cy = float(_get(_cen, "y", 0.0)) if _cen is not None else 0.0
            if _rot == 0.0:
                if (_rx, _ry) != (1.0, 1.0) or (_ox, _oy) != (0.0, 0.0):
                    uv_attr = uv_attr * np.array([_rx, _ry]) \
                        + np.array([_ox, _oy])
            else:
                _c = math.cos(_rot)
                _s = math.sin(_rot)
                _u = uv_attr[:, 0]
                _v = uv_attr[:, 1]
                uv_attr = np.stack([
                    _rx * _c * _u + _rx * _s * _v
                    - _rx * (_c * _cx + _s * _cy) + _cx + _ox,
                    -_ry * _s * _u + _ry * _c * _v
                    - _ry * (-_s * _cx + _c * _cy) + _cy + _oy,
                ], axis=1)
        indices = _index_array(geometry, len(positions))
        if indices.size < 3:
            continue
        tris = indices.reshape(-1, 3)
        # keep only in-range indices
        valid = np.all((tris >= 0) & (tris < len(positions)), axis=1)
        tris = tris[valid]
        if len(tris) == 0:
            continue

        world = _world_matrix(mesh)
        world_pos = positions @ world[:3, :3].T + world[:3, 3]
        view_pos = world_pos @ view[:3, :3].T + view[:3, 3]

        mat_kind = _material_kind(material)
        base_rgb = _color_rgb(_get(material, "color"))
        inst_rgb = _get(mesh, "_instance_color")
        if inst_rgb is not None:
            base_rgb = base_rgb * inst_rgb
        opacity = _get(material, "opacity")
        if opacity is not None:
            try:
                base_rgb = base_rgb * float(opacity)
            except Exception:
                pass

        tri_vcol = None
        if _get(material, "vertexColors", False):
            _vcol = _attr_array(geometry, "color")
            if _vcol is not None and len(_vcol) == len(positions):
                _vc_rgb = np.asarray(_vcol, dtype=np.float64)[:, :3]
                if _vc_rgb.size and float(np.max(_vc_rgb)) > 1.5:
                    _vc_rgb = _vc_rgb / 255.0
                tri_vcol = np.clip(_vc_rgb[tris].mean(axis=1), 0.0, 1.0)

        # flat colour for Basic materials is loop-invariant
        basic_index = (
            _palette_index(base_rgb)
            if (mat_kind == "basic" and tri_vcol is None)
            else None
        )

        wireframe = bool(_get(material, "wireframe", False))
        side = _material_side(material)
        # Specular constants are loop-invariant too.  ``spec_rgb`` /
        # ``spec_exp`` / ``spec_scale`` drive the Blinn-Phong term in
        # _shade for both Phong and Standard.
        spec_rgb = None
        spec_exp = 30.0
        spec_scale = 1.0
        std_roughness = 1.0
        std_metalness = 0.0
        if mat_kind == "phong":
            spec_rgb = _color_rgb(
                _get(material, "specular"), (17.0 / 255.0,) * 3
            )
            try:
                spec_exp = float(_get(material, "shininess", 30.0))
            except Exception:
                spec_exp = 30.0
        elif mat_kind == "standard":
            try:
                std_roughness = min(
                    max(float(_get(material, "roughness", 1.0)), 0.0), 1.0
                )
            except Exception:
                std_roughness = 1.0
            try:
                std_metalness = min(
                    max(float(_get(material, "metalness", 0.0)), 0.0), 1.0
                )
            except Exception:
                std_metalness = 0.0
            # F0: dielectric 0.04 grey -> conductor base colour.
            spec_rgb = 0.04 + (base_rgb - 0.04) * std_metalness
            # roughness -> Blinn-Phong exponent (0 -> 256, 1 -> 1),
            # plus an overall damping so a rough surface loses its
            # highlight instead of only smearing it.
            spec_exp = 2.0 ** ((1.0 - std_roughness) * 8.0)
            spec_scale = 1.0 - 0.75 * std_roughness
        # The diffuse colour actually multiplied by the irradiance:
        # for Standard, metalness eats the diffuse.
        diff_rgb = (
            base_rgb * (1.0 - std_metalness)
            if mat_kind == "standard"
            else base_rgb
        )
        # Environment reflection: a per-material envMap wins
        # over scene.environment.
        env_sample = None
        env_intensity = 1.0
        if mat_kind in ("phong", "standard"):
            menv = _get(material, "envMap")
            env_sample = (
                _env_sampler(menv) if menv is not None else env_ctx
            )
            try:
                env_intensity = float(
                    _get(material, "envMapIntensity", 1.0)
                )
            except Exception:
                env_intensity = 1.0

        # ---- vectorised per-triangle preparation --------------
        wp_tris = world_pos[tris]             # (M,3,3)
        vw_tris = view_pos[tris]              # (M,3,3)
        e1 = wp_tris[:, 1] - wp_tris[:, 0]
        e2 = wp_tris[:, 2] - wp_tris[:, 0]
        nrm = np.stack(
            [
                e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1],
                e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2],
                e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0],
            ],
            axis=1,
        )
        nlen = np.sqrt(np.sum(nrm * nrm, axis=1))
        centres = wp_tris.mean(axis=1)
        to_cam = cam_pos[None, :] - centres
        # sign of the (unnormalised) geometric normal against the view
        # direction: > 0 means the front face looks toward the camera
        facing = _facing_sign(nrm, centres, cam_pos, view)
        valid_tri = nlen > 1e-12
        if side == 0:      # FrontSide: cull back faces
            keep = valid_tri & (facing > 0.0)
        elif side == 1:    # BackSide: cull front faces
            keep = valid_tri & (facing < 0.0)
        else:              # DoubleSide: draw both
            keep = valid_tri
        if not np.any(keep):
            continue
        # Shading normal: unit length, oriented toward the viewer (the
        # visible side is what gets shaded).
        nrm_u = nrm / np.where(valid_tri, nlen, 1.0)[:, None]
        nrm_s = np.where((facing < 0.0)[:, None], -nrm_u, nrm_u)

        # Point/spot per-face geometry: for each punctual
        # light, the unit direction from the face centre toward the
        # light and the distance attenuation (spots pre-multiply their
        # cone factor).  Diffuse and specular share these; the shadow
        # variant reuses them unchanged (only a directional light can
        # ever be excluded).
        punctual = []  # (rgb, intensity, lhat (M, 3), atten (M,))
        for lrgb, inten, lpos, ldist, ldecay in points:
            to_l = lpos[None, :] - centres
            dist = np.sqrt(np.sum(to_l * to_l, axis=1))
            atten = _distance_attenuation(
                dist, ldist, ldecay, legacy=legacy_lights)
            lhat = to_l / np.where(dist > 1e-12, dist, 1.0)[:, None]
            punctual.append((lrgb, inten, lhat, atten))
        for (lrgb, inten, lpos, axis, cos_out, cos_in,
             ldist, ldecay) in spots:
            to_l = lpos[None, :] - centres
            dist = np.sqrt(np.sum(to_l * to_l, axis=1))
            atten = _distance_attenuation(
                dist, ldist, ldecay, legacy=legacy_lights)
            lhat = to_l / np.where(dist > 1e-12, dist, 1.0)[:, None]
            cone = _spot_factor(
                -np.sum(lhat * axis[None, :], axis=1), cos_out, cos_in
            )
            punctual.append((lrgb, inten, lhat, atten * cone))

        # Flat face colours for every triangle, vectorised (same maths
        # as the per-triangle loop).  ``_shade`` optionally drops
        # one directional light: the shadow variant is the
        # same face with the shadow-casting light's contribution
        # removed, selected per fragment by the shadow test.
        def _shade(exclude_dir=-1):
            if ambient or directional or hemisphere or punctual:
                amb = ambient
                dirs = directional
            else:
                amb = [_FALLBACK_AMBIENT]
                dirs = [_FALLBACK_DIRECTIONAL]
            irr = np.zeros((len(tris), 3))
            for lrgb, inten in amb:
                irr += lrgb * inten
            for srgb, grgb, inten, axis in hemisphere:
                # three.js hemisphere irradiance: ground colour at
                # down-facing normals, sky colour at up-facing ones.
                w = (0.5 + 0.5 * (nrm_s @ axis))[:, None]
                irr += inten * (
                    grgb[None, :] * (1.0 - w) + srgb[None, :] * w
                )
            for di, (lrgb, inten, ldir) in enumerate(dirs):
                if di == exclude_dir:
                    continue
                d = np.maximum(nrm_s @ ldir, 0.0)
                irr += (lrgb * inten)[None, :] * d[:, None]
            for lrgb, inten, lhat, atten in punctual:
                d = np.maximum(
                    np.sum(nrm_s * lhat, axis=1), 0.0
                ) * atten
                irr += (lrgb * inten)[None, :] * d[:, None]
            rgb = diff_rgb[None, :] * irr
            if tri_vcol is not None:
                rgb = rgb * tri_vcol
            if spec_rgb is not None:
                # Blinn-Phong specular, flat per face: the view
                # direction is taken from the face centre.  Phong uses
                # its specular colour / shininess verbatim; Standard
                # feeds the same term with the F0 mix as the
                # colour, the roughness-derived exponent, and the
                # roughness damping as spec_scale.
                vn = np.sqrt(np.sum(to_cam * to_cam, axis=1))
                vok = vn > 1e-12
                vdir = to_cam / np.where(vok, vn, 1.0)[:, None]
                spec = np.zeros((len(tris), 3))
                for di, (lrgb, inten, ldir) in enumerate(dirs):
                    if di == exclude_dir:
                        continue
                    hv = ldir[None, :] + vdir
                    hn = np.sqrt(np.sum(hv * hv, axis=1))
                    hok = hn > 1e-12
                    hv = hv / np.where(hok, hn, 1.0)[:, None]
                    s = np.sum(nrm_s * hv, axis=1)
                    contrib = np.where(
                        hok & vok & (s > 0.0),
                        np.maximum(s, 0.0) ** spec_exp,
                        0.0,
                    )
                    spec += (spec_rgb * lrgb * inten)[None, :] \
                        * contrib[:, None]
                # Punctual lights: same Blinn-Phong term,
                # with the per-face light direction and the distance /
                # cone attenuation folded in.
                for lrgb, inten, lhat, atten in punctual:
                    hv = lhat + vdir
                    hn = np.sqrt(np.sum(hv * hv, axis=1))
                    hok = hn > 1e-12
                    hv = hv / np.where(hok, hn, 1.0)[:, None]
                    s = np.sum(nrm_s * hv, axis=1)
                    contrib = np.where(
                        hok & vok & (s > 0.0),
                        np.maximum(s, 0.0) ** spec_exp,
                        0.0,
                    ) * atten
                    spec += (spec_rgb * lrgb * inten)[None, :] \
                        * contrib[:, None]
                rgb = rgb + spec * spec_scale
                if env_sample is not None:
                    # Environment reflection: sample the
                    # cube map along the reflected view direction and
                    # mix it into the specular.  Deliberately not
                    # shadow-tested: the directional shadow map does
                    # not occlude the environment.
                    ndv = np.sum(nrm_s * vdir, axis=1)
                    refl = 2.0 * ndv[:, None] * nrm_s - vdir
                    env_rgb = env_sample(refl)
                    if env_rgb is not None:
                        if mat_kind == "standard":
                            ndv_c = np.clip(ndv, 0.0, 1.0)
                            fres = spec_rgb[None, :] + (
                                1.0 - spec_rgb[None, :]
                            ) * ((1.0 - ndv_c) ** 5)[:, None]
                            env_term = env_rgb * fres * (
                                env_intensity
                                * (1.0 - 0.5 * std_roughness)
                            )
                        else:  # phong: env x specular colour
                            env_term = (
                                env_rgb * spec_rgb[None, :]
                                * env_intensity
                            )
                        rgb = rgb + np.where(
                            vok[:, None], env_term, 0.0
                        )
            return rgb

        # Emissive (Lambert / Phong / Standard): three.js adds
        # emissive x emissiveIntensity to the outgoing colour,
        # independent of lights, shadows and textures.  It is kept
        # separate from face_rgb_all because the textured path uses
        # that array as a multiplier; the additive term is applied
        # where flat colours are formed (per-triangle loop) and
        # inside _raster_fill for the textured paths.
        emis_rgb = None
        if mat_kind in ("lambert", "phong", "standard"):
            _emis = _get(material, "emissive")
            if _emis is not None:
                _ec = _color_rgb(_emis, (0.0, 0.0, 0.0))
                if np.any(_ec):
                    try:
                        _ei = float(
                            _get(material, "emissiveIntensity", 1.0))
                    except Exception:
                        _ei = 1.0
                    emis_rgb = _ec * _ei

        recv_shadow = (
            shadow_ctx is not None
            and bool(_get(mesh, "receiveShadow", False))
            and mat_kind in ("lambert", "phong", "standard")
        )
        if mat_kind == "normal":
            # three.js colours fragments by the view-space normal; under
            # flat shading the (viewer-oriented) face normal is the
            # fragment normal, mapped from [-1, 1] to [0, 1].
            face_rgb_all = (nrm_s @ view[:3, :3].T) * 0.5 + 0.5
        elif mat_kind in ("lambert", "phong", "standard"):
            face_rgb_all = _shade()
        else:
            face_rgb_all = np.broadcast_to(base_rgb, (len(tris), 3))
            if tri_vcol is not None:
                face_rgb_all = face_rgb_all * tri_vcol
        face_rgb_all_sh = (
            _shade(shadow_light[0]) if recv_shadow else None
        )

        # Batched projection of every vertex (the per-triangle loop only
        # indexes into these arrays) + per-triangle clip classification.
        ones = np.ones((len(view_pos), 1))
        clip_all = np.concatenate([view_pos, ones], axis=1) @ proj.T
        w_all = clip_all[:, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndc_all = clip_all[:, :3] / w_all[:, None]
            invw_all = 1.0 / w_all
        scr_all = np.stack(
            [
                (ndc_all[:, 0] * 0.5 + 0.5) * width,
                (-ndc_all[:, 1] * 0.5 + 0.5) * height,
            ],
            axis=1,
        )
        depth_all = ndc_all[:, 2]
        fogd_all = -view_pos[:, 2]
        vz = vw_tris[:, :, 2]
        inside = vz <= -near
        if far > near:
            inside = inside & (vz >= -far)
        tri_inside = inside.all(axis=1)
        w_ok = (np.abs(w_all[tris]) >= 1e-9).all(axis=1)

        # Per-vertex light-space NDC for the shadow test;
        # the orthographic light VP keeps w == 1.
        ls_all = None
        if recv_shadow:
            ls_all = (
                np.concatenate([world_pos, ones], axis=1) @ shadow_ctx[1].T
            )[:, :3]

        # ---- per surviving triangle --------------------------------------
        for ti in np.nonzero(keep)[0]:
            tri = tris[ti]
            face_rgb = face_rgb_all[ti]
            if emis_rgb is not None and tex_rgb is None:
                # Flat path: face_rgb IS the outgoing colour, so the
                # emissive term is added here.  (Textured triangles
                # keep face_rgb as the multiplier and _raster_fill
                # adds emis_rgb after the texel multiply.)
                face_rgb = face_rgb + emis_rgb
            if tex_rgb is not None:
                flat_index, dither_second = None, None
            elif basic_index is not None:
                flat_index, dither_second = basic_index, None
            else:
                flat_index, dither_second = _flat_fill_info(face_rgb)

            face_rgb_sh = (
                face_rgb_all_sh[ti] if recv_shadow else None
            )

            windex = None
            if wireframe:
                if fog_ctx is not None:
                    # flat shading: fog the edge colour by the face
                    # centre's view depth
                    fog_rgb, fog_near, fog_far = fog_ctx
                    cf = float(np.clip(
                        (fogd_all[tri].mean() - fog_near)
                        / (fog_far - fog_near),
                        0.0, 1.0,
                    ))
                    windex = _palette_index(
                        face_rgb * (1.0 - cf) + fog_rgb * cf
                    )
                elif basic_index is not None:
                    windex = basic_index
                else:
                    windex = _palette_index(face_rgb)

            if tri_inside[ti] and w_ok[ti]:
                # fast path: the triangle misses both clip planes
                scr3 = scr_all[tri]
                dep3 = depth_all[tri]
                iw3 = invw_all[tri]
                fd3 = fogd_all[tri]
                uv3 = uv_attr[tri] if tex_rgb is not None else None
                if wireframe:
                    for e0, e1 in ((0, 1), (1, 2), (2, 0)):
                        _wire_edge(
                            img, zbuf, scr3[e0], scr3[e1],
                            dep3[e0], dep3[e1],
                            windex, width, height,
                        )
                else:
                    _raster_fill(
                        img, zbuf, scr3, dep3, iw3, fd3, uv3,
                        tex_rgb, face_rgb, flat_index, dither_second,
                        fog_ctx, width, height,
                        sh3=ls_all[tri] if recv_shadow else None,
                        smap=shadow_ctx[0] if recv_shadow else None,
                        face_rgb_sh=face_rgb_sh,
                        emis_rgb=emis_rgb,
                    )
                continue

            # clipped path: Sutherland-Hodgman in view space
            vw = vw_tris[ti]
            if tex_rgb is not None:
                poly, poly_uv = _clip_with_uv(vw, uv_attr[tri], near, far)
            else:
                poly = _clip_triangle(vw, near, far)
                poly_uv = None
            if len(poly) < 3:
                continue
            poly_np = np.array(poly, dtype=np.float64)
            ones_p = np.ones((len(poly_np), 1))
            clip_p = np.concatenate([poly_np, ones_p], axis=1) @ proj.T
            w_p = clip_p[:, 3]
            if np.any(np.abs(w_p) < 1e-9):
                continue
            ndc_p = clip_p[:, :3] / w_p[:, None]
            scr_p = np.stack(
                [
                    (ndc_p[:, 0] * 0.5 + 0.5) * width,
                    (-ndc_p[:, 1] * 0.5 + 0.5) * height,
                ],
                axis=1,
            )
            depth_p = ndc_p[:, 2]
            invw_p = 1.0 / w_p
            fogd_p = -poly_np[:, 2]
            ls_p = None
            if recv_shadow:
                # poly is in camera view space; take it back to world
                # and into light space for the shadow test.
                wp_p = (
                    np.concatenate([poly_np, ones_p], axis=1) @ view_inv.T
                )[:, :3]
                ls_p = (
                    np.concatenate([wp_p, ones_p], axis=1)
                    @ shadow_ctx[1].T
                )[:, :3]
            poly_uv_np = (
                np.array(poly_uv, dtype=np.float64)
                if poly_uv is not None
                else None
            )
            for k in range(1, len(poly) - 1):
                tri_idx = (0, k, k + 1)
                if wireframe:
                    for e0, e1 in (
                        (tri_idx[0], tri_idx[1]),
                        (tri_idx[1], tri_idx[2]),
                        (tri_idx[2], tri_idx[0]),
                    ):
                        _wire_edge(
                            img, zbuf, scr_p[e0], scr_p[e1],
                            depth_p[e0], depth_p[e1],
                            windex, width, height,
                        )
                    continue
                sel = list(tri_idx)
                _raster_fill(
                    img, zbuf,
                    scr_p[sel], depth_p[sel], invw_p[sel], fogd_p[sel],
                    poly_uv_np[sel] if poly_uv_np is not None else None,
                    tex_rgb, face_rgb, flat_index, dither_second,
                    fog_ctx, width, height,
                    sh3=ls_p[sel] if ls_p is not None else None,
                    smap=shadow_ctx[0] if ls_p is not None else None,
                    face_rgb_sh=face_rgb_sh,
                    emis_rgb=emis_rgb,
                )

    # ---- points / lines (share the mesh z-buffer) ------------------------
    if primitives:
        _draw_primitives(
            primitives, img, zbuf, view, proj, width, height, near, far,
            fog_ctx,
        )

    # ---- sprites (billboards; share the mesh z-buffer) -------------------
    if sprites:
        _draw_sprites(
            sprites, img, zbuf, view, proj, width, height, near, far,
            fog_ctx,
        )
    return img


# --------------------------------------------------------------------------
# GLSL ShaderMaterial / post passes
#
# ShaderMaterial meshes bypass the fixed shading paths above: the
# vertex stage runs on glsl_eval's vector evaluator over all vertices,
# the rasterizer keeps the winning pixels' perspective-correct varying
# fields, and the fragment stage is evaluated over the whole screen
# grid (which is also what makes dFdx/dFdy meaningful).  Compilation
# (glsl.py: prefix + preprocess + parse) is cached per material.
#
# Documented approximations of this layer:
#   - near/far clip classification and back-face culling use the
#     host-side modelView transform; a vertex shader that displaces
#     vertices does not move its own clip test (the clipped path
#     re-projects with the standard transform, like the uv path).
#   - scene.fog and shadow receiving do not apply to shader
#     materials (three.js applies them only through shader chunks the
#     user includes); casting into other materials' shadow maps uses
#     the geometry, as for every material.
#   - fragment alpha is ignored (the palette frame has no alpha);
#     ``wireframe: true`` on a shader material is drawn filled.
# --------------------------------------------------------------------------

_GLSL_MODS = None
_GLSL_CHUNKS = None

# Last-resort fallback, used only if glsl.load_chunks() itself fails:
# glsl.py embeds the full r187 ShaderChunk set, so standalone
# deployments (e.g. Code & Magic) normally resolve every #include
# from the embedded texts.  The fragment prefix references the
# colorspace chunk, and this port does no colour-space conversion
# anywhere, so the identity form is the faithful local equivalent.
_GLSL_FALLBACK_CHUNKS = {
    "colorspace_pars_fragment": (
        "vec4 linearToOutputTexel( vec4 value ) { return value; }\n"),
    "tonemapping_pars_fragment": "",
}


def _load_glsl():
    global _GLSL_MODS
    if _GLSL_MODS is None:
        import glsl
        import glsl_eval

        _GLSL_MODS = (glsl, glsl_eval)
    return _GLSL_MODS


def _shader_chunks():
    global _GLSL_CHUNKS
    if _GLSL_CHUNKS is None:
        glsl, _ = _load_glsl()
        try:
            chunks = glsl.load_chunks()
        except Exception:
            chunks = {}
        _GLSL_CHUNKS = chunks if chunks else dict(_GLSL_FALLBACK_CHUNKS)
    return _GLSL_CHUNKS


def _compile_shader_material(material):
    """Parse (and cache) the vertex/fragment programs of a
    ShaderMaterial.  Raises glsl.GLSLError / glsl_eval.GLSLError on
    bad source, which the host surfaces like any other JS error."""
    glsl, _ = _load_glsl()
    vs = str(_get(material, "vertexShader", "") or "")
    fs = str(_get(material, "fragmentShader", "") or "")
    raw = bool(_get(material, "isRawShaderMaterial", False))
    defines = _get(material, "defines") or {}
    try:
        def_items = sorted(
            (str(k), repr(v)) for k, v in defines.items())
        defs = {str(k): v for k, v in defines.items()}
    except Exception:
        def_items = []
        defs = {}
    key = (vs, fs, raw, tuple(def_items))
    cached = _get(material, "_glsl_compiled")
    if cached is not None and cached[0] == key:
        return cached[1]
    chunks = _shader_chunks()
    compiled = {
        "vertex": glsl.parse_shader(
            vs, stage="vertex", defines=defs, chunks=chunks, raw=raw),
        "fragment": glsl.parse_shader(
            fs, stage="fragment", defines=defs, chunks=chunks, raw=raw),
    }
    try:
        material._glsl_compiled = (key, compiled)
    except Exception:
        pass
    return compiled


def _prog_structs(prog):
    """{struct name: [(field_type, field_name), ...]} for the
    evaluator's type mapping (the parser stores triples)."""
    out = {}
    structs = getattr(prog, "structs", None) or {}
    for name, fields in structs.items():
        out[name] = [(f[0], f[1]) for f in fields]
    return out


def _image_rgb01(img):
    """An image-like value as an (h, w, 3) float64 array in 0..1
    (the _texture_rgb conversion, without the material plumbing)."""
    if img is None:
        return None
    try:
        arr = np.asarray(img, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
        return None
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.ascontiguousarray(arr)


def _sampler_dict(value):
    """A texture-ish uniform value as the evaluator's sampler dict,
    or None when there is nothing to sample."""
    if value is None:
        return None
    frame_rgb = _get(value, "frame_rgb")
    if frame_rgb is not None:
        data = np.asarray(frame_rgb, dtype=np.float64)
    else:
        img = _get(value, "image")
        if img is None:
            img = value  # a bare array was assigned
        data = _image_rgb01(img)
    if data is None:
        return None
    wrap = "repeat"
    ws = _get(value, "wrapS")
    if ws in ("clamp", "ClampToEdgeWrapping", 1001):
        wrap = "clamp"
    return {"data": data, "wrap": wrap}


def _value_components(value, n):
    """Up to n float components of a uniform value: Color / Vector /
    sequence / scalar, zero-padded."""
    if value is None:
        return [0.0] * n
    if hasattr(value, "r") and hasattr(value, "g"):
        comps = [float(value.r), float(value.g), float(value.b)]
        if n >= 4:
            comps.append(float(getattr(value, "a", 1.0)))
        return (comps + [0.0] * n)[:n]
    if hasattr(value, "x") and hasattr(value, "y"):
        comps = [float(value.x), float(value.y)]
        if hasattr(value, "z"):
            comps.append(float(value.z))
        if hasattr(value, "w"):
            comps.append(float(value.w))
        return (comps + [0.0] * n)[:n]
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
        comps = [float(c) for c in arr]
        return (comps + [0.0] * n)[:n]
    except Exception:
        try:
            return ([float(value)] + [0.0] * (n - 1))[:n]
        except Exception:
            return [0.0] * n


def _glsl_default(t, structs):
    """The evaluator's zero value for a type descriptor, built
    locally (used to pad short uniform arrays)."""
    if t == "float":
        return (0.0, "float")
    if t == "int":
        return (0, "int")
    if t == "bool":
        return (False, "bool")
    if t in ("sampler", "samplerCube"):
        return (None, t)
    if isinstance(t, tuple):
        if t[0] in ("vec", "ivec", "bvec"):
            zero = {"vec": 0.0, "ivec": 0, "bvec": False}[t[0]]
            return (tuple(zero for _ in range(t[1])), t)
        if t[0] == "mat":
            return (tuple(tuple(0.0 for _ in range(t[1]))
                          for _ in range(t[1])), t)
        if t[0] == "array":
            return ([_glsl_default(t[1], structs)
                     for _ in range(t[2])], t)
        if t[0] == "struct":
            _, glsl_eval = _load_glsl()
            return ({fn: _glsl_default(
                glsl_eval.glsl_type(ft, structs), structs)
                for ft, fn in structs.get(t[1], [])}, t)
    return (None, t)


def _convert_uniform(value, tname, arr, structs):
    """One material.uniforms entry -> the evaluator's (data, type)
    input pair, or None to leave the evaluator's default in place."""
    _, glsl_eval = _load_glsl()
    if arr is not None:
        elem_t = glsl_eval.glsl_type(tname, structs)
        if arr == ("unsized",):
            n = len(value) if value is not None else 0
        elif isinstance(arr, tuple) and len(arr) == 3 and arr[0] == "num":
            n = int(arr[1])
        else:
            try:
                n = len(value)
            except Exception:
                n = 0
        seq = list(value) if value is not None else []
        vals = []
        for i in range(n):
            item = seq[i] if i < len(seq) else None
            conv = _convert_uniform(item, tname, None, structs)
            vals.append(conv if conv is not None
                        else _glsl_default(elem_t, structs))
        return (vals, ("array", elem_t, n))
    t = glsl_eval.glsl_type(tname, structs)
    if t == "float":
        try:
            return (float(value), "float")
        except Exception:
            return None
    if t == "int":
        try:
            return (int(value), "int")
        except Exception:
            return None
    if t == "bool":
        return (bool(value), "bool")
    if isinstance(t, tuple) and t[0] in ("vec", "ivec", "bvec"):
        kind = {"vec": float, "ivec": int, "bvec": bool}[t[0]]
        comps = _value_components(value, t[1])
        return (tuple(kind(c) for c in comps), t)
    if isinstance(t, tuple) and t[0] == "mat":
        m = _as_mat4(value)
        if m is None:
            try:
                m = np.asarray(value, dtype=np.float64).reshape(
                    t[1], t[1])
            except Exception:
                return None
        elif t[1] != 4:
            m = m[: t[1], : t[1]]
        return (np.asarray(m, dtype=np.float64), t)
    if t in ("sampler", "samplerCube"):
        return (_sampler_dict(value), t)
    return None  # structs and anything else: evaluator default


def _uniform_inputs(prog, material, builtins):
    """Evaluator inputs for every uniform ``prog`` declares: the
    host built-ins first, then the material's uniforms dict."""
    structs = _prog_structs(prog)
    uniforms = _get(material, "uniforms") or {}
    inputs = {}
    for tname, name, arr in prog.uniforms():
        if name in builtins:
            inputs[name] = builtins[name]
            continue
        if isinstance(uniforms, dict):
            entry = uniforms.get(name)
        else:
            entry = _get(uniforms, name)
        if entry is None:
            continue
        value = _get(entry, "value", entry)
        conv = _convert_uniform(value, tname, arr, structs)
        if conv is not None:
            inputs[name] = conv
    return inputs


def _shader_builtins(view, proj, world, cam_pos):
    mv = view @ world
    normal = np.eye(3)
    try:
        normal = np.linalg.inv(mv[:3, :3]).T
    except np.linalg.LinAlgError:
        pass
    return {
        "modelMatrix": (world, "mat4"),
        "modelViewMatrix": (mv, "mat4"),
        "projectionMatrix": (proj, "mat4"),
        "viewMatrix": (view, "mat4"),
        "normalMatrix": (np.asarray(normal, dtype=np.float64), "mat3"),
        "cameraPosition": (
            tuple(float(c) for c in cam_pos), "vec3"),
    }


def _attribute_data(geometry, name, tname):
    """A geometry attribute as an (N, k) float64 array (or (N,) for a
    scalar type), or None when the geometry does not provide it."""
    _, glsl_eval = _load_glsl()
    attrs = _get(geometry, "attributes")
    raw = None
    if isinstance(attrs, dict):
        raw = attrs.get(name)
    if raw is None:
        raw = _get(geometry, name)
    if raw is None:
        return None
    item_size = _get(raw, "itemSize")
    arr = getattr(raw, "array", None)
    if arr is None:
        arr = raw
    try:
        arr = np.asarray(arr, dtype=np.float64)
    except Exception:
        return None
    if item_size is None:
        t = glsl_eval.glsl_type(tname, {})
        item_size = t[1] if isinstance(t, tuple) else 1
    item_size = int(item_size)
    if arr.size == 0 or arr.size % item_size != 0:
        return None
    arr = arr.reshape(-1, item_size)
    if item_size == 1:
        return arr[:, 0]
    return arr


def _per_vertex_components(value, n_verts, ncomp):
    """A vertex-stage output value -> ``ncomp`` (N,) float64 arrays."""
    if value is None:
        return [np.zeros(n_verts) for _ in range(ncomp)]
    data = value[0]
    if isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[0] == n_verts:
            return [np.asarray(data[:, k], dtype=np.float64)
                    for k in range(min(ncomp, data.shape[1]))] + \
                   [np.zeros(n_verts)] * max(
                       0, ncomp - data.shape[1])
        return [np.full(n_verts, float(data))] * ncomp
    if isinstance(data, tuple):
        out = []
        for c in data[:ncomp]:
            out.append(np.broadcast_to(
                np.asarray(c, dtype=np.float64), (n_verts,)).copy())
        while len(out) < ncomp:
            out.append(np.zeros(n_verts))
        return out
    return [np.full(n_verts, float(data))] * ncomp


def _draw_shader_mesh(img, zbuf, mesh, view, proj, cam_pos,
                      width, height, near, far):
    """Draw one ShaderMaterial mesh (see the section header)."""
    _, glsl_eval = _load_glsl()
    geometry = _get(mesh, "geometry")
    material = _get(mesh, "material")
    positions = _attr_array(geometry, "position")
    n_verts = len(positions)
    indices = _index_array(geometry, n_verts)
    if indices.size < 3:
        return
    tris = indices.reshape(-1, 3)
    valid = np.all((tris >= 0) & (tris < n_verts), axis=1)
    tris = tris[valid]
    if len(tris) == 0:
        return

    world = _world_matrix(mesh)
    builtins = _shader_builtins(view, proj, world, cam_pos)
    compiled = _compile_shader_material(material)
    vprog = compiled["vertex"]
    fprog = compiled["fragment"]

    # ---- vertex stage over all vertices -------------------------------
    inputs = _uniform_inputs(vprog, material, builtins)
    for tname, name, arr in vprog.attributes():
        data = _attribute_data(geometry, name, tname)
        if data is not None:
            inputs[name] = (data, tname)
    ev = glsl_eval.GLSLEvaluator(vprog.items)
    vglobals = ev.run_main((n_verts,), inputs)["globals"]

    clip = np.stack(
        _per_vertex_components(vglobals.get("gl_Position"), n_verts, 4),
        axis=1)
    w_all = clip[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc_all = clip[:, :3] / w_all[:, None]
        invw_all = 1.0 / w_all
    scr_all = np.stack(
        [(ndc_all[:, 0] * 0.5 + 0.5) * width,
         (-ndc_all[:, 1] * 0.5 + 0.5) * height], axis=1)
    depth_all = ndc_all[:, 2]

    # Per-vertex varyings, stacked as one (N, K) attribute block so
    # the clip path can carry them exactly like uv.
    var_decls = [(tname, name)
                 for tname, name, arr in fprog.varyings()]
    var_cols = []   # (start, ncomp) per declared varying
    var_stack = []
    for tname, name in var_decls:
        t = glsl_eval.glsl_type(tname, _prog_structs(fprog))
        ncomp = t[1] if isinstance(t, tuple) else 1
        comps = _per_vertex_components(
            vglobals.get(name), n_verts, ncomp)
        var_cols.append((len(var_stack), ncomp))
        var_stack.extend(comps)
    var_block = (np.stack(var_stack, axis=1) if var_stack
                 else np.zeros((n_verts, 0)))

    # ---- facing / clip classification (host transform) -----------------
    world_pos = positions @ world[:3, :3].T + world[:3, 3]
    view_pos = world_pos @ view[:3, :3].T + view[:3, 3]
    wp_tris = world_pos[tris]
    e1 = wp_tris[:, 1] - wp_tris[:, 0]
    e2 = wp_tris[:, 2] - wp_tris[:, 0]
    nrm = np.stack(
        [e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1],
         e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2],
         e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]], axis=1)
    nlen = np.sqrt(np.sum(nrm * nrm, axis=1))
    centres = wp_tris.mean(axis=1)
    facing = _facing_sign(nrm, centres, cam_pos, view)
    side = _material_side(material)
    valid_tri = nlen > 1e-12
    if side == 0:
        keep = valid_tri & (facing > 0.0)
    elif side == 1:
        keep = valid_tri & (facing < 0.0)
    else:
        keep = valid_tri
    vz = view_pos[tris][:, :, 2]
    inside = vz <= -near
    if far > near:
        inside = inside & (vz >= -far)
    tri_inside = inside.all(axis=1)
    w_ok = (np.abs(w_all[tris]) >= 1e-9).all(axis=1)

    # ---- rasterize into varying fields ---------------------------------
    fields = [np.zeros((height, width)) for _ in var_stack]
    covered = np.zeros((height, width), dtype=bool)

    def raster(scr3, dep3, iw3, var3):
        A, B, C = scr3[0], scr3[1], scr3[2]
        x0 = max(int(np.floor(min(A[0], B[0], C[0]))), 0)
        x1 = min(int(np.ceil(max(A[0], B[0], C[0]))), width - 1)
        y0 = max(int(np.floor(min(A[1], B[1], C[1]))), 0)
        y1 = min(int(np.ceil(max(A[1], B[1], C[1]))), height - 1)
        if x0 > x1 or y0 > y1:
            return
        denom = ((B[1] - C[1]) * (A[0] - C[0])
                 + (C[0] - B[0]) * (A[1] - C[1]))
        if abs(denom) < 1e-12:
            return
        xs = np.arange(x0, x1 + 1, dtype=np.float64)[None, :] + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float64)[:, None] + 0.5
        w0 = ((B[1] - C[1]) * (xs - C[0])
              + (C[0] - B[0]) * (ys - C[1])) / denom
        w1 = ((C[1] - A[1]) * (xs - C[0])
              + (A[0] - C[0]) * (ys - C[1])) / denom
        w2 = 1.0 - w0 - w1
        inside_tri = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not np.any(inside_tri):
            return
        iw = w0 * iw3[0] + w1 * iw3[1] + w2 * iw3[2]
        z = w0 * dep3[0] + w1 * dep3[1] + w2 * dep3[2]
        sub_z = zbuf[y0: y1 + 1, x0: x1 + 1]
        upd = inside_tri & (z < sub_z)
        if not np.any(upd):
            return
        sub_z[upd] = z[upd]
        covered[y0: y1 + 1, x0: x1 + 1][upd] = True
        with np.errstate(divide="ignore", invalid="ignore"):
            for k, field in enumerate(fields):
                interp = (w0 * var3[0, k] * iw3[0]
                          + w1 * var3[1, k] * iw3[1]
                          + w2 * var3[2, k] * iw3[2]) / iw
                sub = field[y0: y1 + 1, x0: x1 + 1]
                sub[upd] = np.where(np.isfinite(interp), interp, 0.0)[upd]

    vw_tris = view_pos[tris]
    for ti in np.nonzero(keep)[0]:
        tri = tris[ti]
        if tri_inside[ti] and w_ok[ti]:
            raster(scr_all[tri], depth_all[tri], invw_all[tri],
                   var_block[tri])
            continue
        poly, poly_var = _clip_with_uv(
            vw_tris[ti], var_block[tri], near, far)
        if len(poly) < 3:
            continue
        poly_np = np.array(poly, dtype=np.float64)
        ones_p = np.ones((len(poly_np), 1))
        clip_p = np.concatenate([poly_np, ones_p], axis=1) @ proj.T
        w_p = clip_p[:, 3]
        if np.any(np.abs(w_p) < 1e-9):
            continue
        ndc_p = clip_p[:, :3] / w_p[:, None]
        scr_p = np.stack(
            [(ndc_p[:, 0] * 0.5 + 0.5) * width,
             (-ndc_p[:, 1] * 0.5 + 0.5) * height], axis=1)
        poly_var_np = np.array(poly_var, dtype=np.float64)
        for k in range(1, len(poly) - 1):
            sel = [0, k, k + 1]
            raster(scr_p[sel], ndc_p[sel, 2], 1.0 / w_p[sel],
                   poly_var_np[sel])
    if not np.any(covered):
        return

    # ---- fragment stage over the screen grid ---------------------------
    frag_builtins = {
        "viewMatrix": (view, "mat4"),
        "cameraPosition": (tuple(float(c) for c in cam_pos), "vec3"),
    }
    inputs = _uniform_inputs(fprog, material, frag_builtins)
    for (tname, name), (start, ncomp) in zip(var_decls, var_cols):
        inputs[name] = (
            tuple(fields[start + j] for j in range(ncomp)), tname)
    ev = glsl_eval.GLSLEvaluator(fprog.items)
    out = ev.run_main((height, width), inputs)
    rgb = _frag_color_rgb(out["globals"].get("gl_FragColor"),
                          height, width)
    mask = covered
    if out["discard"] is not None:
        mask = covered & ~out["discard"]
    if np.any(mask):
        _quantize_masked(img, rgb, mask)


def _frag_color_rgb(value, height, width):
    """A gl_FragColor value -> an (h, w, 3) float64 RGB array."""
    if value is None:
        return np.zeros((height, width, 3))
    data = value[0]
    comps = []
    if isinstance(data, tuple):
        for c in data[:3]:
            comps.append(np.broadcast_to(
                np.asarray(c, dtype=np.float64), (height, width)))
    elif isinstance(data, np.ndarray):
        for k in range(min(3, data.shape[-1])):
            comps.append(np.asarray(data[..., k], dtype=np.float64))
    while len(comps) < 3:
        comps.append(np.zeros((height, width)))
    return np.stack(comps[:3], axis=-1)


def render_shader_pass(material, width, height):
    """Evaluate a ShaderMaterial's fragment stage over the whole
    frame (the EffectComposer ShaderPass path).  Returns
    (rgb, discard): an (h, w, 3) float64 array in 0..1 (unclipped;
    quantisation clips) and the discard mask or None.

    The standard ``vUv`` varying is supplied as the fullscreen-quad
    uv; any other varying is left at the evaluator's zero default."""
    _, glsl_eval = _load_glsl()
    compiled = _compile_shader_material(material)
    fprog = compiled["fragment"]
    builtins = {
        "viewMatrix": (np.eye(4), "mat4"),
        "cameraPosition": ((0.0, 0.0, 0.0), "vec3"),
    }
    inputs = _uniform_inputs(fprog, material, builtins)
    for tname, name, arr in fprog.varyings():
        if name == "vUv":
            yy, xx = np.mgrid[0:height, 0:width]
            u = (xx.astype(np.float64) + 0.5) / width
            v = 1.0 - (yy.astype(np.float64) + 0.5) / height
            inputs[name] = ((u, v), "vec2")
    ev = glsl_eval.GLSLEvaluator(fprog.items)
    out = ev.run_main((height, width), inputs)
    rgb = _frag_color_rgb(out["globals"].get("gl_FragColor"),
                          height, width)
    return rgb, out["discard"]


def palette_frame_to_rgb(frame):
    """A palette-index frame -> float64 RGB in 0..1."""
    idx = np.clip(np.asarray(frame), 0, len(PALETTE) - 1).astype(np.int32)
    return PALETTE[idx] / 255.0


def _quantize_masked(img, rgb, mask):
    """4x4 Bayer cube quantisation at ``mask`` pixels only."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return
    px = np.clip(rgb[ys, xs], 0.0, 1.0)
    dz = _BAYER[ys & 3, xs & 3][:, None]
    q = np.clip(np.floor(px * _CUBE_LV + 0.5 + dz), 0.0, _CUBE_LV).astype(np.int32)
    img[ys, xs] = (CUBE_BASE + (q[:, 0] * 7 + q[:, 1]) * 5 + q[:, 2]).astype(np.uint8)


def quantize_rgb(rgb, mask=None):
    """An (h, w, 3) float RGB array -> a palette-index frame,
    applying the port's per-pixel quantisation (incl. the dark-face
    checker dither rule) at the masked pixels (all by default)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    img = np.zeros(rgb.shape[:2], dtype=np.uint8)
    if mask is None:
        mask = np.ones(rgb.shape[:2], dtype=bool)
    _quantize_masked(img, rgb, mask)
    return img

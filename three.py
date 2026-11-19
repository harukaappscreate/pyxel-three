"""three.py -- a small, original Python re-implementation of the core of the
three.js API (math, scene graph, cameras, geometries, materials, lights and a
WebGLRenderer-compatible facade), written for the threejs-port project.

Only a subset of the API is provided.  The actual drawing is done by
the sibling module ``render.py`` (numpy software rasterizer); this module
imports it lazily inside ``WebGLRenderer.render`` so that the core can be
imported and unit-tested on its own.

Conventions (matching three.js):
  * Right-handed coordinates, column vectors.  A point is transformed as
    ``v' = M @ v``.
  * ``Matrix4`` keeps its data internally as a numpy ``float64`` array in
    ordinary (row-major) mathematical layout, and exposes ``.elements`` as a
    list of 16 floats in three.js order (column-major), exactly like the
    original library.
  * ``BufferGeometry.attributes`` maps names to numpy arrays; vertex
    attributes are stored as ``(N, itemSize)`` float32 arrays and ``.index``
    as a 1-D integer array (or ``None``).
  * Angles for ``PerspectiveCamera.fov`` are degrees (three.js convention);
    ``Euler`` angles are radians.
"""

import array
import math
import random
import time

import numpy as np

__all__ = [
    "Color",
    "Vector3",
    "Euler",
    "Quaternion",
    "Matrix4",
    "Object3D",
    "Group",
    "Scene",
    "Camera",
    "PerspectiveCamera",
    "BufferAttribute",
    "Float32BufferAttribute",
    "BufferGeometry",
    "BoxGeometry",
    "PlaneGeometry",
    "SphereGeometry",
    "CylinderGeometry",
    "ConeGeometry",
    "TorusGeometry",
    "IcosahedronGeometry",
    "TetrahedronGeometry",
    "OctahedronGeometry",
    "DodecahedronGeometry",
    "CircleGeometry",
    "RingGeometry",
    "LatheGeometry",
    "Material",
    "MeshBasicMaterial",
    "MeshLambertMaterial",
    "MeshPhongMaterial",
    "MeshNormalMaterial",
    "Mesh",
    "PointsMaterial",
    "LineBasicMaterial",
    "Points",
    "Line",
    "LineSegments",
    "SpriteMaterial",
    "Sprite",
    "GridHelper",
    "AxesHelper",
    "ArrowHelper",
    "BoxHelper",
    "Box3Helper",
    "DirectionalLightHelper",
    "SpotLightHelper",
    "HemisphereLightHelper",
    "CameraHelper",
    "PolarGridHelper",
    "Light",
    "AmbientLight",
    "DirectionalLight",
    "HemisphereLight",
    "PointLight",
    "SpotLight",
    "FrontSide",
    "BackSide",
    "DoubleSide",
    "Fog",
    "WebGLRenderer",
    "GLTFLoader",
    "LoadingManager",
    "DefaultLoadingManager",
    "MTLLoader",
    "MaterialCreator",
    "ImageLoader",
    "CubeTextureLoader",
    "DataTexture",
    "Vector2",
    "Vector3",
    "Matrix3",
    "Box3",
    "Sphere",
    "Plane",
    "Frustum",
    "SplineCurve",
    "TrackballControls",
    "VRButton",
    "UnrealBloomPass",
    "BloomPass",
    "FilmPass",
    "LinearEncoding",
    "sRGBEncoding",
    "GammaEncoding",
    "RGBEEncoding",
    "LogLuvEncoding",
    "RGBM7Encoding",
    "RGBM16Encoding",
    "RGBDEncoding",
    "mergeBufferGeometries",
    "Clock",
    "Ray",
    "Raycaster",
    "OrbitControls",
]


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------

def _hue2rgb(p, q, t):
    # three.js Color's helper, verbatim.
    if t < 0:
        t += 1
    if t > 1:
        t -= 1
    if t < 1 / 6:
        return p + (q - p) * 6 * t
    if t < 1 / 2:
        return q
    if t < 2 / 3:
        return p + (q - p) * (2 / 3 - t) * 6
    return p


# CSS colour keywords -> hex, transcribed from three.js r187
# src/math/Color.js (_colorKeywords; the CSS Color 4 table).
_COLOR_KEYWORDS = {
    "aliceblue": 0xF0F8FF, "antiquewhite": 0xFAEBD7, "aqua": 0x00FFFF, "aquamarine": 0x7FFFD4,
    "azure": 0xF0FFFF, "beige": 0xF5F5DC, "bisque": 0xFFE4C4, "black": 0x000000,
    "blanchedalmond": 0xFFEBCD, "blue": 0x0000FF, "blueviolet": 0x8A2BE2, "brown": 0xA52A2A,
    "burlywood": 0xDEB887, "cadetblue": 0x5F9EA0, "chartreuse": 0x7FFF00, "chocolate": 0xD2691E,
    "coral": 0xFF7F50, "cornflowerblue": 0x6495ED, "cornsilk": 0xFFF8DC, "crimson": 0xDC143C,
    "cyan": 0x00FFFF, "darkblue": 0x00008B, "darkcyan": 0x008B8B, "darkgoldenrod": 0xB8860B,
    "darkgray": 0xA9A9A9, "darkgreen": 0x006400, "darkgrey": 0xA9A9A9, "darkkhaki": 0xBDB76B,
    "darkmagenta": 0x8B008B, "darkolivegreen": 0x556B2F, "darkorange": 0xFF8C00, "darkorchid": 0x9932CC,
    "darkred": 0x8B0000, "darksalmon": 0xE9967A, "darkseagreen": 0x8FBC8F, "darkslateblue": 0x483D8B,
    "darkslategray": 0x2F4F4F, "darkslategrey": 0x2F4F4F, "darkturquoise": 0x00CED1, "darkviolet": 0x9400D3,
    "deeppink": 0xFF1493, "deepskyblue": 0x00BFFF, "dimgray": 0x696969, "dimgrey": 0x696969,
    "dodgerblue": 0x1E90FF, "firebrick": 0xB22222, "floralwhite": 0xFFFAF0, "forestgreen": 0x228B22,
    "fuchsia": 0xFF00FF, "gainsboro": 0xDCDCDC, "ghostwhite": 0xF8F8FF, "gold": 0xFFD700,
    "goldenrod": 0xDAA520, "gray": 0x808080, "green": 0x008000, "greenyellow": 0xADFF2F,
    "grey": 0x808080, "honeydew": 0xF0FFF0, "hotpink": 0xFF69B4, "indianred": 0xCD5C5C,
    "indigo": 0x4B0082, "ivory": 0xFFFFF0, "khaki": 0xF0E68C, "lavender": 0xE6E6FA,
    "lavenderblush": 0xFFF0F5, "lawngreen": 0x7CFC00, "lemonchiffon": 0xFFFACD, "lightblue": 0xADD8E6,
    "lightcoral": 0xF08080, "lightcyan": 0xE0FFFF, "lightgoldenrodyellow": 0xFAFAD2, "lightgray": 0xD3D3D3,
    "lightgreen": 0x90EE90, "lightgrey": 0xD3D3D3, "lightpink": 0xFFB6C1, "lightsalmon": 0xFFA07A,
    "lightseagreen": 0x20B2AA, "lightskyblue": 0x87CEFA, "lightslategray": 0x778899, "lightslategrey": 0x778899,
    "lightsteelblue": 0xB0C4DE, "lightyellow": 0xFFFFE0, "lime": 0x00FF00, "limegreen": 0x32CD32,
    "linen": 0xFAF0E6, "magenta": 0xFF00FF, "maroon": 0x800000, "mediumaquamarine": 0x66CDAA,
    "mediumblue": 0x0000CD, "mediumorchid": 0xBA55D3, "mediumpurple": 0x9370DB, "mediumseagreen": 0x3CB371,
    "mediumslateblue": 0x7B68EE, "mediumspringgreen": 0x00FA9A, "mediumturquoise": 0x48D1CC, "mediumvioletred": 0xC71585,
    "midnightblue": 0x191970, "mintcream": 0xF5FFFA, "mistyrose": 0xFFE4E1, "moccasin": 0xFFE4B5,
    "navajowhite": 0xFFDEAD, "navy": 0x000080, "oldlace": 0xFDF5E6, "olive": 0x808000,
    "olivedrab": 0x6B8E23, "orange": 0xFFA500, "orangered": 0xFF4500, "orchid": 0xDA70D6,
    "palegoldenrod": 0xEEE8AA, "palegreen": 0x98FB98, "paleturquoise": 0xAFEEEE, "palevioletred": 0xDB7093,
    "papayawhip": 0xFFEFD5, "peachpuff": 0xFFDAB9, "peru": 0xCD853F, "pink": 0xFFC0CB,
    "plum": 0xDDA0DD, "powderblue": 0xB0E0E6, "purple": 0x800080, "rebeccapurple": 0x663399,
    "red": 0xFF0000, "rosybrown": 0xBC8F8F, "royalblue": 0x4169E1, "saddlebrown": 0x8B4513,
    "salmon": 0xFA8072, "sandybrown": 0xF4A460, "seagreen": 0x2E8B57, "seashell": 0xFFF5EE,
    "sienna": 0xA0522D, "silver": 0xC0C0C0, "skyblue": 0x87CEEB, "slateblue": 0x6A5ACD,
    "slategray": 0x708090, "slategrey": 0x708090, "snow": 0xFFFAFA, "springgreen": 0x00FF7F,
    "steelblue": 0x4682B4, "tan": 0xD2B48C, "teal": 0x008080, "thistle": 0xD8BFD8,
    "tomato": 0xFF6347, "turquoise": 0x40E0D0, "violet": 0xEE82EE, "wheat": 0xF5DEB3,
    "white": 0xFFFFFF, "whitesmoke": 0xF5F5F5, "yellow": 0xFFFF00, "yellowgreen": 0x9ACD32,
}


class Color:
    """RGB color with float components in 0..1 (values may exceed 1)."""

    def __init__(self, *args):
        self.r = 1.0
        self.g = 1.0
        self.b = 1.0
        if len(args) == 1:
            self.set(args[0])
        elif len(args) == 3:
            self.setRGB(args[0], args[1], args[2])
        elif len(args) > 0:
            raise TypeError("Color takes 0, 1 or 3 arguments")

    # -- setting ---------------------------------------------------------
    def set(self, value):
        if isinstance(value, Color):
            return self.copy(value)
        if isinstance(value, (int, np.integer)):
            return self.setHex(int(value))
        if isinstance(value, float):
            return self.setHex(int(value))
        if isinstance(value, str):
            return self.setStyle(value)
        if isinstance(value, (tuple, list, np.ndarray)):
            vals = list(value)
            if len(vals) == 3:
                return self.setRGB(vals[0], vals[1], vals[2])
        raise TypeError("unsupported Color value: %r" % (value,))

    def setHex(self, hexint):
        hexint = int(hexint) & 0xFFFFFF
        self.r = ((hexint >> 16) & 0xFF) / 255.0
        self.g = ((hexint >> 8) & 0xFF) / 255.0
        self.b = (hexint & 0xFF) / 255.0
        return self

    def setStyle(self, style):
        # Mirrors three.js Color.setStyle (minus colour management):
        # #hex (3/6/8 digits), rgb()/rgba() with plain or percentage
        # components (alpha accepted and ignored), hsl()/hsla(), and
        # the CSS colour keywords.
        s = style.strip().lower()
        if s in _COLOR_KEYWORDS:
            return self.setHex(_COLOR_KEYWORDS[s])
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            elif len(h) == 8:
                h = h[:6]  # trailing alpha byte ignored
            if len(h) == 6:
                return self.setHex(int(h, 16))
            raise ValueError("unsupported color style: %r" % (style,))
        if (s.startswith("rgb(") or s.startswith("rgba(")) \
                and s.endswith(")"):
            inner = s[s.index("(") + 1:-1]
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) in (3, 4):
                vals = []
                for p in parts[:3]:
                    if p.endswith("%"):
                        vals.append(float(p[:-1]) / 100.0)
                    else:
                        vals.append(float(p) / 255.0)
                return self.setRGB(*vals)
        if (s.startswith("hsl(") or s.startswith("hsla(")) \
                and s.endswith(")"):
            inner = s[s.index("(") + 1:-1]
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) in (3, 4):
                h = float(parts[0])
                if parts[1].endswith("%") and parts[2].endswith("%"):
                    return self.setHSL(
                        h / 360.0,
                        float(parts[1][:-1]) / 100.0,
                        float(parts[2][:-1]) / 100.0)
        raise ValueError("unsupported color style: %r" % (style,))

    def setRGB(self, r, g, b, colorSpace=None):
        # three.js r152+ accepts an optional colour-space argument and
        # converts into the working colour space.  This port does no
        # colour management anywhere (setHex stores raw values too),
        # so the argument is accepted for API compatibility and the
        # components are stored as given.
        self.r = float(r)
        self.g = float(g)
        self.b = float(b)
        return self

    def setHSL(self, h, s, l, colorSpace=None):
        # The three.js algorithm verbatim (colour-space argument
        # accepted and ignored, as in setRGB above).
        h = float(h) % 1.0
        s = _clamp(float(s), 0.0, 1.0)
        l = _clamp(float(l), 0.0, 1.0)
        if s == 0.0:
            self.r = self.g = self.b = l
        else:
            p = l * (1.0 + s) if l <= 0.5 else l + s - l * s
            q = 2.0 * l - p
            self.r = _hue2rgb(q, p, h + 1.0 / 3.0)
            self.g = _hue2rgb(q, p, h)
            self.b = _hue2rgb(q, p, h - 1.0 / 3.0)
        return self

    def getHSL(self, target=None):
        # three.js getHSL: returns (or fills) an {h, s, l} object in
        # 0..1.  A dict stands in for the JS object; examples read
        # .h/.s/.l off it, which the interpreter resolves on dicts.
        r, g, b = self.r, self.g, self.b
        mx = max(r, g, b)
        mn = min(r, g, b)
        l = (mx + mn) / 2.0
        if mx == mn:
            h = s = 0.0
        else:
            d = mx - mn
            s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
            if mx == r:
                h = (g - b) / d + (6.0 if g < b else 0.0)
            elif mx == g:
                h = (b - r) / d + 2.0
            else:
                h = (r - g) / d + 4.0
            h /= 6.0
        if target is None:
            target = {}
        if isinstance(target, dict):
            target["h"], target["s"], target["l"] = h, s, l
        else:
            target.h, target.s, target.l = h, s, l
        return target

    def offsetHSL(self, h, s, l):
        hsl = self.getHSL()
        return self.setHSL(hsl["h"] + h, hsl["s"] + s, hsl["l"] + l)

    def setScalar(self, scalar):
        self.r = self.g = self.b = float(scalar)
        return self

    # -- access ----------------------------------------------------------
    def getHex(self):
        return (
            (int(round(_clamp(self.r, 0.0, 1.0) * 255)) << 16)
            | (int(round(_clamp(self.g, 0.0, 1.0) * 255)) << 8)
            | int(round(_clamp(self.b, 0.0, 1.0) * 255))
        )

    def getHexString(self):
        return "%06x" % self.getHex()

    def toArray(self, array=None, offset=0):
        if array is None:
            return [self.r, self.g, self.b]
        offset = int(offset)
        if isinstance(array, list):
            while len(array) <= offset + 2:
                array.append(0.0)
        array[offset] = self.r
        array[offset + 1] = self.g
        array[offset + 2] = self.b
        return array

    def to_numpy(self):
        return np.array([self.r, self.g, self.b], dtype=np.float64)

    # -- operations ------------------------------------------------------
    def copy(self, other):
        self.r = float(other.r)
        self.g = float(other.g)
        self.b = float(other.b)
        return self

    def clone(self):
        return Color(self.r, self.g, self.b)

    def multiplyScalar(self, s):
        self.r *= s
        self.g *= s
        self.b *= s
        return self

    def multiply(self, other):
        self.r *= float(other.r)
        self.g *= float(other.g)
        self.b *= float(other.b)
        return self

    def add(self, other):
        self.r += other.r
        self.g += other.g
        self.b += other.b
        return self

    def __eq__(self, other):
        return (
            isinstance(other, Color)
            and self.r == other.r
            and self.g == other.g
            and self.b == other.b
        )

    def equals(self, other):
        return (
            isinstance(other, Color)
            and self.r == other.r
            and self.g == other.g
            and self.b == other.b
        )

    def __repr__(self):
        return "Color(%.4g, %.4g, %.4g)" % (self.r, self.g, self.b)


def _as_color(value, default=None):
    """Coerce ints / tuples / Colors into a Color (used by params dicts)."""
    if value is None:
        return default
    if isinstance(value, Color):
        return value
    return Color(value)


# ---------------------------------------------------------------------------
# Vector3
# ---------------------------------------------------------------------------

class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # -- setting ---------------------------------------------------------
    def set(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        return self

    def setScalar(self, s):
        self.x = self.y = self.z = float(s)
        return self

    def setX(self, x):
        self.x = float(x)
        return self

    def setY(self, y):
        self.y = float(y)
        return self

    def setZ(self, z):
        self.z = float(z)
        return self

    def getComponent(self, index):
        return (self.x, self.y, self.z)[int(index)]

    def setComponent(self, index, value):
        if int(index) == 0:
            self.x = float(value)
        elif int(index) == 1:
            self.y = float(value)
        else:
            self.z = float(value)
        return self

    def copy(self, v):
        self.x = float(v.x)
        self.y = float(v.y)
        self.z = float(v.z)
        return self

    def clone(self):
        return Vector3(self.x, self.y, self.z)

    def fromArray(self, arr, offset=0):
        # JS callers pass float offsets (loop counters); Python
        # sequences reject float indices where JS arrays accept them.
        offset = int(offset)
        self.x = float(arr[offset])
        self.y = float(arr[offset + 1])
        self.z = float(arr[offset + 2])
        return self

    def toArray(self, array=None, offset=0):
        if array is None:
            return [self.x, self.y, self.z]
        offset = int(offset)
        if isinstance(array, list):
            while len(array) <= offset + 2:
                array.append(0.0)
        array[offset] = self.x
        array[offset + 1] = self.y
        array[offset + 2] = self.z
        return array

    def to_numpy(self):
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    # -- arithmetic ------------------------------------------------------
    def add(self, v):
        self.x += v.x
        self.y += v.y
        self.z += v.z
        return self

    def addVectors(self, a, b):
        self.x = a.x + b.x
        self.y = a.y + b.y
        self.z = a.z + b.z
        return self

    def addScalar(self, s):
        self.x += s
        self.y += s
        self.z += s
        return self

    def addScaledVector(self, v, s):
        s = float(s)
        self.x += v.x * s
        self.y += v.y * s
        self.z += v.z * s
        return self

    def sub(self, v):
        self.x -= v.x
        self.y -= v.y
        self.z -= v.z
        return self

    def subVectors(self, a, b):
        self.x = a.x - b.x
        self.y = a.y - b.y
        self.z = a.z - b.z
        return self

    def multiply(self, v):
        self.x *= v.x
        self.y *= v.y
        self.z *= v.z
        return self

    def multiplyScalar(self, s):
        self.x *= s
        self.y *= s
        self.z *= s
        return self

    def divideScalar(self, s):
        return self.multiplyScalar(1.0 / s)

    def negate(self):
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v):
        return self.crossVectors(self, v)

    def crossVectors(self, a, b):
        ax, ay, az = a.x, a.y, a.z
        bx, by, bz = b.x, b.y, b.z
        self.x = ay * bz - az * by
        self.y = az * bx - ax * bz
        self.z = ax * by - ay * bx
        return self

    def lengthSq(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self):
        return math.sqrt(self.lengthSq())

    def normalize(self):
        n = self.length()
        return self.divideScalar(n) if n > 0 else self.set(0.0, 0.0, 0.0)

    def distanceTo(self, v):
        return math.sqrt(self.distanceToSquared(v))

    def distanceToSquared(self, v):
        dx = self.x - v.x
        dy = self.y - v.y
        dz = self.z - v.z
        return dx * dx + dy * dy + dz * dz

    def lerp(self, v, t):
        self.x += (v.x - self.x) * t
        self.y += (v.y - self.y) * t
        self.z += (v.z - self.z) * t
        return self

    def lerpVectors(self, a, b, t):
        t = float(t)
        self.x = a.x + (b.x - a.x) * t
        self.y = a.y + (b.y - a.y) * t
        self.z = a.z + (b.z - a.z) * t
        return self

    def clampScalar(self, lo, hi):
        self.x = _clamp(self.x, lo, hi)
        self.y = _clamp(self.y, lo, hi)
        self.z = _clamp(self.z, lo, hi)
        return self

    # -- transforms ------------------------------------------------------
    def applyMatrix4(self, m):
        a = m._a
        x, y, z = self.x, self.y, self.z
        w = a[3, 0] * x + a[3, 1] * y + a[3, 2] * z + a[3, 3]
        if w != 0.0:
            inv = 1.0 / w
        else:
            inv = 1.0
        self.x = (a[0, 0] * x + a[0, 1] * y + a[0, 2] * z + a[0, 3]) * inv
        self.y = (a[1, 0] * x + a[1, 1] * y + a[1, 2] * z + a[1, 3]) * inv
        self.z = (a[2, 0] * x + a[2, 1] * y + a[2, 2] * z + a[2, 3]) * inv
        return self

    def applyQuaternion(self, q):
        # v' = q * v * q^-1, expanded.
        x, y, z = self.x, self.y, self.z
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        ix = qw * x + qy * z - qz * y
        iy = qw * y + qz * x - qx * z
        iz = qw * z + qx * y - qy * x
        iw = -qx * x - qy * y - qz * z
        self.x = ix * qw + iw * -qx + iy * -qz - iz * -qy
        self.y = iy * qw + iw * -qy + iz * -qx - ix * -qz
        self.z = iz * qw + iw * -qz + ix * -qy - iy * -qx
        return self

    def applyEuler(self, e):
        return self.applyQuaternion(Quaternion().setFromEuler(e))

    def transformDirection(self, m):
        """Apply the upper 3x3 of ``m`` and normalize (three.js semantics)."""
        a = m._a
        x, y, z = self.x, self.y, self.z
        self.x = a[0, 0] * x + a[0, 1] * y + a[0, 2] * z
        self.y = a[1, 0] * x + a[1, 1] * y + a[1, 2] * z
        self.z = a[2, 0] * x + a[2, 1] * y + a[2, 2] * z
        return self.normalize()

    def project(self, camera):
        return self.applyMatrix4(camera.matrixWorldInverse).applyMatrix4(
            camera.projectionMatrix
        )

    def unproject(self, camera):
        inv_proj = Matrix4().copy(camera.projectionMatrix).invert()
        return self.applyMatrix4(inv_proj).applyMatrix4(camera.matrixWorld)

    def fromBufferAttribute(self, attribute, index):
        idx = int(index)
        if isinstance(attribute, np.ndarray):
            row = attribute[idx]
            self.x = float(row[0]) if len(row) > 0 else 0.0
            self.y = float(row[1]) if len(row) > 1 else 0.0
            self.z = float(row[2]) if len(row) > 2 else 0.0
        else:
            sz = int(getattr(attribute, "itemSize", 3))
            arr = attribute.array
            base = idx * sz
            self.x = float(arr[base]) if sz > 0 else 0.0
            self.y = float(arr[base + 1]) if sz > 1 else 0.0
            self.z = float(arr[base + 2]) if sz > 2 else 0.0
        return self

    def setFromSpherical(self, s):
        return self.setFromSphericalCoords(s.radius, s.phi, s.theta)

    def setFromSphericalCoords(self, radius, phi, theta):
        sin_phi_r = math.sin(float(phi)) * float(radius)
        self.x = sin_phi_r * math.sin(float(theta))
        self.y = math.cos(float(phi)) * float(radius)
        self.z = sin_phi_r * math.cos(float(theta))
        return self

    def setFromMatrixPosition(self, m):
        a = m._a
        self.x = float(a[0, 3])
        self.y = float(a[1, 3])
        self.z = float(a[2, 3])
        return self

    def setFromMatrixColumn(self, m, index):
        a = m._a
        c = int(index)
        self.x = float(a[0, c])
        self.y = float(a[1, c])
        self.z = float(a[2, c])
        return self

    def setFromMatrixScale(self, m):
        sx = Vector3().setFromMatrixColumn(m, 0).length()
        sy = Vector3().setFromMatrixColumn(m, 1).length()
        sz = Vector3().setFromMatrixColumn(m, 2).length()
        self.x, self.y, self.z = sx, sy, sz
        return self

    def __eq__(self, other):
        return (
            isinstance(other, Vector3)
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        )

    def equals(self, other):
        return (
            isinstance(other, Vector3)
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        )

    def __repr__(self):
        return "Vector3(%.4g, %.4g, %.4g)" % (self.x, self.y, self.z)


# ---------------------------------------------------------------------------
# Euler / Quaternion
# ---------------------------------------------------------------------------

class Euler:
    """Euler angles in radians.  Default order 'XYZ' (three.js convention:
    the rotation matrix is R = Rx * Ry * Rz for order 'XYZ')."""

    def __init__(self, x=0.0, y=0.0, z=0.0, order="XYZ"):
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        self._order = order
        self._on_change = None  # callable, wired by Object3D

    def _notify(self):
        if self._on_change is not None:
            self._on_change()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = float(v)
        self._notify()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = float(v)
        self._notify()

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, v):
        self._z = float(v)
        self._notify()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, v):
        self._order = v
        self._notify()

    def set(self, x, y, z, order=None):
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        if order is not None:
            self._order = order
        self._notify()
        return self

    def copy(self, e):
        return self.set(e.x, e.y, e.z, e.order)

    def clone(self):
        return Euler(self._x, self._y, self._z, self._order)

    def equals(self, e):
        return (
            self._x == e.x
            and self._y == e.y
            and self._z == e.z
            and self._order == e.order
        )

    def setFromQuaternion(self, q, order=None, _notify=True):
        if order is not None:
            self._order = order
        m = Matrix4().makeRotationFromQuaternion(q)
        self._extract_from_matrix(m)
        if _notify:
            self._notify()
        return self

    def setFromRotationMatrix(self, m, order=None):
        if order is not None:
            self._order = order
        self._extract_from_matrix(m)
        self._notify()
        return self

    def _extract_from_matrix(self, m):
        # Only 'XYZ' extraction is implemented; other orders
        # fall back to the XYZ reading of the same rotation matrix.
        a = m._a
        self._y = math.asin(_clamp(a[0, 2], -1.0, 1.0))
        if abs(math.cos(self._y)) > 1e-6:
            self._x = math.atan2(-a[1, 2], a[2, 2])
            self._z = math.atan2(-a[0, 1], a[0, 0])
        else:
            self._x = math.atan2(a[2, 1], a[1, 1])
            self._z = 0.0

    def __repr__(self):
        return "Euler(%.4g, %.4g, %.4g, %r)" % (
            self._x,
            self._y,
            self._z,
            self._order,
        )


def _axis_quaternion(axis, angle):
    half = angle * 0.5
    s = math.sin(half)
    c = math.cos(half)
    if axis == "X":
        return (s, 0.0, 0.0, c)
    if axis == "Y":
        return (0.0, s, 0.0, c)
    return (0.0, 0.0, s, c)


def _quat_multiply(ax, ay, az, aw, bx, by, bz, bw):
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


class Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        self._w = float(w)
        self._on_change = None  # callable, wired by Object3D

    def fromArray(self, array, offset=0):
        offset = int(offset)
        self._x = float(array[offset])
        self._y = float(array[offset + 1])
        self._z = float(array[offset + 2])
        self._w = float(array[offset + 3])
        self._notify()
        return self

    def toArray(self, array=None, offset=0):
        if array is None:
            return [self._x, self._y, self._z, self._w]
        offset = int(offset)
        array[offset] = self._x
        array[offset + 1] = self._y
        array[offset + 2] = self._z
        array[offset + 3] = self._w
        return array

    def _notify(self):
        if self._on_change is not None:
            self._on_change()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = float(v)
        self._notify()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = float(v)
        self._notify()

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, v):
        self._z = float(v)
        self._notify()

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, v):
        self._w = float(v)
        self._notify()

    def set(self, x, y, z, w):
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        self._w = float(w)
        self._notify()
        return self

    def copy(self, q):
        return self.set(q.x, q.y, q.z, q.w)

    def clone(self):
        return Quaternion(self._x, self._y, self._z, self._w)

    def identity(self):
        return self.set(0.0, 0.0, 0.0, 1.0)

    def setFromEuler(self, e, _notify=True):
        # q = q_axis0 * q_axis1 * q_axis2 following the order string, which
        # matches the matrix product R = R_axis0 @ R_axis1 @ R_axis2.
        angles = {"X": e.x, "Y": e.y, "Z": e.z}
        order = e.order or "XYZ"
        result = (0.0, 0.0, 0.0, 1.0)
        for axis in order:
            qa = _axis_quaternion(axis, angles[axis])
            result = _quat_multiply(*(result + qa))
        self._x, self._y, self._z, self._w = result
        if _notify:
            self._notify()
        return self

    def setFromRotationMatrix(self, m):
        a = m._a
        trace = a[0, 0] + a[1, 1] + a[2, 2]
        if trace > 0.0:
            s = 0.5 / math.sqrt(trace + 1.0)
            self._w = 0.25 / s
            self._x = (a[2, 1] - a[1, 2]) * s
            self._y = (a[0, 2] - a[2, 0]) * s
            self._z = (a[1, 0] - a[0, 1]) * s
        elif a[0, 0] > a[1, 1] and a[0, 0] > a[2, 2]:
            s = 2.0 * math.sqrt(1.0 + a[0, 0] - a[1, 1] - a[2, 2])
            self._w = (a[2, 1] - a[1, 2]) / s
            self._x = 0.25 * s
            self._y = (a[0, 1] + a[1, 0]) / s
            self._z = (a[0, 2] + a[2, 0]) / s
        elif a[1, 1] > a[2, 2]:
            s = 2.0 * math.sqrt(1.0 + a[1, 1] - a[0, 0] - a[2, 2])
            self._w = (a[0, 2] - a[2, 0]) / s
            self._x = (a[0, 1] + a[1, 0]) / s
            self._y = 0.25 * s
            self._z = (a[1, 2] + a[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + a[2, 2] - a[0, 0] - a[1, 1])
            self._w = (a[1, 0] - a[0, 1]) / s
            self._x = (a[0, 2] + a[2, 0]) / s
            self._y = (a[1, 2] + a[2, 1]) / s
            self._z = 0.25 * s
        self._notify()
        return self

    def setFromAxisAngle(self, axis, angle):
        half = angle * 0.5
        s = math.sin(half)
        self._x = axis.x * s
        self._y = axis.y * s
        self._z = axis.z * s
        self._w = math.cos(half)
        self._notify()
        return self

    def multiply(self, q):
        return self.multiplyQuaternions(self, q)

    def premultiply(self, q):
        return self.multiplyQuaternions(q, self)

    def multiplyQuaternions(self, a, b):
        self._x, self._y, self._z, self._w = _quat_multiply(
            a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w
        )
        self._notify()
        return self

    def length(self):
        return math.sqrt(self._x ** 2 + self._y ** 2 + self._z ** 2 + self._w ** 2)

    def normalize(self):
        n = self.length()
        if n == 0.0:
            self._x = self._y = self._z = 0.0
            self._w = 1.0
        else:
            inv = 1.0 / n
            self._x *= inv
            self._y *= inv
            self._z *= inv
            self._w *= inv
        self._notify()
        return self

    def conjugate(self):
        self._x = -self._x
        self._y = -self._y
        self._z = -self._z
        self._notify()
        return self

    def invert(self):
        return self.conjugate()

    def dot(self, q):
        return self._x * q.x + self._y * q.y + self._z * q.z + self._w * q.w

    def equals(self, q):
        return (
            self._x == q.x
            and self._y == q.y
            and self._z == q.z
            and self._w == q.w
        )

    def __repr__(self):
        return "Quaternion(%.4g, %.4g, %.4g, %.4g)" % (
            self._x,
            self._y,
            self._z,
            self._w,
        )


# ---------------------------------------------------------------------------
# Matrix4
# ---------------------------------------------------------------------------

class Matrix4:
    """4x4 matrix.  Internal storage ``_a`` is a numpy float64 array in
    row-major mathematical layout; ``elements`` exposes three.js ordering."""

    def __init__(self, *args):
        self._a = np.identity(4, dtype=np.float64)
        if len(args) == 16:
            self.set(*args)
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            self._a = np.array(args[0], dtype=np.float64).reshape(4, 4)

    # -- three.js style element access ------------------------------------
    @property
    def elements(self):
        # column-major flat list, as in three.js
        return [float(v) for v in self._a.flatten(order="F")]

    @elements.setter
    def elements(self, values):
        self._a = np.array(values, dtype=np.float64).reshape(4, 4, order="F")

    def to_numpy(self):
        """Return a copy of the matrix as a (4,4) row-major numpy array."""
        return self._a.copy()

    def __array__(self, dtype=None):
        if dtype is not None:
            return self._a.astype(dtype)
        return self._a

    def set(self, n11, n12, n13, n14, n21, n22, n23, n24,
            n31, n32, n33, n34, n41, n42, n43, n44):
        # arguments are given row by row, as in three.js Matrix4.set()
        self._a = np.array(
            [
                [n11, n12, n13, n14],
                [n21, n22, n23, n24],
                [n31, n32, n33, n34],
                [n41, n42, n43, n44],
            ],
            dtype=np.float64,
        )
        return self

    def fromArray(self, array, offset=0):
        offset = int(offset)
        vals = [float(array[offset + i]) for i in range(16)]
        self._a = np.array(vals, dtype=np.float64).reshape(4, 4, order="F")
        return self

    def toArray(self, array=None, offset=0):
        if array is None:
            return self.elements
        offset = int(offset)
        for i, value in enumerate(self.elements):
            array[offset + i] = value
        return array

    # -- basics ------------------------------------------------------------
    def identity(self):
        self._a = np.identity(4, dtype=np.float64)
        return self

    def copy(self, m):
        self._a = np.array(m._a, dtype=np.float64)
        return self

    def clone(self):
        return Matrix4(self._a.copy())

    def equals(self, m):
        return self.elements == m.elements

    def transpose(self):
        self._a = self._a.T.copy()
        return self

    def determinant(self):
        return float(np.linalg.det(self._a))

    def invert(self):
        try:
            self._a = np.linalg.inv(self._a)
        except np.linalg.LinAlgError:
            self._a = np.zeros((4, 4), dtype=np.float64)
        return self

    def getInverse(self, m=None):
        """Non-standard convenience: return a new inverse (of ``m`` or self)."""
        src = m if m is not None else self
        out = Matrix4(src._a.copy())
        return out.invert()

    # -- multiplication ------------------------------------------------------
    def multiply(self, m):
        return self.multiplyMatrices(self, m)

    def premultiply(self, m):
        return self.multiplyMatrices(m, self)

    def multiplyMatrices(self, a, b):
        self._a = a._a @ b._a
        return self

    def multiplyScalar(self, s):
        self._a = self._a * float(s)
        return self

    # -- construction ---------------------------------------------------------
    def makeTranslation(self, x, y, z):
        self.set(
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1,
        )
        return self

    def makeScale(self, x, y, z):
        self.set(
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1,
        )
        return self

    def makeRotationX(self, theta):
        c, s = math.cos(theta), math.sin(theta)
        self.set(
            1, 0, 0, 0,
            0, c, -s, 0,
            0, s, c, 0,
            0, 0, 0, 1,
        )
        return self

    def makeRotationY(self, theta):
        c, s = math.cos(theta), math.sin(theta)
        self.set(
            c, 0, s, 0,
            0, 1, 0, 0,
            -s, 0, c, 0,
            0, 0, 0, 1,
        )
        return self

    def makeRotationZ(self, theta):
        c, s = math.cos(theta), math.sin(theta)
        self.set(
            c, -s, 0, 0,
            s, c, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        )
        return self

    def makeRotationFromEuler(self, e):
        return self.makeRotationFromQuaternion(Quaternion().setFromEuler(e))

    def makeRotationFromQuaternion(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        x2, y2, z2 = x + x, y + y, z + z
        xx, xy, xz = x * x2, x * y2, x * z2
        yy, yz, zz = y * y2, y * z2, z * z2
        wx, wy, wz = w * x2, w * y2, w * z2
        self.set(
            1 - (yy + zz), xy - wz, xz + wy, 0,
            xy + wz, 1 - (xx + zz), yz - wx, 0,
            xz - wy, yz + wx, 1 - (xx + yy), 0,
            0, 0, 0, 1,
        )
        return self

    def makeRotationFromAxisAngle(self, axis, angle):
        return self.makeRotationFromQuaternion(
            Quaternion().setFromAxisAngle(axis, angle)
        )

    def makePerspective(self, left, right, top, bottom, near, far):
        """Frustum projection, same signature/semantics as current three.js
        (WebGL coordinate system: camera looks down -Z, NDC z in [-1, 1])."""
        dx = right - left
        dy = top - bottom
        dz = far - near
        if abs(dx) < 1e-12:
            dx = 1e-12
        if abs(dy) < 1e-12:
            dy = 1e-12
        if abs(dz) < 1e-12:
            dz = 1e-12
        x = 2.0 * near / dx
        y = 2.0 * near / dy
        a = (right + left) / dx
        b = (top + bottom) / dy
        c = -(far + near) / dz
        d = -2.0 * far * near / dz
        self.set(
            x, 0, a, 0,
            0, y, b, 0,
            0, 0, c, d,
            0, 0, -1, 0,
        )
        return self

    def makeOrthographic(self, left, right, top, bottom, near, far):
        dx = right - left
        dy = top - bottom
        dz = far - near
        w = 1.0 / (dx if abs(dx) >= 1e-12 else 1e-12)
        h = 1.0 / (dy if abs(dy) >= 1e-12 else 1e-12)
        p = 1.0 / (dz if abs(dz) >= 1e-12 else 1e-12)
        x = (right + left) * w
        y = (top + bottom) * h
        z = (far + near) * p
        self.set(
            2 * w, 0, 0, -x,
            0, 2 * h, 0, -y,
            0, 0, -2 * p, -z,
            0, 0, 0, 1,
        )
        return self

    def lookAt(self, eye, target, up):
        """Rotation whose -Z points from ``eye`` toward ``target``
        (three.js Matrix4.lookAt semantics)."""
        z = Vector3().subVectors(eye, target)
        if z.lengthSq() == 0.0:
            z.z = 1.0
        z.normalize()
        x = Vector3().crossVectors(up, z)
        if x.lengthSq() == 0.0:
            if abs(up.z) == 1.0:
                z.x += 0.0001
            else:
                z.z += 0.0001
            z.normalize()
            x = Vector3().crossVectors(up, z)
        x.normalize()
        y = Vector3().crossVectors(z, x)
        self.set(
            x.x, y.x, z.x, 0,
            x.y, y.y, z.y, 0,
            x.z, y.z, z.z, 0,
            0, 0, 0, 1,
        )
        return self

    def compose(self, position, quaternion, scale):
        self.makeRotationFromQuaternion(quaternion)
        a = self._a
        a[:, 0] *= scale.x
        a[:, 1] *= scale.y
        a[:, 2] *= scale.z
        a[0, 3] = position.x
        a[1, 3] = position.y
        a[2, 3] = position.z
        return self

    def decompose(self, position, quaternion, scale):
        a = self._a
        position.set(a[0, 3], a[1, 3], a[2, 3])
        sx = math.sqrt(a[0, 0] ** 2 + a[1, 0] ** 2 + a[2, 0] ** 2)
        sy = math.sqrt(a[0, 1] ** 2 + a[1, 1] ** 2 + a[2, 1] ** 2)
        sz = math.sqrt(a[0, 2] ** 2 + a[1, 2] ** 2 + a[2, 2] ** 2)
        if self.determinant() < 0:
            sx = -sx
        scale.set(sx, sy, sz)
        rot = Matrix4()
        rot._a = a.copy()
        if sx != 0:
            rot._a[:, 0] /= sx
        if sy != 0:
            rot._a[:, 1] /= sy
        if sz != 0:
            rot._a[:, 2] /= sz
        rot._a[:, 3] = 0.0
        rot._a[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
        quaternion.setFromRotationMatrix(rot)
        return self

    def scale(self, v):
        a = self._a
        a[:, 0] *= v.x
        a[:, 1] *= v.y
        a[:, 2] *= v.z
        return self

    def setPosition(self, v):
        if isinstance(v, Vector3):
            self._a[0, 3], self._a[1, 3], self._a[2, 3] = v.x, v.y, v.z
        else:
            self._a[0, 3], self._a[1, 3], self._a[2, 3] = v
        return self

    def getPosition(self):
        return Vector3(self._a[0, 3], self._a[1, 3], self._a[2, 3])

    def extractRotation(self, m):
        out = Matrix4()
        pos = Vector3()
        quat = Quaternion()
        scl = Vector3()
        m.decompose(pos, quat, scl)
        return out.makeRotationFromQuaternion(quat)

    def __repr__(self):
        return "Matrix4(%s)" % (self._a.tolist(),)


# ---------------------------------------------------------------------------
# Object3D / Scene / Camera
# ---------------------------------------------------------------------------

class Object3D:
    isObject3D = True

    _next_id = 1

    def __init__(self, name=""):
        self.name = name
        # three.js assigns every Object3D a process-unique numeric id
        # at construction; getObjectById and editor-style lookups rely
        # on it.
        self.id = Object3D._next_id
        Object3D._next_id += 1
        self.parent = None
        self.children = []
        self.position = Vector3(0.0, 0.0, 0.0)
        self.rotation = Euler(0.0, 0.0, 0.0, "XYZ")
        self.quaternion = Quaternion(0.0, 0.0, 0.0, 1.0)
        self.scale = Vector3(1.0, 1.0, 1.0)
        self._syncing_rotation = False

        def _euler_changed():
            if self._syncing_rotation:
                return
            self._syncing_rotation = True
            try:
                self.quaternion.setFromEuler(self.rotation, _notify=False)
            finally:
                self._syncing_rotation = False

        def _quat_changed():
            if self._syncing_rotation:
                return
            self._syncing_rotation = True
            try:
                self.rotation.setFromQuaternion(
                    self.quaternion, _notify=False
                )
            finally:
                self._syncing_rotation = False

        self.rotation._on_change = _euler_changed
        self.quaternion._on_change = _quat_changed

        self.matrix = Matrix4()
        self.matrixWorld = Matrix4()
        self.normalMatrix = Matrix3()
        self.matrixAutoUpdate = True
        self.matrixWorldNeedsUpdate = False
        self.visible = True
        # Shadow flags (three.js Object3D): a mesh casts into the shadow
        # map when castShadow is set, and is shadow-tested when
        # receiveShadow is set.  Both default to False.
        self.castShadow = False
        self.receiveShadow = False
        self.layers = Layers()
        self.userData = {}

    # -- hierarchy ---------------------------------------------------------
    def add(self, *objects):
        for obj in objects:
            if obj is self:
                continue
            if obj.parent is not None:
                obj.parent.remove(obj)
            obj.parent = self
            self.children.append(obj)
        return self

    def remove(self, *objects):
        for obj in objects:
            if obj in self.children:
                self.children.remove(obj)
                obj.parent = None
        return self

    def clear(self):
        return self.remove(*list(self.children))

    def traverse(self, callback):
        callback(self)
        for child in list(self.children):
            child.traverse(callback)
        return self

    def getObjectByName(self, name):
        """Depth-first search (self first) for the first descendant
        whose ``name`` matches — three.js Object3D.getObjectByName."""
        if self.name == name:
            return self
        for child in self.children:
            found = child.getObjectByName(name)
            if found is not None:
                return found
        return None

    def getObjectById(self, id):
        return self.getObjectByProperty("id", id)

    def getObjectByProperty(self, name, value):
        prop = getattr(self, name, None)
        if prop is not None and prop == value:
            return self
        for child in self.children:
            found = child.getObjectByProperty(name, value)
            if found is not None:
                return found
        return None

    def getObjectsByProperty(self, name, value):
        found = []
        if getattr(self, name, None) == value:
            found.append(self)
        for child in self.children:
            found.extend(child.getObjectsByProperty(name, value))
        return found

    # -- transforms ----------------------------------------------------------
    def updateMatrix(self):
        self.matrix.compose(self.position, self.quaternion, self.scale)
        self.matrixWorldNeedsUpdate = True
        return self

    def applyMatrix4(self, matrix):
        """three.js Object3D.applyMatrix4: premultiply the local
        matrix by ``matrix`` and decompose the result back into
        position/quaternion/scale (the state the renderer reads)."""
        if self.matrixAutoUpdate:
            self.updateMatrix()
        self.matrix.premultiply(matrix)
        self.matrix.decompose(self.position, self.quaternion, self.scale)
        return self

    def updateWorldMatrix(self, updateParents, updateChildren):
        if updateParents and self.parent is not None:
            self.parent.updateWorldMatrix(True, False)
        if self.matrixAutoUpdate:
            self.updateMatrix()
        if self.parent is None:
            self.matrixWorld.copy(self.matrix)
        else:
            self.matrixWorld.multiplyMatrices(
                self.parent.matrixWorld, self.matrix
            )
        self.matrixWorldNeedsUpdate = False
        if updateChildren:
            for child in self.children:
                child.updateWorldMatrix(False, True)
        return self

    def updateMatrixWorld(self, force=False):
        if self.matrixAutoUpdate:
            self.updateMatrix()
        if self.matrixWorldNeedsUpdate or force:
            if self.parent is None:
                self.matrixWorld.copy(self.matrix)
            else:
                self.matrixWorld.multiplyMatrices(
                    self.parent.matrixWorld, self.matrix
                )
            self.matrixWorldNeedsUpdate = False
            force = True
        for child in self.children:
            child.updateMatrixWorld(force)
        return self

    def getWorldPosition(self, target=None):
        if target is None:
            target = Vector3()
        self.updateWorldMatrix(True, False)
        return target.set(
            self.matrixWorld._a[0, 3],
            self.matrixWorld._a[1, 3],
            self.matrixWorld._a[2, 3],
        )

    def getWorldQuaternion(self, target=None):
        if target is None:
            target = Quaternion()
        pos = Vector3()
        scl = Vector3()
        self.updateWorldMatrix(True, False)
        self.matrixWorld.decompose(pos, target, scl)
        return target

    def getWorldScale(self, target=None):
        if target is None:
            target = Vector3()
        pos = Vector3()
        quat = Quaternion()
        self.updateWorldMatrix(True, False)
        self.matrixWorld.decompose(pos, quat, target)
        return target

    def lookAt(self, *args):
        if len(args) == 1 and isinstance(args[0], Vector3):
            target = args[0]
        elif len(args) == 3:
            target = Vector3(args[0], args[1], args[2])
        else:
            raise TypeError("lookAt takes a Vector3 or x, y, z")
        world_pos = self.getWorldPosition(Vector3())
        m = Matrix4()
        if getattr(self, "isCamera", False) or getattr(self, "isLight", False):
            m.lookAt(world_pos, target, self.up)
        else:
            m.lookAt(target, world_pos, self.up)
        self.quaternion.setFromRotationMatrix(m)
        if self.parent is not None:
            parent_quat = self.parent.getWorldQuaternion(Quaternion())
            parent_quat.invert()
            self.quaternion.premultiply(parent_quat)
        return self

    @property
    def up(self):
        if not hasattr(self, "_up"):
            self._up = Vector3(0.0, 1.0, 0.0)
        return self._up

    @up.setter
    def up(self, v):
        self._up = v

    def rotateX(self, angle):
        return self._rotateOnAxis(Vector3(1, 0, 0), angle)

    def rotateY(self, angle):
        return self._rotateOnAxis(Vector3(0, 1, 0), angle)

    def rotateZ(self, angle):
        return self._rotateOnAxis(Vector3(0, 0, 1), angle)

    def rotateOnAxis(self, axis, angle):
        return self._rotateOnAxis(axis, angle)

    def _rotateOnAxis(self, axis, angle):
        q = Quaternion().setFromAxisAngle(axis, angle)
        self.quaternion.multiply(q)
        return self

    def translateX(self, distance):
        return self._translateOnAxis(Vector3(1, 0, 0), distance)

    def translateY(self, distance):
        return self._translateOnAxis(Vector3(0, 1, 0), distance)

    def translateZ(self, distance):
        return self._translateOnAxis(Vector3(0, 0, 1), distance)

    def translateOnAxis(self, axis, distance):
        return self._translateOnAxis(axis, distance)

    def _translateOnAxis(self, axis, distance):
        v = axis.clone().applyQuaternion(self.quaternion)
        self.position.add(v.multiplyScalar(distance))
        return self

    def copy(self, source, recursive=True):
        self.name = source.name
        self.up.copy(source.up)
        self.position.copy(source.position)
        self.rotation.order = source.rotation.order
        self.quaternion.copy(source.quaternion)
        self.scale.copy(source.scale)
        self.matrix.copy(source.matrix)
        self.matrixWorld.copy(source.matrixWorld)
        self.matrixAutoUpdate = source.matrixAutoUpdate
        self.matrixWorldNeedsUpdate = source.matrixWorldNeedsUpdate
        self.visible = source.visible
        self.castShadow = source.castShadow
        self.receiveShadow = source.receiveShadow
        self.userData = dict(source.userData) if isinstance(source.userData, dict) else {}
        if recursive:
            for child in source.children:
                self.add(child.clone())
        return self

    def clone(self, recursive=True):
        return type(self)().copy(self, recursive)

    def addEventListener(self, type, listener, *args, **kwargs):
        if not hasattr(self, "_listeners"):
            self._listeners = {}
        self._listeners.setdefault(str(type), []).append(listener)

    def removeEventListener(self, type, listener, *args, **kwargs):
        if hasattr(self, "_listeners"):
            callbacks = self._listeners.get(str(type), [])
            if listener in callbacks:
                callbacks.remove(listener)

    def dispatchEvent(self, event):
        if hasattr(self, "_listeners") and isinstance(event, dict):
            for cb in list(self._listeners.get(str(event.get("type", "")), [])):
                if callable(cb):
                    cb(event)

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.name)


class Group(Object3D):
    isGroup = True


class Scene(Object3D):
    isScene = True

    def __init__(self, name=""):
        super().__init__(name)
        self._background = None
        self.fog = None
        # Environment map: a CubeTexture whose reflection is
        # mixed into Standard/Phong specular by the renderer.  Plain
        # attribute, exactly like three.js Scene.environment.
        self.environment = None

    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, value):
        # three.js Scene.background accepts a Color, a Texture or a
        # CubeTexture (or null); anything else (hex int, style
        # string) is coerced to a Color.
        if value is None or isinstance(value, Color) \
                or getattr(value, "isTexture", False) \
                or getattr(value, "isCubeTexture", False):
            self._background = value
        else:
            self._background = Color(value)


class Camera(Object3D):
    isCamera = True

    def __init__(self, name=""):
        super().__init__(name)
        self.projectionMatrix = Matrix4()
        self.matrixWorldInverse = Matrix4()

    def updateMatrixWorld(self, force=False):
        super().updateMatrixWorld(force)
        self.matrixWorldInverse.copy(self.matrixWorld).invert()
        return self


class PerspectiveCamera(Camera):
    isPerspectiveCamera = True

    def __init__(self, fov=50, aspect=1.0, near=0.1, far=2000, name=""):
        super().__init__(name)
        self.fov = float(fov)          # degrees, three.js convention
        self.aspect = float(aspect)
        self.near = float(near)
        self.far = float(far)
        self.zoom = 1.0
        self.updateProjectionMatrix()

    def updateProjectionMatrix(self):
        near = self.near
        top = near * math.tan(math.radians(0.5 * self.fov)) / self.zoom
        height = 2.0 * top
        width = self.aspect * height
        left = -0.5 * width
        self.projectionMatrix.makePerspective(
            left, left + width, top, top - height, near, self.far
        )
        return self

    def setFocalLength(self, focal):
        # 35mm full-frame height is 24mm, as in three.js
        self.fov = 2.0 * math.degrees(math.atan(12.0 / float(focal)))
        self.updateProjectionMatrix()
        return self


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class BufferAttribute:
    """Minimal BufferAttribute: wraps a numpy array plus an item size."""

    def __init__(self, array, itemSize, normalized=False):
        arr = np.asarray(array)
        if arr.dtype != np.float32 and not np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32)
        self.array = arr
        self.itemSize = int(itemSize)
        self.normalized = bool(normalized)

    @property
    def count(self):
        if self.itemSize == 0:
            return 0
        return int(self.array.size // self.itemSize)

    def __len__(self):
        return self.count

    def setUsage(self, usage):
        """Record the intended GPU usage pattern (three.js setUsage).

        The software renderer re-reads the array every frame, so the
        hint has no behavioural effect; it is stored for inspection
        and the attribute is returned for chaining."""
        self.usage = usage
        return self

    def onUpload(self, callback):
        """Register the upload callback (three.js onUpload).

        There is no GPU upload in this port; the callback is stored
        (as ``onUploadCallback``, the property the original sets) and
        the attribute is returned for chaining, which is the use the
        official examples make of it."""
        self.onUploadCallback = callback
        return self

    def copyAt(self, index1, attribute, index2):
        i1 = int(index1) * self.itemSize
        i2 = int(index2) * int(getattr(attribute, "itemSize", self.itemSize))
        src = attribute.array
        for k in range(self.itemSize):
            self.array[i1 + k] = src[i2 + k]
        return self


class Float32BufferAttribute(BufferAttribute):
    def __init__(self, array, itemSize, normalized=False):
        super().__init__(np.asarray(array, dtype=np.float32), itemSize, normalized)
        self.array = self.array.astype(np.float32, copy=False)


class Int8BufferAttribute(BufferAttribute):
    def __init__(self, array, itemSize, normalized=False):
        super().__init__(np.asarray(array, dtype=np.int8), itemSize, normalized)


class Uint8BufferAttribute(BufferAttribute):
    def __init__(self, array, itemSize, normalized=False):
        super().__init__(np.asarray(array, dtype=np.uint8), itemSize, normalized)


class Uint8ClampedBufferAttribute(BufferAttribute):
    def __init__(self, array, itemSize, normalized=False):
        arr = np.clip(np.asarray(array, dtype=np.float64), 0, 255).astype(np.uint8)
        super().__init__(arr, itemSize, normalized)


class Int16BufferAttribute(BufferAttribute):
    def __init__(self, array, itemSize, normalized=False):
        super().__init__(np.asarray(array, dtype=np.int16), itemSize, normalized)


class Uint16BufferAttribute(BufferAttribute):
    def __init__(self, array, itemSize, normalized=False):
        super().__init__(np.asarray(array, dtype=np.uint16), itemSize, normalized)


class Int32BufferAttribute(BufferAttribute):
    def __init__(self, array, itemSize, normalized=False):
        super().__init__(np.asarray(array, dtype=np.int32), itemSize, normalized)


class Uint32BufferAttribute(BufferAttribute):
    def __init__(self, array, itemSize, normalized=False):
        super().__init__(np.asarray(array, dtype=np.uint32), itemSize, normalized)


_ATTRIBUTE_ITEM_SIZES = {
    "position": 3,
    "normal": 3,
    "color": 3,
    "uv": 2,
    "uv1": 2,
}


def _coerce_attribute(name, value):
    """Normalize an attribute value to a numpy array.

    Vertex attributes (position/normal/color/uv) are stored shaped
    ``(N, itemSize)``; anything else is stored as a flat 1-D array.
    """
    if isinstance(value, BufferAttribute):
        arr = np.asarray(value.array)
        if getattr(value, "normalized", False):
            dt = arr.dtype
            if dt == np.uint8:
                arr = arr.astype(np.float32) / np.float32(255.0)
            elif dt == np.int8:
                arr = np.maximum(arr.astype(np.float32) / np.float32(127.0), np.float32(-1.0))
            elif dt == np.uint16:
                arr = arr.astype(np.float32) / np.float32(65535.0)
            elif dt == np.int16:
                arr = np.maximum(arr.astype(np.float32) / np.float32(32767.0), np.float32(-1.0))
        item = value.itemSize
        if item > 1:
            arr = arr.reshape(-1, item)
        return arr
    arr = np.asarray(value)
    item = _ATTRIBUTE_ITEM_SIZES.get(name)
    if item and arr.ndim == 1 and arr.size % item == 0:
        arr = arr.reshape(-1, item)
    return arr


class BufferGeometry:
    isBufferGeometry = True

    def __init__(self):
        self.attributes = {}
        self._index = None
        self.boundingBox = None
        self.boundingSphere = None
        self.userData = {}
        self.drawRange = {"start": 0, "count": float("inf")}
        self.groups = []
        self.morphAttributes = {}

    # -- index ---------------------------------------------------------------
    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        if value is None:
            self._index = None
        elif isinstance(value, BufferAttribute):
            self._index = np.asarray(value.array).reshape(-1).astype(np.int64)
        else:
            self._index = np.asarray(value).reshape(-1).astype(np.int64)

    def setIndex(self, index):
        self.index = index
        return self

    def getIndex(self):
        return self._index

    # -- attributes --------------------------------------------------------------
    def setAttribute(self, name, attribute):
        self.attributes[name] = _coerce_attribute(name, attribute)
        return self

    def getAttribute(self, name):
        return self.attributes.get(name)

    def hasAttribute(self, name):
        return name in self.attributes

    def deleteAttribute(self, name):
        self.attributes.pop(name, None)
        return self

    # -- draw range / groups ------------------------------------------------
    def setDrawRange(self, start=0, count=float("inf")):
        """Restrict rendering to a window of the index/vertex stream
        (three.js setDrawRange).  render.py applies the window in
        both the triangle path and the points/lines path."""
        self.drawRange = {"start": float(start), "count": float(count)}
        return self

    def dispose(self):
        """three.js dispose(): releases GPU resources.  The software
        renderer holds none per geometry, so this only records that
        disposal happened (memory-test samples call it in a loop)."""
        self._disposed = True

    def addGroup(self, start, count, materialIndex=0):
        """Record a geometry group (three.js addGroup).  Groups are
        stored for inspection; the renderer draws the geometry with
        the mesh's material as a whole (no per-group material split).
        """
        group = {"start": float(start), "count": float(count),
                 "materialIndex": int(materialIndex)}
        self.groups.append(group)
        return group

    def clearGroups(self):
        self.groups = []
        return self

    # -- copy / clone ---------------------------------------------------------
    def copy(self, source):
        """Deep-copy another BufferGeometry's state (three.js copy):
        attribute arrays and the index are copied, not shared."""
        self.attributes = {
            name: (arr.copy() if hasattr(arr, "copy") else arr)
            for name, arr in source.attributes.items()
        }
        self._index = None if source._index is None \
            else np.array(source._index, copy=True)
        self.drawRange = dict(source.drawRange)
        self.groups = [dict(g) for g in source.groups]
        self.userData = dict(source.userData) \
            if isinstance(source.userData, dict) else source.userData
        for field in ("boundingBox", "boundingSphere"):
            value = getattr(source, field, None)
            if value is not None and hasattr(value, "clone"):
                value = value.clone()
            setattr(self, field, value)
        return self

    def clone(self):
        """A new geometry of the same class with copied state.  The
        instance is created without running the subclass constructor
        (whose parameters the copy would overwrite anyway)."""
        new = object.__new__(type(self))
        BufferGeometry.__init__(new)
        new.copy(self)
        return new

    @property
    def position(self):
        return self.attributes.get("position")

    def setFromPoints(self, points):
        """three.js ``BufferGeometry.setFromPoints``: build the position
        attribute from a list of Vector3 (or Vector2, z defaults to 0)
        points and return this geometry."""
        coords = []
        for p in points:
            coords.append(float(p.x))
            coords.append(float(p.y))
            z = getattr(p, "z", 0.0)
            coords.append(float(z) if z is not None else 0.0)
        self.setAttribute(
            "position",
            Float32BufferAttribute(np.asarray(coords, dtype=np.float32), 3),
        )
        return self

    # -- helpers ---------------------------------------------------------------------
    def computeVertexNormals(self):
        pos = self.attributes.get("position")
        if pos is None:
            return self
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        normals = np.zeros_like(pos)
        if self._index is not None:
            tris = np.asarray(self._index).reshape(-1, 3)
        else:
            tris = np.arange(pos.shape[0]).reshape(-1, 3)
        a = pos[tris[:, 0]]
        b = pos[tris[:, 1]]
        c = pos[tris[:, 2]]
        face = np.cross(b - a, c - a)
        for k in range(3):
            np.add.at(normals, tris[:, k], face)
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0.0] = 1.0
        normals = (normals / lengths[:, None]).astype(np.float32)
        self.attributes["normal"] = normals
        return self

    def computeBoundingBox(self):
        pos = self.attributes.get("position")
        if pos is None:
            self.boundingBox = (Vector3(), Vector3())
            return self
        arr = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        if len(arr):
            lo = arr.min(axis=0)
            hi = arr.max(axis=0)
        else:
            lo = hi = np.zeros(3, dtype=np.float64)
        self.boundingBox = (
            Vector3(lo[0], lo[1], lo[2]),
            Vector3(hi[0], hi[1], hi[2]),
        )
        return self

    def computeBoundingSphere(self):
        pos = self.attributes.get("position")
        if pos is None:
            self.boundingSphere = (Vector3(), 0.0)
            return self
        arr = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        center = arr.mean(axis=0)
        radius = float(np.linalg.norm(arr - center, axis=1).max()) if len(arr) else 0.0
        self.boundingSphere = (
            Vector3(center[0], center[1], center[2]),
            radius,
        )
        return self

    def applyMatrix4(self, m):
        pos = self.attributes.get("position")
        if pos is not None:
            arr = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
            hom = np.concatenate(
                [arr, np.ones((arr.shape[0], 1), dtype=np.float64)], axis=1
            )
            out = hom @ m._a.T
            w = out[:, 3:4]
            w[w == 0.0] = 1.0
            self.attributes["position"] = (out[:, :3] / w).astype(np.float32)
        nrm = self.attributes.get("normal")
        if nrm is not None:
            normal_matrix = np.linalg.inv(m._a[:3, :3]).T
            arr = np.asarray(nrm, dtype=np.float64).reshape(-1, 3)
            out = arr @ normal_matrix.T
            lengths = np.linalg.norm(out, axis=1)
            lengths[lengths == 0.0] = 1.0
            self.attributes["normal"] = (out / lengths[:, None]).astype(np.float32)
        return self

    def translate(self, x, y, z):
        pos = self.attributes.get("position")
        if pos is not None:
            arr = np.asarray(pos).reshape(-1, 3).copy()
            arr[:, 0] += x
            arr[:, 1] += y
            arr[:, 2] += z
            self.attributes["position"] = arr
        return self

    def scale(self, x, y, z):
        pos = self.attributes.get("position")
        if pos is not None:
            arr = np.asarray(pos).reshape(-1, 3).copy()
            arr[:, 0] *= x
            arr[:, 1] *= y
            arr[:, 2] *= z
            self.attributes["position"] = arr
        nrm = self.attributes.get("normal")
        if nrm is not None:
            self.computeVertexNormals()
        return self

    def rotateX(self, angle):
        return self.applyMatrix4(Matrix4().makeRotationX(angle))

    def rotateY(self, angle):
        return self.applyMatrix4(Matrix4().makeRotationY(angle))

    def rotateZ(self, angle):
        return self.applyMatrix4(Matrix4().makeRotationZ(angle))

    def toNonIndexed(self):
        if self._index is None:
            return self
        out = BufferGeometry()
        idx = np.asarray(self._index)
        for name, arr in self.attributes.items():
            arr2 = np.asarray(arr)
            if arr2.ndim >= 1 and arr2.shape[0] == self._vertex_count():
                out.attributes[name] = arr2[idx].copy()
            else:
                out.attributes[name] = arr2
        return out

    def _vertex_count(self):
        pos = self.attributes.get("position")
        if pos is None:
            return 0
        return int(np.asarray(pos).reshape(-1, 3).shape[0])

    def __repr__(self):
        return "BufferGeometry(vertices=%d)" % self._vertex_count()


def _make_box_face(origin, u, v, normal, w, h):
    """One box face: 4 corners starting at ``origin`` going along u then v."""
    o = np.array(origin, dtype=np.float32)
    uu = np.array(u, dtype=np.float32) * np.float32(w)
    vv = np.array(v, dtype=np.float32) * np.float32(h)
    corners = np.stack([o, o + uu, o + uu + vv, o + vv]).astype(np.float32)
    normals = np.tile(np.array(normal, dtype=np.float32), (4, 1))
    return corners, normals


class BoxGeometry(BufferGeometry):
    """Box with configurable segments per axis (three.js BoxGeometry)."""

    def __init__(self, width=1, height=1, depth=1,
                 widthSegments=1, heightSegments=1, depthSegments=1):
        super().__init__()
        w, h, d = float(width), float(height), float(depth)
        ws = max(1, int(widthSegments))
        hs = max(1, int(heightSegments))
        ds = max(1, int(depthSegments))
        hw, hh, hd = w / 2.0, h / 2.0, d / 2.0
        # (origin, u, v, normal, size_u, size_v, seg_u, seg_v) per face, CCW from outside
        faces = [
            # +z
            ((-hw, -hh, hd), (1, 0, 0), (0, 1, 0), (0, 0, 1), w, h, ws, hs),
            # -z
            ((hw, -hh, -hd), (-1, 0, 0), (0, 1, 0), (0, 0, -1), w, h, ws, hs),
            # +x
            ((hw, -hh, hd), (0, 0, -1), (0, 1, 0), (1, 0, 0), d, h, ds, hs),
            # -x
            ((-hw, -hh, -hd), (0, 0, 1), (0, 1, 0), (-1, 0, 0), d, h, ds, hs),
            # +y
            ((-hw, hh, hd), (1, 0, 0), (0, 0, -1), (0, 1, 0), w, d, ws, ds),
            # -y
            ((-hw, -hh, -hd), (1, 0, 0), (0, 0, 1), (0, -1, 0), w, d, ws, ds),
        ]
        positions = []
        normals = []
        uvs = []
        idx = []
        v_offset = 0
        for origin, u, v, n, su, sv, seg_u, seg_v in faces:
            o = np.array(origin, dtype=np.float32)
            uu = np.array(u, dtype=np.float32) * np.float32(su)
            vv = np.array(v, dtype=np.float32) * np.float32(sv)
            nn = np.array(n, dtype=np.float32)
            for iy in range(seg_v + 1):
                ty = iy / seg_v
                for ix in range(seg_u + 1):
                    tx = ix / seg_u
                    positions.append(o + uu * np.float32(tx) + vv * np.float32(ty))
                    normals.append(nn)
                    uvs.append((tx, ty))
            stride = seg_u + 1
            for iy in range(seg_v):
                for ix in range(seg_u):
                    a = v_offset + iy * stride + ix
                    b = v_offset + iy * stride + (ix + 1)
                    c = v_offset + (iy + 1) * stride + (ix + 1)
                    d_idx = v_offset + (iy + 1) * stride + ix
                    idx += [a, b, c, a, c, d_idx]
            v_offset += (seg_u + 1) * (seg_v + 1)
        self.attributes["position"] = np.asarray(positions, dtype=np.float32)
        self.attributes["normal"] = np.asarray(normals, dtype=np.float32)
        self.attributes["uv"] = np.asarray(uvs, dtype=np.float32)
        self.index = np.asarray(idx, dtype=np.int32)


# three.js names the constructor arguments (width, height, depth); expose a
# friendly alias.
class PlaneGeometry(BufferGeometry):
    """Plane in the XY plane facing +Z, centered on the origin.

    The 1x1 case keeps the original four-vertex layout; segment
    counts above 1 build the three.js grid (rows from +Y down,
    uv (0, 1) at the top-left)."""

    def __init__(self, width=1, height=1, widthSegments=1,
                 heightSegments=1):
        super().__init__()
        hw, hh = float(width) / 2.0, float(height) / 2.0
        wseg = max(1, int(widthSegments))
        hseg = max(1, int(heightSegments))
        if wseg == 1 and hseg == 1:
            self.attributes["position"] = np.array(
                [
                    [-hw, -hh, 0.0],
                    [hw, -hh, 0.0],
                    [hw, hh, 0.0],
                    [-hw, hh, 0.0],
                ],
                dtype=np.float32,
            )
            self.attributes["normal"] = np.tile(
                np.array([0.0, 0.0, 1.0], dtype=np.float32), (4, 1)
            )
            self.attributes["uv"] = np.array(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                dtype=np.float32
            )
            self.index = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
            return
        positions = []
        uvs = []
        for iy in range(hseg + 1):
            y = hh - (float(height) * iy / hseg)
            for ix in range(wseg + 1):
                x = -hw + (float(width) * ix / wseg)
                positions.append([x, y, 0.0])
                uvs.append([ix / wseg, 1.0 - iy / hseg])
        indices = []
        row = wseg + 1
        for iy in range(hseg):
            for ix in range(wseg):
                a = ix + row * iy
                b = ix + row * (iy + 1)
                c = (ix + 1) + row * (iy + 1)
                d = (ix + 1) + row * iy
                indices += [a, b, d, b, c, d]
        count = len(positions)
        self.attributes["position"] = np.array(positions, dtype=np.float32)
        self.attributes["normal"] = np.tile(
            np.array([0.0, 0.0, 1.0], dtype=np.float32), (count, 1)
        )
        self.attributes["uv"] = np.array(uvs, dtype=np.float32)
        self.index = np.array(indices, dtype=np.int64)


class SphereGeometry(BufferGeometry):
    """UV sphere.  Default segments: wseg=12, hseg=9.
    The full three.js argument set (phi/theta start and length) is
    honoured: partial spheres come out as open shells, exactly the
    shapes the tank / panorama lessons build."""

    def __init__(self, radius=1, widthSegments=12, heightSegments=9,
                 phiStart=0.0, phiLength=None, thetaStart=0.0,
                 thetaLength=None):
        super().__init__()
        r = float(radius)
        wseg = max(3, int(widthSegments))
        hseg = max(2, int(heightSegments))
        phi_start = float(phiStart)
        phi_length = (2.0 * math.pi if phiLength is None
                      else float(phiLength))
        theta_start = float(thetaStart)
        theta_length = (math.pi if thetaLength is None
                        else float(thetaLength))
        self.parameters = {
            "radius": r,
            "widthSegments": wseg,
            "heightSegments": hseg,
            "phiStart": phi_start,
            "phiLength": phi_length,
            "thetaStart": theta_start,
            "thetaLength": theta_length,
        }
        positions = []
        normals = []
        uvs = []
        for iy in range(hseg + 1):
            theta = theta_start + theta_length * iy / hseg
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            for ix in range(wseg + 1):
                phi = phi_start + phi_length * ix / wseg
                x = -r * math.cos(phi) * sin_t
                y = r * cos_t
                z = r * math.sin(phi) * sin_t
                positions.append([x, y, z])
                n = np.array([x, y, z], dtype=np.float64)
                ln = np.linalg.norm(n)
                if ln > 0:
                    n = n / ln
                normals.append(n.tolist())
                uvs.append([ix / wseg, 1.0 - iy / hseg])
        self.attributes["position"] = np.array(positions, dtype=np.float32)
        self.attributes["normal"] = np.array(normals, dtype=np.float32)
        self.attributes["uv"] = np.array(uvs, dtype=np.float32)
        grid_x = wseg + 1
        idx = []
        for iy in range(hseg):
            for ix in range(wseg):
                a = ix + grid_x * iy
                b = ix + grid_x * (iy + 1)
                c = (ix + 1) + grid_x * (iy + 1)
                d = (ix + 1) + grid_x * iy
                # The pole rows degenerate to a point; three.js
                # emits only the non-degenerate triangle there.
                top_pole = theta_start == 0.0
                bottom_pole = (theta_start + theta_length) >= math.pi
                if iy != 0 or not top_pole:
                    idx += [a, b, d]
                if iy != hseg - 1 or not bottom_pole:
                    idx += [b, c, d]
        self.index = np.array(idx, dtype=np.int64)


class CylinderGeometry(BufferGeometry):
    """Cylinder / frustum (three.js CylinderGeometry).

    The side is a grid of (radialSegments+1) x (heightSegments+1)
    vertices.  Each cap whose radius is > 0 (and the geometry is not
    open-ended) adds a centre vertex plus a ring of radialSegments+1
    vertices with flat cap normals.  All vertices carry uv.
    """

    def __init__(self, radiusTop=1, radiusBottom=1, height=1,
                 radialSegments=16, heightSegments=1, openEnded=False):
        super().__init__()
        r_top = float(radiusTop)
        r_bottom = float(radiusBottom)
        h = float(height)
        seg = max(3, int(radialSegments))
        hseg = max(1, int(heightSegments))
        positions = []
        normals = []
        uvs = []
        idx = []

        # ---- side -----------------------------------------------------
        # v goes 0 at the top to 1 at the bottom.  With slope
        # k = r_bottom - r_top the outward side normal is proportional
        # to (h*sin(phi), k, h*cos(phi)).
        slope = r_bottom - r_top
        for iy in range(hseg + 1):
            v = iy / hseg
            radius = r_top + slope * v
            y = h / 2.0 - v * h
            for ix in range(seg + 1):
                phi = 2.0 * math.pi * ix / seg
                sin_p = math.sin(phi)
                cos_p = math.cos(phi)
                positions.append([radius * sin_p, y, radius * cos_p])
                n = np.array([h * sin_p, slope, h * cos_p], dtype=np.float64)
                ln = np.linalg.norm(n)
                if ln > 0:
                    n = n / ln
                normals.append(n.tolist())
                uvs.append([ix / seg, 1.0 - v])
        grid = seg + 1
        for iy in range(hseg):
            for ix in range(seg):
                a = ix + grid * iy
                b = ix + grid * (iy + 1)
                c = (ix + 1) + grid * (iy + 1)
                d = (ix + 1) + grid * iy
                idx += [a, b, d, b, c, d]

        # ---- caps -------------------------------------------------------
        def add_cap(top):
            radius = r_top if top else r_bottom
            if openEnded or radius <= 0.0:
                return
            sign = 1.0 if top else -1.0
            y = sign * h / 2.0
            centre = len(positions)
            positions.append([0.0, y, 0.0])
            normals.append([0.0, sign, 0.0])
            uvs.append([0.5, 0.5])
            ring = len(positions)
            for ix in range(seg + 1):
                phi = 2.0 * math.pi * ix / seg
                sin_p = math.sin(phi)
                cos_p = math.cos(phi)
                positions.append([radius * sin_p, y, radius * cos_p])
                normals.append([0.0, sign, 0.0])
                uvs.append([0.5 + 0.5 * sin_p, 0.5 + 0.5 * cos_p * sign])
            for ix in range(seg):
                if top:
                    idx.extend([centre, ring + ix, ring + ix + 1])
                else:
                    idx.extend([centre, ring + ix + 1, ring + ix])

        add_cap(True)
        add_cap(False)

        self.attributes["position"] = np.array(positions, dtype=np.float32)
        self.attributes["normal"] = np.array(normals, dtype=np.float32)
        self.attributes["uv"] = np.array(uvs, dtype=np.float32)
        self.index = np.array(idx, dtype=np.int64)


class ConeGeometry(CylinderGeometry):
    """Cone: a cylinder whose top radius is zero (three.js ConeGeometry)."""

    def __init__(self, radius=1, height=1, radialSegments=16,
                 heightSegments=1, openEnded=False):
        super().__init__(0.0, radius, height, radialSegments,
                         heightSegments, openEnded)


class TorusGeometry(BufferGeometry):
    """Torus lying in the XY plane (three.js TorusGeometry)."""

    def __init__(self, radius=1, tube=0.4, radialSegments=10,
                 tubularSegments=20, arc=2.0 * math.pi):
        super().__init__()
        r = float(radius)
        tb = float(tube)
        rseg = max(3, int(radialSegments))
        tseg = max(3, int(tubularSegments))
        positions = []
        normals = []
        uvs = []
        idx = []
        for j in range(rseg + 1):
            v = 2.0 * math.pi * j / rseg
            cos_v = math.cos(v)
            sin_v = math.sin(v)
            for i in range(tseg + 1):
                u = float(arc) * i / tseg
                cos_u = math.cos(u)
                sin_u = math.sin(u)
                ring = r + tb * cos_v
                positions.append([ring * cos_u, ring * sin_u, tb * sin_v])
                normals.append([cos_v * cos_u, cos_v * sin_u, sin_v])
                uvs.append([i / tseg, j / rseg])
        grid = tseg + 1
        for j in range(rseg):
            for i in range(tseg):
                a = i + grid * j
                b = i + grid * (j + 1)
                c = (i + 1) + grid * (j + 1)
                d = (i + 1) + grid * j
                idx += [a, d, b, b, d, c]
        self.attributes["position"] = np.array(positions, dtype=np.float32)
        self.attributes["normal"] = np.array(normals, dtype=np.float32)
        self.attributes["uv"] = np.array(uvs, dtype=np.float32)
        self.index = np.array(idx, dtype=np.int64)


def _apply_polyhedron(geometry, vertices, faces, radius, detail):
    """Fill ``geometry`` as a Platonic solid (three.js PolyhedronGeometry):
    subdivide each triangle edge into ``detail + 1`` segments, project
    onto the sphere of ``radius``, and compute normals & spherical UVs."""
    base_verts = [np.array(v, dtype=np.float64) for v in vertices]
    base_tris = [tuple(int(i) for i in f) for f in faces]
    d = max(0, int(detail))
    if d == 0:
        verts = base_verts
        tris = base_tris
    else:
        cols = d + 1
        verts = []
        v_cache = {}

        def _get_vert(p):
            n = float(np.linalg.norm(p))
            u = p / n if n > 1e-12 else p
            key = (round(float(u[0]), 6), round(float(u[1]), 6), round(float(u[2]), 6))
            idx = v_cache.get(key)
            if idx is None:
                idx = len(verts)
                v_cache[key] = idx
                verts.append(u)
            return idx

        tris = []
        for ia, ib, ic in base_tris:
            a, b, c = base_verts[ia], base_verts[ib], base_verts[ic]
            grid = []
            for i in range(cols + 1):
                row = []
                aj = a + (c - a) * (i / cols)
                bj = b + (c - b) * (i / cols)
                num = cols - i
                for j in range(num + 1):
                    if j == 0 and i == cols:
                        p = c
                    else:
                        p = aj + (bj - aj) * (j / num) if num > 0 else aj
                    row.append(_get_vert(p))
                grid.append(row)
            for i in range(cols):
                for j in range(2 * (cols - i) - 1):
                    k = j // 2
                    if j % 2 == 0:
                        tris.append((grid[i][k], grid[i][k + 1], grid[i + 1][k]))
                    else:
                        tris.append((grid[i][k + 1], grid[i + 1][k + 1], grid[i + 1][k]))
    r = float(radius)
    pos_list = []
    for v in verts:
        n = float(np.linalg.norm(v))
        pos_list.append((v / n * r) if n > 1e-12 else v * 0.0)
    pos = np.array(pos_list, dtype=np.float64)
    if r != 0.0:
        normals = (pos / r).astype(np.float32)
        uvs = np.array(
            [
                [
                    math.atan2(p[2], p[0]) / (2.0 * math.pi) + 0.5,
                    math.asin(_clamp(p[1] / r, -1.0, 1.0)) / math.pi + 0.5,
                ]
                for p in pos
            ],
            dtype=np.float32,
        )
    else:
        normals = np.zeros_like(pos, dtype=np.float32)
        uvs = np.zeros((len(pos), 2), dtype=np.float32)
    geometry.attributes["position"] = pos.astype(np.float32)
    geometry.attributes["normal"] = normals
    geometry.attributes["uv"] = uvs
    geometry.index = np.array(
        [i for tri in tris for i in tri], dtype=np.int32
    )
    return geometry


class PolyhedronGeometry(BufferGeometry):
    """Generic polyhedron (three.js PolyhedronGeometry(vertices, indices, radius, detail))."""

    def __init__(self, vertices=None, indices=None, radius=1.0, detail=0):
        super().__init__()
        if not vertices or not indices:
            return
        v_raw = list(vertices)
        i_raw = list(indices)
        verts = [(float(v_raw[k]), float(v_raw[k + 1]), float(v_raw[k + 2]))
                 for k in range(0, len(v_raw) - 2, 3)]
        faces = [(int(i_raw[k]), int(i_raw[k + 1]), int(i_raw[k + 2]))
                 for k in range(0, len(i_raw) - 2, 3)]
        _apply_polyhedron(self, verts, faces, radius, detail)


class IcosahedronGeometry(BufferGeometry):
    """Regular icosahedron (three.js IcosahedronGeometry)."""

    def __init__(self, radius=1, detail=0):
        super().__init__()
        t = (1.0 + math.sqrt(5.0)) / 2.0
        vertices = [
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
        ]
        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]
        _apply_polyhedron(self, vertices, faces, radius, detail)


class TetrahedronGeometry(BufferGeometry):
    """Regular tetrahedron (three.js TetrahedronGeometry)."""

    def __init__(self, radius=1, detail=0):
        super().__init__()
        vertices = [
            (1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1),
        ]
        faces = [
            (2, 1, 0), (0, 3, 2), (1, 3, 0), (2, 3, 1),
        ]
        _apply_polyhedron(self, vertices, faces, radius, detail)


class OctahedronGeometry(BufferGeometry):
    """Regular octahedron (three.js OctahedronGeometry)."""

    def __init__(self, radius=1, detail=0):
        super().__init__()
        vertices = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ]
        faces = [
            (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
            (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),
        ]
        _apply_polyhedron(self, vertices, faces, radius, detail)


class DodecahedronGeometry(BufferGeometry):
    """Regular dodecahedron (three.js DodecahedronGeometry)."""

    def __init__(self, radius=1, detail=0):
        super().__init__()
        t = (1.0 + math.sqrt(5.0)) / 2.0
        r = 1.0 / t
        vertices = [
            (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
            (0, -r, -t), (0, -r, t), (0, r, -t), (0, r, t),
            (-r, -t, 0), (-r, t, 0), (r, -t, 0), (r, t, 0),
            (-t, 0, -r), (t, 0, -r), (-t, 0, r), (t, 0, r),
        ]
        faces = [
            (3, 11, 7), (3, 7, 15), (3, 15, 13),
            (7, 19, 17), (7, 17, 6), (7, 6, 15),
            (17, 4, 8), (17, 8, 10), (17, 10, 6),
            (8, 0, 16), (8, 16, 2), (8, 2, 10),
            (0, 12, 1), (0, 1, 18), (0, 18, 16),
            (6, 10, 2), (6, 2, 13), (6, 13, 15),
            (2, 16, 18), (2, 18, 3), (2, 3, 13),
            (18, 1, 9), (18, 9, 11), (18, 11, 3),
            (4, 14, 12), (4, 12, 0), (4, 0, 8),
            (11, 9, 5), (11, 5, 19), (11, 19, 7),
            (19, 5, 14), (19, 14, 4), (19, 4, 17),
            (1, 12, 14), (1, 14, 5), (1, 5, 9),
        ]
        _apply_polyhedron(self, vertices, faces, radius, detail)


class CircleGeometry(BufferGeometry):
    """Flat disc in the XY plane facing +Z (three.js CircleGeometry)."""

    def __init__(self, radius=1, segments=16):
        super().__init__()
        r = float(radius)
        seg = max(3, int(segments))
        positions = [[0.0, 0.0, 0.0]]
        normals = [[0.0, 0.0, 1.0]]
        uvs = [[0.5, 0.5]]
        for s in range(seg + 1):
            theta = 2.0 * math.pi * s / seg
            x = r * math.sin(theta)
            y = r * math.cos(theta)
            positions.append([x, y, 0.0])
            normals.append([0.0, 0.0, 1.0])
            uvs.append([x / (2.0 * r) + 0.5 if r != 0.0 else 0.5,
                        y / (2.0 * r) + 0.5 if r != 0.0 else 0.5])
        idx = []
        for i in range(1, seg + 1):
            idx += [0, i + 1, i]
        self.attributes["position"] = np.array(positions, dtype=np.float32)
        self.attributes["normal"] = np.array(normals, dtype=np.float32)
        self.attributes["uv"] = np.array(uvs, dtype=np.float32)
        self.index = np.array(idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Materials / Mesh
# ---------------------------------------------------------------------------

def _split_params(params, kwargs):
    merged = {}
    if params is not None:
        if isinstance(params, dict):
            merged.update(params)
        else:
            raise TypeError("material params must be a dict")
    merged.update(kwargs)
    return merged


class Material:
    isMaterial = True

    def __init__(self, params=None, **kwargs):
        p = _split_params(params, kwargs)
        self.color = _as_color(p.get("color"), Color(1.0, 1.0, 1.0))
        self.visible = bool(p.get("visible", True))
        self.opacity = float(p.get("opacity", 1.0))
        self.transparent = bool(p.get("transparent", False))
        self.side = p.get("side", 0)  # 0=FrontSide, 1=BackSide, 2=DoubleSide
        # Use the geometry's 'color' attribute for per-vertex colours
        # (three.js Material.vertexColors; the helpers rely on it).
        self.vertexColors = bool(p.get("vertexColors", False))
        self.name = p.get("name", "")
        self.userData = {}
        # Texture map (a Texture or None); honoured by the renderer for
        # MeshBasicMaterial / MeshLambertMaterial.
        self.map = p.get("map", None)
        # Draw only the triangle edges when True (all materials).
        self.wireframe = bool(p.get("wireframe", False))

    def setValues(self, values):
        for key, value in (values or {}).items():
            if key == "color":
                self.color = _as_color(value, self.color)
            elif hasattr(self, key):
                setattr(self, key, value)
        return self

    def copy(self, other):
        self.color = other.color.clone() if other.color else None
        self.map = getattr(other, "map", None)
        self.visible = other.visible
        self.opacity = other.opacity
        self.transparent = other.transparent
        self.side = other.side
        self.vertexColors = getattr(other, "vertexColors", False)
        return self

    def clone(self):
        return type(self)().copy(self)

    def dispose(self):
        """three.js dispose(): releases GPU programs; the software
        renderer holds none per material, so this is a record-only
        no-op (memory-test samples call it in a loop)."""
        self._disposed = True


class MeshBasicMaterial(Material):
    isMeshBasicMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        self.wireframe = bool(p.get("wireframe", False))


class MeshLambertMaterial(Material):
    isMeshLambertMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        self.emissive = _as_color(p.get("emissive"), Color(0.0, 0.0, 0.0))
        self.emissiveIntensity = float(p.get("emissiveIntensity", 1.0))


class MeshPhongMaterial(Material):
    """Blinn-Phong material (three.js MeshPhongMaterial).  The renderer
    approximates the specular term per face (flat shading)."""

    isMeshPhongMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        self.emissive = _as_color(p.get("emissive"), Color(0.0, 0.0, 0.0))
        self.emissiveIntensity = float(p.get("emissiveIntensity", 1.0))
        self.specular = _as_color(p.get("specular"), Color(0x111111))
        self.shininess = float(p.get("shininess", 30.0))


class MeshNormalMaterial(Material):
    """Material that colours faces by their normal (three.js
    MeshNormalMaterial).  Lighting and ``color`` do not apply."""

    isMeshNormalMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)


class Mesh(Object3D):
    isMesh = True

    def __init__(self, geometry=None, material=None, name=""):
        super().__init__(name)
        self.geometry = geometry if geometry is not None else BufferGeometry()
        self.material = material if material is not None else MeshBasicMaterial()

    def copy(self, source, recursive=True):
        super().copy(source, recursive)
        self.geometry = source.geometry
        self.material = source.material
        return self

    def computeBoundingSphere(self):
        if self.geometry is not None and hasattr(self.geometry, "computeBoundingSphere"):
            self.geometry.computeBoundingSphere()
            self.boundingSphere = self.geometry.boundingSphere
        return self.boundingSphere


# ---------------------------------------------------------------------------
# Fog and material side constants
# ---------------------------------------------------------------------------

# Material.side values, matching three.js.
FrontSide = 0
BackSide = 1
DoubleSide = 2


class Fog:
    """Linear fog (three.js THREE.Fog): fragments are blended toward
    ``color`` between ``near`` and ``far`` view depth.  Assign to
    ``scene.fog``."""

    isFog = True

    def __init__(self, color=0xFFFFFF, near=1.0, far=2000.0):
        self.color = _as_color(color, Color(1.0, 1.0, 1.0))
        self.near = float(near)
        self.far = float(far)

    def clone(self):
        return Fog(self.color.clone() if self.color else None,
                   self.near, self.far)

    def copy(self, other):
        self.color = other.color.clone() if other.color else None
        self.near = other.near
        self.far = other.far
        return self


class FogExp2:
    """Exponential squared fog (three.js THREE.FogExp2)."""

    isFogExp2 = True

    def __init__(self, color=0xFFFFFF, density=0.00025):
        self.color = _as_color(color, Color(1.0, 1.0, 1.0))
        self.density = float(density)

    def clone(self):
        return FogExp2(self.color.clone() if self.color else None, self.density)

    def copy(self, other):
        self.color = other.color.clone() if other.color else None
        self.density = float(other.density)
        return self


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------

class LightShadow:
    """three.js LightShadow: the shadow camera plus map settings.

    The port renders shadows for one directional light with its own
    internal camera; this object exists so sample code
    that reads or tunes ``light.shadow.camera`` (near/far GUI
    helpers in the manual's shadow lessons) and ``light.shadow``
    fields runs as written."""

    isLightShadow = True

    def __init__(self, camera):
        self.camera = camera
        self.mapSize = Vector2(512.0, 512.0)
        self.bias = 0.0
        self.normalBias = 0.0
        self.radius = 1.0
        self.map = None


class Light(Object3D):
    isLight = True

    def __init__(self, color=0xFFFFFF, intensity=1.0, name=""):
        super().__init__(name)
        self.color = _as_color(color, Color(1.0, 1.0, 1.0))
        self.intensity = float(intensity)


class AmbientLight(Light):
    isAmbientLight = True

    def __init__(self, color=0xFFFFFF, intensity=1.0, name=""):
        super().__init__(color, intensity, name)


class DirectionalLight(Light):
    isDirectionalLight = True

    def __init__(self, color=0xFFFFFF, intensity=1.0, name=""):
        super().__init__(color, intensity, name)
        # three.js default position is (1, 1, 1); the light shines from
        # ``position`` toward ``target`` (default the origin).
        self.position.set(1.0, 1.0, 1.0)
        self.target = Object3D()
        # three.js Light.castShadow: this light produces a shadow map
        # when the renderer's shadowMap is enabled (One
        # directional light only).
        self.castShadow = False
        self.shadow = LightShadow(
            OrthographicCamera(-5.0, 5.0, 5.0, -5.0, 0.5, 500.0))


class HemisphereLight(Light):
    isHemisphereLight = True

    def __init__(self, skyColor=0xFFFFFF, groundColor=0xFFFFFF,
                 intensity=1.0, name=""):
        # Light.color holds the sky colour, as in three.js.
        super().__init__(skyColor, intensity, name)
        self.groundColor = _as_color(groundColor, Color(1.0, 1.0, 1.0))
        # three.js default position is (0, 1, 0); the renderer derives
        # the hemisphere axis from the normalised position.
        self.position.set(0.0, 1.0, 0.0)


class PointLight(Light):
    isPointLight = True

    def __init__(self, color=0xFFFFFF, intensity=1.0, distance=0.0,
                 decay=2.0, name=""):
        super().__init__(color, intensity, name)
        # three.js: distance == 0 means no cutoff; decay is the
        # physical falloff exponent (default 2).
        self.distance = float(distance)
        self.decay = float(decay)
        # Accepted for API compatibility; the renderer implements
        # shadows for one directional light only.
        self.castShadow = False
        self.shadow = LightShadow(
            PerspectiveCamera(90.0, 1.0, 0.5, 500.0))


class SpotLight(Light):
    isSpotLight = True

    def __init__(self, color=0xFFFFFF, intensity=1.0, distance=0.0,
                 angle=math.pi / 3.0, penumbra=0.0, decay=2.0, name=""):
        super().__init__(color, intensity, name)
        self.distance = float(distance)
        # three.js: angle is the cone half-angle in radians (max
        # pi / 2); penumbra is the soft-edge fraction in [0, 1].
        self.angle = float(angle)
        self.penumbra = float(penumbra)
        self.decay = float(decay)
        # The light shines from ``position`` toward ``target``
        # (default the origin), like DirectionalLight.
        self.target = Object3D()
        self.castShadow = False
        self.shadow = LightShadow(
            PerspectiveCamera(50.0, 1.0, 0.5, 500.0))


# ---------------------------------------------------------------------------
# WebGLRenderer facade
# ---------------------------------------------------------------------------

class _DummyElement:
    """Stand-in for the canvas DOM element (three.js ``domElement``)."""

    def __init__(self):
        self.width = 300
        self.height = 150
        self.style = {}
        self.tagName = "canvas"
        self.children = []
        # Event listeners registered from JS ({type: [callback, ...]});
        # the host dispatches synthetic pointer events through the
        # interpreter (see jsinterp.JSInterpreter.dispatch_event).
        self._listeners = {}

    def appendChild(self, child):
        self.children.append(child)
        return child

    def setAttribute(self, name, value):
        setattr(self, name, value)

    def addEventListener(self, type, listener, *args, **kwargs):
        self._listeners.setdefault(str(type), []).append(listener)

    def removeEventListener(self, type, listener, *args, **kwargs):
        callbacks = self._listeners.get(str(type), [])
        if listener in callbacks:
            callbacks.remove(listener)


class _ShadowMapSettings:
    """Stand-in for three.js ``renderer.shadowMap``: ``enabled`` turns
    shadow mapping on, ``type`` is accepted for API compatibility (the
    port implements one fixed hard-shadow filter)."""

    def __init__(self):
        self.enabled = False
        self.type = 1  # PCFShadowMap in three.js; informational only


class _XRManagerStub:
    """Inert XR/VR manager (``renderer.xr`` / legacy ``renderer.vr``):
    the flags exist and can be flipped by page code, but no session
    ever presents -- ``isPresenting`` stays False, which is the state
    the WebVR lessons branch on for their non-VR path."""

    def __init__(self):
        self.enabled = False
        self.isPresenting = False
        self._listeners = {}
        self._controllers = {}

    def getController(self, index=0):
        idx = int(index)
        if idx not in self._controllers:
            self._controllers[idx] = Group()
        return self._controllers[idx]

    def getControllerGrip(self, index=0):
        return self.getController(index)

    def addEventListener(self, type, listener, *args):
        self._listeners.setdefault(str(type), []).append(listener)

    def removeEventListener(self, type, listener, *args):
        callbacks = self._listeners.get(str(type), [])
        if listener in callbacks:
            callbacks.remove(listener)

    def getSession(self):
        return None

    def setSession(self, session):
        return None

    def getReferenceSpace(self):
        return None

    def setReferenceSpaceType(self, type):
        return None

    def getCamera(self, camera=None):
        return camera

    def updateCamera(self, camera):
        return None


class _CapabilitiesStub:
    """Minimal WebGLCapabilities stub for renderer.capabilities."""

    def __init__(self):
        self.isWebGL2 = True
        self.precision = "highp"
        self.logarithmicDepthBuffer = False
        self.maxTextures = 16
        self.maxVertexTextures = 16
        self.maxTextureSize = 4096
        self.maxCubemapSize = 4096
        self.maxAttributes = 16
        self.maxVertexUniforms = 256
        self.maxVaryings = 16
        self.maxFragmentUniforms = 256
        self.vertexTextures = True
        self.floatFragmentTextures = True
        self.floatVertexTextures = True

    def getMaxAnisotropy(self):
        return 1

    def getMaxPrecision(self, precision="highp"):
        return str(precision)


class _GLContextStub:
    """The slice of WebGLRenderingContext page code reads off
    ``renderer.getContext()``: drawing-buffer size and inert
    capability queries."""

    def __init__(self, width, height):
        self.drawingBufferWidth = int(width)
        self.drawingBufferHeight = int(height)
        self.canvas = None

    def getParameter(self, pname):
        return 0

    def getExtension(self, name):
        return None

    def getShaderPrecisionFormat(self, shaderType, precisionType):
        return {"rangeMin": 127, "rangeMax": 127, "precision": 23}

    def getContextAttributes(self):
        return {}


class WebGLRenderer:
    isWebGLRenderer = True

    def __init__(self, params=None, **kwargs):
        p = _split_params(params, kwargs)
        self.domElement = _DummyElement()
        self.width = 256
        self.height = 192
        self.shadowMap = _ShadowMapSettings()
        self.capabilities = _CapabilitiesStub()
        canvas = p.get("canvas")
        if canvas is not None:
            self.domElement = canvas
        self.frame = None  # uint8 (h, w) palette-index array after render()
        self._clear_color = Color(0.0, 0.0, 0.0)
        self._clear_alpha = 1.0
        self._animation_loop = None
        self._viewport = None  # None = full canvas (see setViewport)
        self._scissor = None
        self._scissor_test = False
        self._render_target = None
        # XR/VR session state: there is no headset behind this host,
        # so the managers exist as inert property bags -- the WebVR
        # lessons set ``renderer.vr.enabled = true`` and read it back
        # to decide whether to offer VR, and must not crash doing so.
        self.xr = _XRManagerStub()
        self.vr = self.xr
        self.outputColorSpace = "srgb"
        self.toneMapping = 0
        # Lights mode, with three.js's own historical defaults:
        # legacy lights (useLegacyLights = true /
        # physicallyCorrectLights = false), as every three.js
        # release before r155 defaulted and as the manual lessons
        # (r132 generation) are written against.  Page code that
        # opts into physical lights -- the manual's physically
        # correct lesson sets renderer.physicallyCorrectLights =
        # true -- flips the mode through the properties below, and
        # render() forwards it to render.render_scene.
        self._legacy_lights = True
        if "width" in p and "height" in p:
            self.setSize(p["width"], p["height"])

    # -- size / state ------------------------------------------------------
    def setSize(self, width, height, updateStyle=True):
        self.width = int(width)
        self.height = int(height)
        self.domElement.width = self.width
        self.domElement.height = self.height
        # three.js setSize also resets the viewport and scissor
        # rectangles to the full canvas.
        self._viewport = None
        self._scissor = None
        return self

    def getSize(self, target=None):
        if target is not None and hasattr(target, "set"):
            target.set(self.width, self.height)
            return target
        return {"width": self.width, "height": self.height}

    def getDrawingBufferSize(self, target=None):
        if target is not None and hasattr(target, "set"):
            target.set(float(self.width), float(self.height))
            return target
        return Vector2(float(self.width), float(self.height))

    def setEffects(self, *args, **kwargs):
        return None

    def setPixelRatio(self, ratio):
        self._pixel_ratio = float(ratio)
        return self

    def getPixelRatio(self):
        return getattr(self, "_pixel_ratio", 1.0)

    def setClearColor(self, color, alpha=1.0):
        self._clear_color = _as_color(color, self._clear_color)
        self._clear_alpha = float(alpha)
        return self

    def getClearColor(self):
        return self._clear_color

    def setAnimationLoop(self, callback):
        self._animation_loop = callback
        return self

    def getAnimationLoop(self):
        return self._animation_loop

    # -- render targets / context --------------------------------------
    def setRenderTarget(self, renderTarget, activeCubeFaceIndex=0,
                        activeMipmapLevel=0):
        self._render_target = renderTarget
        return self

    def getRenderTarget(self):
        return self._render_target

    def getContext(self):
        """A minimal GL-context-shaped object: the attributes page
        code reads (drawing buffer size) plus no-op queries."""
        return _GLContextStub(self.width, self.height)

    def readRenderTargetPixels(self, renderTarget, x, y, width,
                               height, buffer, activeCubeFaceIndex=0):
        """Copy a rectangle out of a render target into ``buffer``
        as RGBA bytes, GL-style (origin bottom-left, rows bottom-up)
        -- the screenshot / readback lessons use this."""
        frame = getattr(renderTarget, "frame_rgb", None)
        width = int(width)
        height = int(height)
        x = int(x)
        y = int(y)
        if frame is None or width <= 0 or height <= 0:
            return buffer
        target_h = int(renderTarget.height)
        region = np.zeros((height, width, 4), dtype=np.uint8)
        region[:, :, 3] = 255
        for row in range(height):
            src_y = target_h - 1 - (y + row)
            if src_y < 0 or src_y >= frame.shape[0]:
                continue
            src = frame[src_y]
            x0 = max(0, x)
            x1 = min(frame.shape[1], x + width)
            if x1 <= x0:
                continue
            pixels = np.clip(
                np.round(src[x0:x1] * 255.0), 0, 255).astype(np.uint8)
            region[row, x0 - x:x1 - x, :3] = pixels
            region[row, x0 - x:x1 - x, 3] = 255
        flat = region.tobytes()
        data = getattr(buffer, "_data", None)
        if data is not None and getattr(data, "typecode", None) == "B":
            data[0:len(flat)] = array.array("B", flat)
            return buffer
        try:
            for i, value in enumerate(flat):
                buffer[i] = value
        except Exception:
            pass
        return buffer

    # -- lights mode (legacy / physical) --------------------------------
    @property
    def useLegacyLights(self):
        return self._legacy_lights

    @useLegacyLights.setter
    def useLegacyLights(self, value):
        self._legacy_lights = bool(value)

    @property
    def physicallyCorrectLights(self):
        return not self._legacy_lights

    @physicallyCorrectLights.setter
    def physicallyCorrectLights(self, value):
        self._legacy_lights = not bool(value)

    # -- viewport / scissor / clear -------------------------------------------
    # three.js coordinates for both rectangles are GL-style: origin at
    # the BOTTOM-left of the canvas.  ``None`` means "full canvas"
    # (the state setSize restores).  render() honours the pair by
    # rendering the scene into a viewport-sized buffer and compositing
    # only the active clip rectangle into the persistent frame, which
    # also gives multi-scene samples (multiple_elements and friends)
    # their accumulate-into-one-canvas behaviour.
    def setViewport(self, x, y, width, height):
        self._viewport = (float(x), float(y), float(width), float(height))
        return self

    def getViewport(self, target=None):
        vx, vy, vw, vh = self._viewport or (0.0, 0.0, float(self.width),
                                            float(self.height))
        if target is None:
            target = {}
        if isinstance(target, dict):
            target["x"], target["y"] = vx, vy
            target["width"], target["height"] = vw, vh
        else:
            target.x, target.y, target.z, target.w = vx, vy, vw, vh
        return target

    def setScissor(self, x, y, width, height):
        self._scissor = (float(x), float(y), float(width), float(height))
        return self

    def getScissor(self, target=None):
        sx, sy, sw, sh = self._scissor or (0.0, 0.0, float(self.width),
                                           float(self.height))
        if target is None:
            target = {}
        if isinstance(target, dict):
            target["x"], target["y"] = sx, sy
            target["width"], target["height"] = sw, sh
        else:
            target.x, target.y, target.z, target.w = sx, sy, sw, sh
        return target

    def setScissorTest(self, enabled):
        self._scissor_test = bool(enabled)
        return self

    def getScissorTest(self):
        return bool(getattr(self, "_scissor_test", False))

    def _clip_rect(self):
        """Active clip rectangle in frame coordinates (row 0 = top):
        the scissor rectangle while the scissor test is on, else the
        viewport rectangle, else the full frame."""
        if getattr(self, "_scissor_test", False) and \
                getattr(self, "_scissor", None) is not None:
            x, y, w, h = self._scissor
        elif getattr(self, "_viewport", None) is not None:
            x, y, w, h = self._viewport
        else:
            return 0, 0, self.width, self.height
        x0 = int(round(x))
        y0 = int(round(self.height - y - h))
        return x0, y0, int(round(w)), int(round(h))

    def _ensure_frame(self):
        if self.frame is None or \
                self.frame.shape != (self.height, self.width):
            self.frame = np.zeros((self.height, self.width),
                                  dtype=np.uint8)
        return self.frame

    def clear(self, color=True, depth=True, stencil=True):
        """Fill the active clip region with the clear colour (the
        depth/stencil flags are accepted and ignored: the port keeps
        no separate buffers between render calls)."""
        if not color:
            return self
        render_module = self._load_render_module()
        frame = self._ensure_frame()
        idx = render_module._palette_index(
            (self._clear_color.r, self._clear_color.g, self._clear_color.b))
        x0, y0, w, h = self._clip_rect()
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(self.width, x0 + w), min(self.height, y0 + h)
        if xb > xa and yb > ya:
            frame[ya:yb, xa:xb] = idx
        return self

    # -- rendering -------------------------------------------------------------
    def render(self, scene, camera):
        """Draw ``scene`` from ``camera`` via render.py and keep the result
        (a uint8 (height, width) palette-index array) in ``self.frame``."""
        render_module = self._load_render_module()
        shadows = bool(getattr(self.shadowMap, "enabled", False))
        legacy = bool(getattr(self, "_legacy_lights", True))
        target = getattr(self, "_render_target", None)
        if target is not None:
            # Render into the target at its own size; the screen
            # frame is left untouched, as in three.js.  The target's
            # texture image is updated so materials sampling it (and
            # readRenderTargetPixels) see this frame.
            palette = render_module.render_scene(
                scene, camera, int(target.width), int(target.height),
                shadow_enabled=shadows,
                legacy_lights=legacy,
            )
            target.frame_rgb = render_module.palette_frame_to_rgb(
                palette)
            target.texture.image = target.frame_rgb
            return self.frame
        if getattr(self, "_viewport", None) is None:
            self.frame = render_module.render_scene(
                scene, camera, self.width, self.height,
                shadow_enabled=shadows,
                legacy_lights=legacy,
            )
            return self.frame
        # Viewport path: render at viewport size (projection follows
        # the viewport, as in three.js), then composite the clip rect.
        vx, vy, vw, vh = (int(round(v)) for v in self._viewport)
        if vw <= 0 or vh <= 0:
            return self._ensure_frame()
        temp = render_module.render_scene(
            scene, camera, vw, vh, shadow_enabled=shadows,
            legacy_lights=legacy)
        frame = self._ensure_frame()
        x0, y0, w, h = self._clip_rect()
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(self.width, x0 + w), min(self.height, y0 + h)
        if xb > xa and yb > ya:
            frame[ya:yb, xa:xb] = temp[ya - y0:yb - y0, xa - x0:xb - x0]
        self.frame = frame
        return self.frame

    @staticmethod
    def _load_render_module():
        try:
            import render as render_module

            return render_module
        except ImportError as exc:
            raise ImportError(
                "three.py: sibling module 'render.py' is required for "
                "WebGLRenderer.render() but could not be imported"
            ) from exc


# ---------------------------------------------------------------------------
# Points / Line
# ---------------------------------------------------------------------------

class PointsMaterial(Material):
    isPointsMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        # ``size`` is in pixels when sizeAttenuation is False; with
        # attenuation the renderer scales it by the perspective factor
        # (three.js PointsMaterial semantics).
        self.size = float(p.get("size", 1.0))
        self.sizeAttenuation = bool(p.get("sizeAttenuation", True))


class LineBasicMaterial(Material):
    isLineBasicMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        # Stored for API compatibility; the software renderer always
        # draws 1-pixel lines (as does WebGL on most platforms).
        self.linewidth = float(p.get("linewidth", 1.0))


class Points(Object3D):
    isPoints = True

    def __init__(self, geometry=None, material=None, name=""):
        super().__init__(name)
        self.geometry = geometry if geometry is not None else BufferGeometry()
        self.material = material if material is not None else PointsMaterial()


class Line(Object3D):
    isLine = True

    def __init__(self, geometry=None, material=None, name=""):
        super().__init__(name)
        self.geometry = geometry if geometry is not None else BufferGeometry()
        self.material = material if material is not None else LineBasicMaterial()


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------

def checker_texture_image(pixels=64, cells=8, dark=(0, 0, 0),
                          light=(255, 255, 255)):
    """Procedural checkerboard as a numpy RGB array (pixels x pixels, with
    ``cells`` x ``cells`` squares).  No external assets (CC0 rule)."""
    pixels = max(2, int(pixels))
    cells = max(1, int(cells))
    yy, xx = np.mgrid[0:pixels, 0:pixels]
    cell = np.maximum(1, pixels // cells)
    board = ((xx // cell) + (yy // cell)) % 2
    img = np.empty((pixels, pixels, 3), dtype=np.uint8)
    img[:, :] = np.array(light, dtype=np.uint8)
    img[board == 1] = np.array(dark, dtype=np.uint8)
    return img


def _coerce_texture_image(image):
    """Normalise a texture image to an (h, w, 3) numpy array (or None)."""
    if image is None:
        return None
    arr = getattr(image, "image", None)  # a Texture was passed
    if arr is not None:
        image = arr
    try:
        arr = np.asarray(image)
    except Exception:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


# Managers whose onLoad notification is due, in order.  See
# LoadingManager.itemEnd for why the notification is deferred.
_MANAGERS_DUE_ONLOAD = []


def drain_loading_managers():
    """Fire every deferred LoadingManager.onLoad, including any
    queued by loads started inside an onLoad handler itself."""
    while _MANAGERS_DUE_ONLOAD:
        manager = _MANAGERS_DUE_ONLOAD.pop(0)
        manager._onload_due = False
        if manager.onLoad is not None:
            manager.onLoad()


class LoadingManager:
    """three.js ``LoadingManager``: loaders report item start/end and
    the manager fires ``onLoad`` once every started item has ended.

    The port loads synchronously (the bytes are already on device),
    but the notification is DEFERRED to the host's task boundary
    (drain_loading_managers, called by the interpreter after each
    script and before each animation frame): in a browser the items
    of a load loop all start before any completes, so onLoad fires
    once, after the last one; fired synchronously it would go off
    after the FIRST item of the loop, before page code has seen the
    other items' own callbacks.  ``onProgress`` still fires per item,
    synchronously, receiving ``(url, itemsLoaded, itemsTotal)`` and
    ``onError`` the url, as in three.js."""

    isLoadingManager = True

    def __init__(self, onLoad=None, onProgress=None, onError=None):
        self.onLoad = onLoad
        self.onProgress = onProgress
        self.onError = onError
        self._loading = 0
        self._loaded = 0
        self._total = 0
        self._onload_due = False

    def itemStart(self, url):
        self._total += 1
        self._loading += 1

    def itemEnd(self, url):
        self._loaded += 1
        if self._loading > 0:
            self._loading -= 1
        if self.onProgress is not None:
            self.onProgress(url, float(self._loaded), float(self._total))
        if self._loading == 0 and self.onLoad is not None \
                and not self._onload_due:
            self._onload_due = True
            _MANAGERS_DUE_ONLOAD.append(self)

    def itemError(self, url):
        if self.onError is not None:
            self.onError(url)


#: three.js ``THREE.DefaultLoadingManager``: the shared manager loaders
#: report to when constructed without one.  Its handlers stay None in
#: the port, so reporting to it is inert.
DefaultLoadingManager = LoadingManager()


def _manager_of(loader):
    """The LoadingManager a loader reports to (never None)."""
    manager = getattr(loader, "manager", None)
    return manager if manager is not None else DefaultLoadingManager


class Texture:
    """A texture holding a numpy RGB image (h, w, 3).

    Sampling (renderer side) is nearest-neighbour with repeat wrapping and
    three.js-style flipY (uv (0, 0) is the image's bottom-left)."""

    isTexture = True

    def __init__(self, image=None, name=""):
        self.name = str(name)
        arr = _coerce_texture_image(image)
        # A texture with no image falls back to the procedural checker so
        # rendering never breaks.
        self.image = arr if arr is not None else checker_texture_image()
        self.wrapS = "repeat"
        self.wrapT = "repeat"
        # three.js uv transform (matrixAutoUpdate for textures uses
        # these directly): the renderer applies the full transform --
        # offset, repeat and rotation about ``center``.
        self.offset = Vector2(0.0, 0.0)
        self.repeat = Vector2(1.0, 1.0)
        self.center = Vector2(0.0, 0.0)
        self.rotation = 0.0
        # Legacy numeric encoding (r132-generation API the manual
        # lessons are written against): 1000 = LinearEncoding.  The
        # palette renderer does not colour-manage, so the value is
        # recorded, not applied.
        self.encoding = 1000
        self.needsUpdate = True

    def setImage(self, image):
        arr = _coerce_texture_image(image)
        if arr is not None:
            self.image = arr
            self.needsUpdate = True
        return self

    def dispose(self):
        """three.js dispose(): releases GPU resources; none are held
        per texture in the software renderer, so this is a record-only
        no-op (memory-test samples call it in a loop)."""
        self._disposed = True

    def clone(self):
        tex = Texture(self.image, self.name)
        tex.offset = self.offset.clone()
        tex.repeat = self.repeat.clone()
        tex.center = self.center.clone()
        tex.rotation = self.rotation
        tex.encoding = self.encoding
        tex.wrapS = self.wrapS
        tex.wrapT = self.wrapT
        return tex

    def __repr__(self):
        h, w = self.image.shape[:2] if self.image is not None else (0, 0)
        return "Texture(%r, %dx%d)" % (self.name, w, h)


class TextureLoader:
    """Resolves texture names against a host-provided registry.

    The host (main.py) registers decoded dropped images (and the built-in
    'checker') by file name; ``load(name)`` returns a Texture for the
    registered image, or the procedural 8x8 checker when the name cannot
    be resolved (it never raises)."""

    isTextureLoader = True

    _registry = {}

    def __init__(self, manager=None):
        self.manager = manager

    @classmethod
    def register(cls, name, image):
        arr = _coerce_texture_image(image)
        if arr is not None:
            cls._registry[str(name)] = arr
        return arr

    @classmethod
    def unregister(cls, name):
        cls._registry.pop(str(name), None)

    @classmethod
    def registered_names(cls):
        return sorted(cls._registry.keys())

    def load(self, name, onLoad=None, onProgress=None, onError=None):
        """three.js ``TextureLoader.load(url, onLoad, onProgress,
        onError)``: the image is already on device, so the texture is
        built immediately, ``onLoad`` runs before ``load`` returns
        (the GLTFLoader facade's synchronous convention), and the
        texture is also returned.  An unresolvable name yields the
        procedural checker rather than an error, as before."""
        key = str(name)
        manager = _manager_of(self)
        manager.itemStart(key)
        try:
            arr = self._registry.get(key)
            if arr is None and ("/" in key or "\\" in key):
                arr = self._registry.get(
                    key.replace("\\", "/").split("/")[-1])
            if arr is None:
                texture = Texture(checker_texture_image(), name=key)
            else:
                texture = Texture(arr, name=key)
        except Exception as exc:
            manager.itemError(key)
            manager.itemEnd(key)
            if onError is not None:
                onError(exc)
                return None
            raise
        if onProgress is not None:
            onProgress({"loaded": 1.0, "total": 1.0,
                        "lengthComputable": False})
        if onLoad is not None:
            onLoad(texture)
        manager.itemEnd(key)
        return texture

    def loadAsync(self, name, onProgress=None):
        """Synchronous bridge for three.js ``Loader.loadAsync``
       : the bytes are already on device, so this runs
        ``load`` immediately and returns an already-resolved
        SyncPromise.  Code that depends on the real asynchronous
        ordering (work happening between the call and the resolution)
        is out of scope."""
        try:
            return SyncPromise.resolved(self.load(name))
        except Exception as exc:
            return SyncPromise.rejected(exc)

    # three.js also exposes setPath(); paths are meaningless here (the
    # registry is keyed by plain names), so it is accepted and ignored.
    def setPath(self, path):
        return self


def register_texture(name, image):
    """Module-level shortcut for TextureLoader.register()."""
    return TextureLoader.register(name, image)


# ---------------------------------------------------------------------------
# GLTFLoader -- facade; the parser lives in gltf.py
# ---------------------------------------------------------------------------

class GLTFLoader:
    """Minimal glTF loader facade (three.js ``GLTFLoader`` shape).

    The host (main.py) registers the raw bytes of dropped ``.glb`` /
    ``.gltf`` files by file name; ``load(name, onLoad)`` parses them with
    the sibling module ``gltf.py`` and calls ``onLoad`` with a result
    object whose ``.scene`` is a Group of static meshes.  Loading is
    synchronous (the bytes are already on device), so ``onLoad`` runs
    before ``load`` returns; the parsed result is also returned.

    Out of scope (reported on the result's ``warnings`` list, never a
    crash): skinning, animation, external buffers, Draco etc.
    """

    isGLTFLoader = True

    _registry = {}

    def __init__(self, manager=None):
        self.manager = manager

    @classmethod
    def register(cls, name, data):
        if isinstance(data, (bytes, bytearray, memoryview)):
            cls._registry[str(name)] = bytes(data)
            return True
        return False

    @classmethod
    def unregister(cls, name):
        cls._registry.pop(str(name), None)

    @classmethod
    def registered_names(cls):
        return sorted(cls._registry.keys())

    def load(self, name, onLoad=None, onProgress=None, onError=None):
        gltf_module = self._load_gltf_module()
        manager = _manager_of(self)
        manager.itemStart(str(name))
        try:
            result = gltf_module.load_named(name)
        except Exception as exc:
            manager.itemError(str(name))
            manager.itemEnd(str(name))
            if onError is not None:
                onError(exc)
                return None
            raise
        if onLoad is not None:
            onLoad(result)
        manager.itemEnd(str(name))
        return result

    def loadAsync(self, name, onProgress=None):
        """Synchronous bridge for three.js ``Loader.loadAsync``
       : runs the synchronous ``load`` now and returns an
        already-settled SyncPromise (rejected when the load fails).
        Code that depends on the real asynchronous ordering is out
        of scope."""
        try:
            return SyncPromise.resolved(self.load(name))
        except Exception as exc:
            return SyncPromise.rejected(exc)

    def parse(self, data, onLoad=None, onError=None):
        """Parse raw bytes directly (three.js GLTFLoader.parse shape)."""
        gltf_module = self._load_gltf_module()
        try:
            result = gltf_module.load_bytes(data)
        except Exception as exc:
            if onError is not None:
                onError(exc)
                return None
            raise
        if onLoad is not None:
            onLoad(result)
        return result

    # three.js also exposes setPath(); paths are meaningless here (the
    # registry is keyed by plain names), so it is accepted and ignored.
    def setPath(self, path):
        return self

    @staticmethod
    def _load_gltf_module():
        try:
            import gltf as gltf_module

            return gltf_module
        except ImportError as exc:
            raise ImportError(
                "three.py: sibling module 'gltf.py' is required for "
                "GLTFLoader but could not be imported"
            ) from exc


# ---------------------------------------------------------------------------
# Input and time: Vector2, Clock, host pointer state,
# Ray / Raycaster, OrbitControls
# ---------------------------------------------------------------------------

class Vector2:
    """Minimal three.js ``Vector2`` (used e.g. for Raycaster NDC input)."""

    isVector2 = True

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def set(self, x, y):
        self.x = float(x)
        self.y = float(y)
        return self

    def setScalar(self, s):
        self.x = float(s)
        self.y = float(s)
        return self

    def copy(self, v):
        self.x = float(v.x)
        self.y = float(v.y)
        return self

    def clone(self):
        return Vector2(self.x, self.y)

    def fromArray(self, arr, offset=0):
        offset = int(offset)
        self.x = float(arr[offset])
        self.y = float(arr[offset + 1])
        return self

    def toArray(self, array=None, offset=0):
        if array is None:
            return [self.x, self.y]
        offset = int(offset)
        if isinstance(array, list):
            while len(array) <= offset + 1:
                array.append(0.0)
        array[offset] = self.x
        array[offset + 1] = self.y
        return array

    def multiplyScalar(self, s):
        s = float(s)
        self.x *= s
        self.y *= s
        return self

    def divideScalar(self, s):
        s = float(s)
        return self.multiplyScalar(1.0 / s) if s != 0.0 else self.setScalar(0.0)

    def add(self, v):
        self.x += float(v.x)
        self.y += float(v.y)
        return self

    def sub(self, v):
        self.x -= float(v.x)
        self.y -= float(v.y)
        return self

    def addScaledVector(self, v, s):
        s = float(s)
        self.x += float(v.x) * s
        self.y += float(v.y) * s
        return self

    def setX(self, x):
        self.x = float(x)
        return self

    def setY(self, y):
        self.y = float(y)
        return self

    def getComponent(self, index):
        return (self.x, self.y)[int(index)]

    def setComponent(self, index, value):
        if int(index) == 0:
            self.x = float(value)
        else:
            self.y = float(value)
        return self

    def multiplyScalar(self, s):
        self.x *= float(s)
        self.y *= float(s)
        return self

    def divideScalar(self, s):
        return self.multiplyScalar(1.0 / float(s))

    def lengthSq(self):
        return self.x * self.x + self.y * self.y

    def length(self):
        return math.sqrt(self.lengthSq())

    def normalize(self):
        n = self.length()
        return self.divideScalar(n if n != 0.0 else 1.0)

    def distanceTo(self, v):
        dx = self.x - float(v.x)
        dy = self.y - float(v.y)
        return math.sqrt(dx * dx + dy * dy)

    def lerp(self, v, t):
        self.x += (float(v.x) - self.x) * float(t)
        self.y += (float(v.y) - self.y) * float(t)
        return self

    def lerpVectors(self, a, b, t):
        t = float(t)
        self.x = float(a.x) + (float(b.x) - float(a.x)) * t
        self.y = float(a.y) + (float(b.y) - float(a.y)) * t
        return self

    def __eq__(self, other):
        return (
            isinstance(other, Vector2)
            and self.x == other.x
            and self.y == other.y
        )

    def equals(self, other):
        return (
            isinstance(other, Vector2)
            and self.x == other.x
            and self.y == other.y
        )

    def __repr__(self):
        return "Vector2(%.4g, %.4g)" % (self.x, self.y)


class Clock:
    """three.js ``Clock``: elapsed/delta time based on ``time.time()``.

    Follows the original's contract: with ``autoStart`` (default) the
    clock starts lazily on the first ``getDelta``/``getElapsedTime``
    call, which then reports 0; ``getElapsedTime`` internally advances
    the delta timer exactly like three.js.
    """

    isClock = True

    def __init__(self, autoStart=True):
        self.autoStart = bool(autoStart)
        self.startTime = 0.0
        self.oldTime = 0.0
        self.elapsedTime = 0.0
        self.running = False

    def start(self):
        self.startTime = time.time()
        self.oldTime = self.startTime
        self.elapsedTime = 0.0
        self.running = True
        return self

    def stop(self):
        self.getElapsedTime()
        self.running = False
        self.autoStart = False
        return self

    def getDelta(self):
        diff = 0.0
        if self.autoStart and not self.running:
            self.start()
            return 0.0
        if self.running:
            new_time = time.time()
            diff = new_time - self.oldTime
            self.oldTime = new_time
            self.elapsedTime += diff
        return diff

    def getElapsedTime(self):
        self.getDelta()
        return self.elapsedTime


# -- host pointer state ------------------------------------------------------
# There is no DOM here: the host (main.py) samples Pyxel's mouse (touch
# arrives as mouse input on Pyxel) once per frame and stores the result
# in this module-level state.  OrbitControls polls it in update(), the
# same place three.js OrbitControls applies its DOM-event deltas.
# ``wheel`` is a per-frame delta (positive = wheel up = zoom in).

_pointer_state = {
    "x": 0.0,
    "y": 0.0,
    "left": False,
    "middle": False,
    "right": False,
    "wheel": 0.0,
}


def set_pointer_state(x, y, left=False, right=False, wheel=0.0,
                      middle=False):
    """Host entry point: record the current pointer state (screen px)."""
    _pointer_state["x"] = float(x)
    _pointer_state["y"] = float(y)
    _pointer_state["left"] = bool(left)
    _pointer_state["middle"] = bool(middle)
    _pointer_state["right"] = bool(right)
    _pointer_state["wheel"] = float(wheel)


def get_pointer_state():
    """A snapshot copy of the host-fed pointer state."""
    return dict(_pointer_state)


# -- raycasting ----------------------------------------------------------------

class Ray:
    """three.js ``Ray``: an origin and a (normalised) direction."""

    isRay = True

    def __init__(self, origin=None, direction=None):
        self.origin = origin if origin is not None else Vector3()
        self.direction = direction if direction is not None else Vector3(0, 0, -1)

    def set(self, origin, direction):
        self.origin.copy(origin)
        self.direction.copy(direction)
        return self

    def copy(self, ray):
        self.origin.copy(ray.origin)
        self.direction.copy(ray.direction)
        return self

    def clone(self):
        return Ray(self.origin.clone(), self.direction.clone())

    def at(self, t, target=None):
        if target is None:
            target = Vector3()
        return target.copy(self.direction).multiplyScalar(float(t)).add(self.origin)

    def lookAt(self, v):
        self.direction.copy(v).sub(self.origin).normalize()
        return self

    def __repr__(self):
        return "Ray(%r, %r)" % (self.origin, self.direction)


def _coords_xy(coords):
    """(x, y) floats out of a Vector2/Vector3, a dict, or any object
    with ``x``/``y`` attributes (JS pointer objects arrive as dicts)."""
    if isinstance(coords, dict):
        return float(coords.get("x", 0.0)), float(coords.get("y", 0.0))
    return float(getattr(coords, "x", 0.0)), float(getattr(coords, "y", 0.0))


class Raycaster:
    """three.js ``Raycaster`` (mesh triangles only).

    ``intersectObject`` / ``intersectObjects`` return a list of dicts
    shaped like three.js intersections -- ``{distance, point, object,
    face, faceIndex, uv}`` -- sorted by ascending distance.  Face
    culling honours the material's ``side`` (FrontSide / BackSide /
    DoubleSide), as in three.js Mesh.raycast.
    """

    isRaycaster = True

    def __init__(self, origin=None, direction=None, near=0.0, far=math.inf):
        self.ray = Ray(
            origin.clone() if origin is not None else Vector3(),
            direction.clone() if direction is not None else Vector3(0, 0, -1),
        )
        if direction is not None:
            self.ray.direction.normalize()
        self.near = float(near)
        self.far = float(far)
        self.params = {
            "Mesh": {},
            "Line": {"threshold": 1.0},
            "Points": {"threshold": 1.0},
            "LOD": {},
        }

    def set(self, origin, direction):
        self.ray.origin.copy(origin)
        self.ray.direction.copy(direction).normalize()
        return self

    def setFromCamera(self, coords, camera):
        x, y = _coords_xy(coords)
        camera.updateWorldMatrix(True, False)
        if getattr(camera, "isOrthographicCamera", False):
            # three.js orthographic branch: the ray starts on the near
            # plane through (x, y) and runs parallel to the camera's -Z
            # (it does NOT diverge from the camera position).
            origin = Vector3(
                x, y,
                (camera.near + camera.far) / (camera.near - camera.far),
            )
            origin.unproject(camera)
            self.ray.origin.copy(origin)
            direction = Vector3(0, 0, -1).transformDirection(
                camera.matrixWorld)
            self.ray.direction.copy(direction)
            return self
        origin = camera.getWorldPosition(Vector3())
        self.ray.origin.copy(origin)
        vec = Vector3(x, y, 0.5)
        vec.unproject(camera)
        self.ray.direction.copy(vec.sub(origin)).normalize()
        return self

    def intersectObject(self, object, recursive=True):
        intersects = []
        self._intersect(object, intersects, bool(recursive))
        intersects.sort(key=lambda hit: hit["distance"])
        return intersects

    def intersectObjects(self, objects, recursive=True):
        intersects = []
        if objects is None:
            return intersects
        if isinstance(objects, Object3D):
            objects = [objects]
        for obj in objects:
            self._intersect(obj, intersects, bool(recursive))
        intersects.sort(key=lambda hit: hit["distance"])
        return intersects

    # -- internals ---------------------------------------------------------
    def _intersect(self, obj, out, recursive):
        if getattr(obj, "isMesh", False):
            self._raycast_mesh(obj, out)
        if recursive:
            for child in list(getattr(obj, "children", [])):
                self._intersect(child, out, recursive)

    def _raycast_mesh(self, mesh, out):
        geometry = getattr(mesh, "geometry", None)
        material = getattr(mesh, "material", None)
        if geometry is None or material is None:
            return
        if isinstance(material, (list, tuple)):
            return  # material groups are out of scope
        attributes = getattr(geometry, "attributes", None)
        if not attributes:
            return
        pos = attributes.get("position")
        if pos is None:
            return
        mesh.updateWorldMatrix(True, False)
        mw = mesh.matrixWorld._a
        arr = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] == 0:
            return
        hom = np.concatenate(
            [arr, np.ones((arr.shape[0], 1), dtype=np.float64)], axis=1
        )
        world = (hom @ mw.T)[:, :3]
        index = geometry.index
        if index is not None:
            tris = np.asarray(index, dtype=np.int64).reshape(-1, 3)
        else:
            tris = np.arange(arr.shape[0], dtype=np.int64).reshape(-1, 3)
        if tris.shape[0] == 0:
            return
        a = world[tris[:, 0]]
        b = world[tris[:, 1]]
        c = world[tris[:, 2]]
        e1 = b - a
        e2 = c - a
        origin = np.array(
            [self.ray.origin.x, self.ray.origin.y, self.ray.origin.z],
            dtype=np.float64,
        )
        direction = np.array(
            [self.ray.direction.x, self.ray.direction.y, self.ray.direction.z],
            dtype=np.float64,
        )
        # Möller–Trumbore, vectorised over the mesh's triangles.
        pvec = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        side = getattr(material, "side", DoubleSide)
        eps = 1e-12
        if side == FrontSide:
            facing = det > eps
        elif side == BackSide:
            facing = det < -eps
        else:
            facing = np.abs(det) > eps
        if not np.any(facing):
            return
        safe_det = np.where(facing, det, 1.0)
        tvec = origin - a
        u = np.einsum("ij,ij->i", tvec, pvec) / safe_det
        qvec = np.cross(tvec, e1)
        v = np.einsum("j,ij->i", direction, qvec) / safe_det
        t = np.einsum("ij,ij->i", e2, qvec) / safe_det
        hit_mask = (
            facing
            & (u >= -1e-9)
            & (v >= -1e-9)
            & (u + v <= 1.0 + 1e-9)
            & (t >= self.near)
            & (t <= self.far)
        )
        uv_attr = attributes.get("uv")
        uv_arr = None
        if uv_attr is not None:
            uv_arr = np.asarray(uv_attr, dtype=np.float64).reshape(-1, 2)
        normals = np.cross(e1, e2)
        for i in np.nonzero(hit_mask)[0]:
            dist = float(t[i])
            point = self.ray.at(dist, Vector3())
            normal = normals[i]
            nlen = float(np.linalg.norm(normal))
            if nlen > 0.0:
                normal = normal / nlen
            if side == BackSide:
                normal = -normal
            face_normal = Vector3(
                float(normal[0]), float(normal[1]), float(normal[2])
            )
            hit_uv = None
            if uv_arr is not None:
                w0 = 1.0 - float(u[i]) - float(v[i])
                tuv = (
                    w0 * uv_arr[tris[i, 0]]
                    + float(u[i]) * uv_arr[tris[i, 1]]
                    + float(v[i]) * uv_arr[tris[i, 2]]
                )
                hit_uv = Vector2(float(tuv[0]), float(tuv[1]))
            out.append({
                "distance": dist,
                "point": point,
                "object": mesh,
                "face": {
                    "a": int(tris[i, 0]),
                    "b": int(tris[i, 1]),
                    "c": int(tris[i, 2]),
                    "normal": face_normal,
                    "materialIndex": 0,
                },
                "faceIndex": int(i),
                "uv": hit_uv,
            })


class OrbitControls:
    """three.js ``OrbitControls`` (examples/jsm/controls), re-implemented
    for this port's input model.

    In three.js the controls subscribe to DOM pointer/wheel events; here
    there is no DOM, so ``update()`` polls the host-fed pointer state
    (see ``set_pointer_state``: Pyxel mouse, with touch delivered as
    mouse input) and applies drag/wheel deltas exactly where the
    original applies its event deltas:

      * left-button drag  -> orbit (rotateLeft / rotateUp signs follow
        three.js: dragging right turns the camera so the scene appears
        to follow the cursor),
      * right-button drag -> pan (screen-space by default),
      * wheel             -> dolly (wheel up zooms in).

    The public surface mirrors the original: ``target``, the
    enable*/speed/damping/distance/polar properties, ``update()``,
    ``reset()``, ``dispose()`` and no-op event-listener methods.
    """

    isOrbitControls = True

    def __init__(self, camera, domElement=None):
        self.object = camera
        self.domElement = domElement
        self.target = Vector3(0.0, 0.0, 0.0)

        self.enableDamping = False
        self.dampingFactor = 0.05
        self.enableZoom = True
        self.zoomSpeed = 1.0
        self.enableRotate = True
        self.rotateSpeed = 1.0
        self.enablePan = True
        self.panSpeed = 1.0
        self.screenSpacePanning = True
        self.minDistance = 0.0
        self.maxDistance = math.inf
        self.minPolarAngle = 0.0
        self.maxPolarAngle = math.pi
        self.minAzimuthAngle = -math.inf
        self.maxAzimuthAngle = math.inf
        self.autoRotate = False
        self.autoRotateSpeed = 2.0

        # state captured for reset()
        self.target0 = self.target.clone()
        self.position0 = camera.position.clone()
        self.zoom0 = getattr(camera, "zoom", 1.0)

        # per-frame interaction state (deltas are consumed by update())
        self._dtheta = 0.0
        self._dphi = 0.0
        self._scale = 1.0
        self._pan_offset = Vector3()
        self._last_x = 0.0
        self._last_y = 0.0
        self._prev_left = False
        self._prev_right = False
        self._listeners = {}

    # -- DOM-shaped no-ops ---------------------------------------------------
    def addEventListener(self, type, listener, *args):
        self._listeners.setdefault(str(type), []).append(listener)

    def removeEventListener(self, type, listener, *args):
        callbacks = self._listeners.get(str(type), [])
        if listener in callbacks:
            callbacks.remove(listener)

    def dispatchEvent(self, event):
        return True

    def dispose(self):
        self._listeners.clear()

    # -- queries ---------------------------------------------------------------
    def getDistance(self):
        pos = self.object.getWorldPosition(Vector3())
        return pos.distanceTo(self.target)

    def getPolarAngle(self):
        return self._spherical()[1]

    def getAzimuthalAngle(self):
        return self._spherical()[0]

    def reset(self):
        self.target.copy(self.target0)
        self.object.position.copy(self.position0)
        if hasattr(self.object, "zoom"):
            self.object.zoom = self.zoom0
            if hasattr(self.object, "updateProjectionMatrix"):
                self.object.updateProjectionMatrix()
        self._dtheta = 0.0
        self._dphi = 0.0
        self._scale = 1.0
        self._pan_offset.set(0.0, 0.0, 0.0)
        self.object.lookAt(self.target)
        return self

    # -- internals ---------------------------------------------------------------
    def _element_height(self):
        el = self.domElement
        height = getattr(el, "height", None) if el is not None else None
        if not height:
            return 192.0
        return float(height)

    def _spherical(self):
        """(theta, phi, radius) of the camera offset around the target."""
        pos = self.object.getWorldPosition(Vector3())
        offset = pos.sub(self.target)
        radius = offset.length()
        if radius == 0.0:
            return 0.0, 0.0, 0.0
        theta = math.atan2(offset.x, offset.z)
        phi = math.acos(_clamp(offset.y / radius, -1.0, 1.0))
        return theta, phi, radius

    def _poll_input(self):
        state = get_pointer_state()
        x = state["x"]
        y = state["y"]
        left = state["left"]
        right = state["right"]
        height = self._element_height()

        if left and self.enableRotate:
            if self._prev_left:
                dx = x - self._last_x
                dy = y - self._last_y
                # three.js rotateLeft/rotateUp: the deltas subtract from
                # theta/phi, scaled by 2*pi per element height.
                self._dtheta -= (
                    2.0 * math.pi * dx / height * float(self.rotateSpeed)
                )
                self._dphi -= (
                    2.0 * math.pi * dy / height * float(self.rotateSpeed)
                )
        if right and self.enablePan:
            if self._prev_right:
                dx = x - self._last_x
                dy = y - self._last_y
                self._accumulate_pan(dx, dy)
        self._prev_left = bool(left)
        self._prev_right = bool(right)
        self._last_x = x
        self._last_y = y

        wheel = state["wheel"]
        if wheel and self.enableZoom:
            # wheel up (positive) zooms in, one notch scales the radius
            # by pow(0.95, zoomSpeed), as in three.js getZoomScale().
            self._scale *= math.pow(0.95, float(self.zoomSpeed)) ** wheel

        if self.autoRotate and not left and not right and self.enableRotate:
            # getAutoRotationAngle(): 2*pi / 60 / 60 * autoRotateSpeed
            # per update() call.
            self._dtheta -= (
                2.0 * math.pi / 3600.0 * float(self.autoRotateSpeed)
            )

    def _accumulate_pan(self, dx, dy):
        if dx == 0.0 and dy == 0.0:
            return
        camera = self.object
        camera.updateWorldMatrix(True, False)
        m = camera.matrixWorld._a
        right = Vector3(float(m[0, 0]), float(m[1, 0]), float(m[2, 0]))
        radius = self.getDistance()
        fov = float(getattr(camera, "fov", 50.0))
        target_distance = radius * math.tan(math.radians(fov) * 0.5)
        height = self._element_height()
        speed = float(self.panSpeed)
        # panLeft: dragging right pulls the world with the cursor, so
        # the target moves along the camera's -right axis.
        self._pan_offset.add(
            right.multiplyScalar(-2.0 * dx * target_distance / height * speed)
        )
        if self.screenSpacePanning:
            up = Vector3(float(m[0, 1]), float(m[1, 1]), float(m[2, 1]))
            self._pan_offset.add(
                up.multiplyScalar(2.0 * dy * target_distance / height * speed)
            )
        else:
            # Pan within the world XZ plane: use the camera's forward
            # direction projected onto the ground.
            up_world = Vector3(0.0, 1.0, 0.0)
            forward = Vector3()
            forward.crossVectors(right, up_world)
            if forward.lengthSq() > 0.0:
                forward.normalize()
                self._pan_offset.add(
                    forward.multiplyScalar(
                        -2.0 * dy * target_distance / height * speed
                    )
                )

    # -- the three.js contract ---------------------------------------------------
    def update(self, *args):
        self._poll_input()

        theta, phi, radius = self._spherical()
        theta += self._dtheta
        phi += self._dphi
        theta = _clamp(theta, self.minAzimuthAngle, self.maxAzimuthAngle)
        phi = _clamp(phi, self.minPolarAngle, self.maxPolarAngle)
        # three.js keeps phi a hair away from the poles.
        phi = _clamp(phi, 1e-6, math.pi - 1e-6)
        radius = _clamp(
            radius * self._scale, self.minDistance, self.maxDistance
        )

        if self._pan_offset.lengthSq() > 0.0:
            self.target.add(self._pan_offset)

        sin_phi = math.sin(phi)
        offset = Vector3(
            radius * sin_phi * math.sin(theta),
            radius * math.cos(phi),
            radius * sin_phi * math.cos(theta),
        )
        self.object.position.copy(self.target).add(offset)
        self.object.lookAt(self.target)

        if self.enableDamping:
            keep = 1.0 - float(self.dampingFactor)
            self._dtheta *= keep
            self._dphi *= keep
            self._pan_offset.multiplyScalar(keep)
        else:
            self._dtheta = 0.0
            self._dphi = 0.0
            self._pan_offset.set(0.0, 0.0, 0.0)
        self._scale = 1.0
        return True


# ---------------------------------------------------------------------------
# Animation: KeyframeTrack / AnimationClip / PropertyBinding /
# AnimationMixer / AnimationAction.
#
# Programmatic clips are the target (tracks assembled in JS and played via
# a mixer); GLTF animation playback is out of scope.  Interpolation is
# linear (quaternions use slerp); InterpolateSmooth falls back to linear.
# Cross-fades (fadeIn/fadeOut/crossFadeTo), additive blending and mixer
# events ('finished' / 'loop') are not implemented.
# ---------------------------------------------------------------------------

LoopOnce = 1000
LoopRepeat = 1001
LoopPingPong = 1002

InterpolateDiscrete = 2300
InterpolateLinear = 2301
InterpolateSmooth = 2302

NormalAnimationBlendMode = 2500
AdditiveAnimationBlendMode = 2501

__all__ += [
    "LoopOnce",
    "LoopRepeat",
    "LoopPingPong",
    "InterpolateDiscrete",
    "InterpolateLinear",
    "InterpolateSmooth",
    "NormalAnimationBlendMode",
    "AdditiveAnimationBlendMode",
    "KeyframeTrack",
    "NumberKeyframeTrack",
    "VectorKeyframeTrack",
    "QuaternionKeyframeTrack",
    "ColorKeyframeTrack",
    "BooleanKeyframeTrack",
    "StringKeyframeTrack",
    "AnimationClip",
    "PropertyBinding",
    "AnimationAction",
    "AnimationMixer",
]


def _slerp_flat(dst, dst_offset, a, a_offset, b, b_offset, t):
    """Quaternion slerp on flat component arrays (three.js
    ``Quaternion.slerpFlat``).  Writes 4 components into ``dst``."""
    x0, y0, z0, w0 = a[a_offset], a[a_offset + 1], a[a_offset + 2], a[a_offset + 3]
    x1, y1, z1, w1 = b[b_offset], b[b_offset + 1], b[b_offset + 2], b[b_offset + 3]

    if x0 == x1 and y0 == y1 and z0 == z1 and w0 == w1:
        dst[dst_offset:dst_offset + 4] = [x0, y0, z0, w0]
        return dst

    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if dot < 0.0:
        x1, y1, z1, w1 = -x1, -y1, -z1, -w1
        dot = -dot

    if dot > 0.9995:
        # Very close: nlerp.
        s0 = 1.0 - t
        s1 = t
    else:
        sin_sq = 1.0 - dot * dot
        if sin_sq < 1e-12:
            s0 = 1.0 - t
            s1 = t
        else:
            sin = math.sqrt(sin_sq)
            theta = math.atan2(sin, dot)
            s0 = math.sin((1.0 - t) * theta) / sin
            s1 = math.sin(t * theta) / sin

    dst[dst_offset] = s0 * x0 + s1 * x1
    dst[dst_offset + 1] = s0 * y0 + s1 * y1
    dst[dst_offset + 2] = s0 * z0 + s1 * z1
    dst[dst_offset + 3] = s0 * w0 + s1 * w1
    return dst


def _quaternion_slerp(self, qb, t):
    out = [0.0, 0.0, 0.0, 0.0]
    _slerp_flat(out, 0, [self.x, self.y, self.z, self.w], 0,
                [qb.x, qb.y, qb.z, qb.w], 0, float(t))
    return self.set(out[0], out[1], out[2], out[3])


# three.js exposes slerp on Quaternion itself; attach it here so the
# animation section stays self-contained at the end of the module.
Quaternion.slerp = _quaternion_slerp
Quaternion.slerpFlat = staticmethod(_slerp_flat)


def _flat_float_list(data):
    if data is None:
        return []
    if isinstance(data, np.ndarray):
        return [float(v) for v in data.reshape(-1)]
    return [float(v) for v in data]


class KeyframeTrack:
    """Base keyframe track: ``name`` is a property binding path such as
    ``'.position'`` or ``'boneName.material.opacity'``; ``times`` are in
    seconds (ascending) and ``values`` is the flat concatenation of the
    per-key values."""

    ValueTypeName = None
    DefaultInterpolation = InterpolateLinear

    def __init__(self, name, times, values, interpolation=None):
        if not name:
            raise ValueError("KeyframeTrack: name is required")
        self.name = str(name)
        self.times = _flat_float_list(times)
        self.values = _flat_float_list(values)
        if interpolation is None:
            interpolation = self.DefaultInterpolation
        self._interpolation = interpolation

    # -- shape -----------------------------------------------------------
    def getValueSize(self):
        n = len(self.times)
        if n == 0:
            return len(self.values)
        return len(self.values) // n

    def getInterpolation(self):
        return self._interpolation

    def setInterpolation(self, interpolation):
        self._interpolation = interpolation
        return self

    def validate(self):
        size = self.getValueSize()
        if size <= 0:
            return False
        if len(self.times) == 0 or len(self.values) == 0:
            return False
        for i in range(1, len(self.times)):
            if self.times[i] < self.times[i - 1]:
                return False
        return True

    def optimize(self):
        # three.js trims redundant keys; values here are user-authored and
        # small, so this is a no-op kept for API compatibility.
        return self

    def clone(self):
        return type(self)(
            self.name, list(self.times), list(self.values), self._interpolation
        )

    # -- evaluation --------------------------------------------------------
    def _evaluate(self, t):
        """Interpolated flat value (list of floats) at time ``t``."""
        times = self.times
        values = self.values
        size = self.getValueSize()
        n = len(times)
        if n == 0 or size == 0:
            return [0.0] * size
        if n == 1 or t <= times[0]:
            return list(values[0:size])
        if t >= times[-1]:
            return list(values[(n - 1) * size:n * size])

        # Binary search for the interval [times[i], times[i + 1]).
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if times[mid] <= t:
                lo = mid
            else:
                hi = mid
        i = lo

        if self._interpolation == InterpolateDiscrete:
            return list(values[i * size:(i + 1) * size])

        span = times[i + 1] - times[i]
        alpha = 0.0 if span <= 0.0 else (t - times[i]) / span
        a = values[i * size:(i + 1) * size]
        b = values[(i + 1) * size:(i + 2) * size]

        if self.ValueTypeName == "quaternion":
            out = [0.0] * size
            _slerp_flat(out, 0, a, 0, b, 0, alpha)
            return out

        # Linear (InterpolateSmooth also lands here: linear fallback).
        return [a[k] + (b[k] - a[k]) * alpha for k in range(size)]

    def __repr__(self):
        return "%s(%r, %d keys)" % (
            type(self).__name__, self.name, len(self.times))


class NumberKeyframeTrack(KeyframeTrack):
    ValueTypeName = "number"


class VectorKeyframeTrack(KeyframeTrack):
    ValueTypeName = "vector"


class QuaternionKeyframeTrack(KeyframeTrack):
    ValueTypeName = "quaternion"


class ColorKeyframeTrack(KeyframeTrack):
    ValueTypeName = "color"


class BooleanKeyframeTrack(KeyframeTrack):
    ValueTypeName = "bool"
    DefaultInterpolation = InterpolateDiscrete


class StringKeyframeTrack(KeyframeTrack):
    ValueTypeName = "string"
    DefaultInterpolation = InterpolateDiscrete


class AnimationClip:
    _counter = 0

    def __init__(self, name="default", duration=-1, tracks=None):
        AnimationClip._counter += 1
        self.name = str(name)
        self.tracks = list(tracks) if tracks else []
        self.uuid = "clip-%d" % AnimationClip._counter
        self.duration = float(duration)
        if self.duration < 0:
            self.resetDuration()

    def resetDuration(self):
        duration = 0.0
        for track in self.tracks:
            if track.times:
                duration = max(duration, float(track.times[-1]))
        self.duration = duration
        return self.duration

    def optimize(self):
        for track in self.tracks:
            track.optimize()
        return self

    def validate(self):
        return all(track.validate() for track in self.tracks)

    def clone(self):
        return AnimationClip(
            self.name, self.duration, [t.clone() for t in self.tracks]
        )

    def __repr__(self):
        return "AnimationClip(%r, %.3gs, %d tracks)" % (
            self.name, self.duration, len(self.tracks))


# ---------------------------------------------------------------------------
# PropertyBinding: minimal dot-path resolution.
#
# Supported track names: '.position', '.quaternion', '.scale',
# '.material.opacity', component forms like '.position[x]', and named
# nodes like 'arm.position' or 'arm.material.opacity'.  A '/'-separated
# search path prefix is also accepted ('group/arm.position').
# ---------------------------------------------------------------------------

def _split_index_token(token):
    if token.endswith("]") and "[" in token:
        base, idx = token[:-1].split("[", 1)
        return base, idx
    return token, None


def _parse_track_name(track_name):
    result = {
        "nodeName": None,
        "objectName": None,
        "objectIndex": None,
        "propertyName": None,
        "propertyIndex": None,
    }
    name = str(track_name)

    if "/" in name:
        segments = name.split("/")
        last = segments[-1]
        if last.startswith(".") and len(segments) >= 2:
            result["nodeName"] = segments[-2]
        name = last

    if name.startswith("."):
        rest = name[1:]
    elif "." in name:
        node, rest = name.split(".", 1)
        result["nodeName"] = node or result["nodeName"]
    else:
        rest = name

    parts = [p for p in rest.split(".") if p != ""]
    if len(parts) == 1:
        result["propertyName"], result["propertyIndex"] = _split_index_token(
            parts[0])
    elif len(parts) >= 2:
        result["objectName"], result["objectIndex"] = _split_index_token(
            parts[0])
        result["propertyName"], result["propertyIndex"] = _split_index_token(
            parts[1])
    return result


def _find_node_by_name(root, name):
    found = []

    def visit(node):
        if found:
            return
        if getattr(node, "name", None) == name:
            found.append(node)
            return
        for child in getattr(node, "children", []):
            visit(child)

    visit(root)
    return found[0] if found else None


_COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2, "w": 3}


class PropertyBinding:
    """Resolves a track name against a root Object3D and reads/writes the
    bound property as a flat float array."""

    def __init__(self, rootNode, path, parsedPath=None):
        self.rootNode = rootNode
        self.path = path
        self.parsedPath = parsedPath if parsedPath is not None else (
            _parse_track_name(path))
        self.node = None
        self.object = None  # container that owns the property
        self._bound = False

    @staticmethod
    def parseTrackName(track_name):
        return _parse_track_name(track_name)

    @staticmethod
    def findNode(root, nodeName):
        return _find_node_by_name(root, nodeName)

    def bind(self):
        if self._bound:
            return
        self._bound = True
        parsed = self.parsedPath
        node_name = parsed["nodeName"]
        if node_name is None:
            node = self.rootNode
        else:
            node = _find_node_by_name(self.rootNode, node_name)
        self.node = node
        if node is None:
            self.object = None
            return

        target = node
        object_name = parsed["objectName"]
        if object_name is not None:
            target = getattr(node, object_name, None)
            object_index = parsed["objectIndex"]
            if object_index is not None and target is not None:
                try:
                    target = target[int(object_index)]
                except (TypeError, ValueError, IndexError, KeyError):
                    target = None
        self.object = target

    # -- value access ------------------------------------------------------
    def getValue(self, array, offset):
        self.bind()
        if self.object is None:
            return
        parsed = self.parsedPath
        prop = parsed["propertyName"]
        idx = parsed["propertyIndex"]
        if idx is not None:
            array[offset] = self._read_component(prop, idx)
            return
        current = getattr(self.object, prop, None)
        if isinstance(current, Vector3):
            array[offset] = current.x
            array[offset + 1] = current.y
            array[offset + 2] = current.z
        elif isinstance(current, Quaternion):
            array[offset] = current.x
            array[offset + 1] = current.y
            array[offset + 2] = current.z
            array[offset + 3] = current.w
        elif isinstance(current, Color):
            array[offset] = current.r
            array[offset + 1] = current.g
            array[offset + 2] = current.b
        elif isinstance(current, (int, float)):
            array[offset] = float(current)
        elif isinstance(current, bool):
            array[offset] = 1.0 if current else 0.0

    def setValue(self, array, offset):
        self.bind()
        if self.object is None:
            return
        parsed = self.parsedPath
        prop = parsed["propertyName"]
        idx = parsed["propertyIndex"]
        if idx is not None:
            self._write_component(prop, idx, float(array[offset]))
            return
        current = getattr(self.object, prop, None)
        if isinstance(current, Vector3):
            current.set(
                float(array[offset]),
                float(array[offset + 1]),
                float(array[offset + 2]),
            )
        elif isinstance(current, Quaternion):
            current.set(
                float(array[offset]),
                float(array[offset + 1]),
                float(array[offset + 2]),
                float(array[offset + 3]),
            )
        elif isinstance(current, Color):
            current.r = float(array[offset])
            current.g = float(array[offset + 1])
            current.b = float(array[offset + 2])
        elif isinstance(current, bool):
            setattr(self.object, prop, bool(array[offset]))
        elif isinstance(current, (int, float)):
            setattr(self.object, prop, float(array[offset]))
        elif current is None:
            # Unknown yet scalar-shaped slot (e.g. material.opacity before
            # the material exists is inert; a plain attribute is set).
            try:
                setattr(self.object, prop, float(array[offset]))
            except Exception:
                pass

    def _component_target(self, prop):
        current = getattr(self.object, prop, None)
        return current

    def _read_component(self, prop, idx):
        current = self._component_target(prop)
        if current is None:
            return 0.0
        if isinstance(current, (Vector3, Quaternion)):
            return float(getattr(current, idx))
        if isinstance(current, (list, tuple, np.ndarray)):
            return float(current[int(idx)])
        return 0.0

    def _write_component(self, prop, idx, value):
        current = self._component_target(prop)
        if current is None:
            return
        if isinstance(current, (Vector3, Quaternion)):
            setattr(current, idx, value)
        elif isinstance(current, list):
            current[int(idx)] = value


class _PropertyMixer:
    """Per-binding accumulation buffer (three.js PropertyMixer analogue):
    weighted running average of all active actions' values for one bound
    property, blended with the saved original state when the total weight
    is below 1."""

    def __init__(self, binding, typeName, valueSize):
        self.binding = binding
        self.typeName = typeName
        self.valueSize = valueSize
        self._buffer = [0.0] * valueSize
        self.cumulativeWeight = 0.0
        self.useCount = 0
        self.referenceCount = 0
        self._original = None

    def saveOriginalState(self):
        arr = [0.0] * self.valueSize
        self.binding.getValue(arr, 0)
        self._original = arr

    def restoreOriginalState(self):
        if self._original is not None:
            self.binding.setValue(self._original, 0)

    def accumulate(self, value, weight):
        if weight <= 0.0:
            return
        if self.cumulativeWeight == 0.0:
            self._buffer = list(value)
        else:
            t = weight / (self.cumulativeWeight + weight)
            if self.typeName == "quaternion":
                out = [0.0] * self.valueSize
                _slerp_flat(out, 0, self._buffer, 0, value, 0, t)
                self._buffer = out
            else:
                buf = self._buffer
                for k in range(self.valueSize):
                    buf[k] += (value[k] - buf[k]) * t
        self.cumulativeWeight += weight

    def apply(self):
        weight = self.cumulativeWeight
        self.cumulativeWeight = 0.0
        if weight <= 0.0:
            return
        if weight < 1.0 and self._original is not None:
            t = 1.0 - weight
            if self.typeName == "quaternion":
                out = [0.0] * self.valueSize
                _slerp_flat(out, 0, self._buffer, 0, self._original, 0, t)
                self._buffer = out
            else:
                buf = self._buffer
                orig = self._original
                for k in range(self.valueSize):
                    buf[k] += (orig[k] - buf[k]) * t
        self.binding.setValue(self._buffer, 0)


class AnimationAction:
    def __init__(self, mixer, clip, localRoot=None, blendMode=None):
        self._mixer = mixer
        self._clip = clip
        self._localRoot = localRoot if localRoot is not None else mixer._root
        self.blendMode = (
            NormalAnimationBlendMode if blendMode is None else blendMode)

        self.loop = LoopRepeat
        self.repetitions = math.inf
        self.time = 0.0
        self.timeScale = 1.0
        self.weight = 1.0
        self.enabled = True
        self.paused = False
        self.clampWhenFinished = False
        self.zeroSlopeAtStart = True
        self.zeroSlopeAtEnd = True

        self._loopCount = 0  # completed loops so far
        self._direction = 1  # ping-pong travel direction
        self._propertyMixers = []
        self._bound = False

    # -- transport ---------------------------------------------------------
    def play(self):
        self._mixer._activateAction(self)
        return self

    def stop(self):
        self._mixer._deactivateAction(self)
        return self

    def reset(self):
        self.paused = False
        self.enabled = True
        self.time = 0.0
        self._loopCount = 0
        self._direction = 1
        return self

    def isRunning(self):
        return self._mixer._isActiveAction(self)

    def isScheduled(self):
        return self.isRunning()

    # -- settings ------------------------------------------------------------
    def setLoop(self, mode, repetitions):
        self.loop = mode
        self.repetitions = repetitions
        return self

    def setEffectiveWeight(self, weight):
        self.weight = float(weight)
        self.enabled = self.weight > 0.0
        return self

    def getEffectiveWeight(self):
        return self.weight if self.enabled else 0.0

    def setEffectiveTimeScale(self, timeScale):
        self.timeScale = float(timeScale)
        return self

    def getEffectiveTimeScale(self):
        return self.timeScale

    def getMixer(self):
        return self._mixer

    def getClip(self):
        return self._clip

    def getRoot(self):
        return self._localRoot

    # -- internals -----------------------------------------------------------
    def _updateTime(self, delta):
        """Advance the action time; returns (clipTime, finished)."""
        duration = self._clip.duration if self._clip is not None else 0.0
        if duration <= 0.0:
            self.time = 0.0
            return 0.0, self.loop == LoopOnce

        if self.loop == LoopOnce:
            t = self.time + delta
            finished = False
            if t >= duration:
                t = duration
                finished = delta > 0.0
            elif t <= 0.0:
                t = 0.0
                finished = delta < 0.0
            self.time = t
            return t, finished

        ping_pong = self.loop == LoopPingPong
        t = self.time + delta * self._direction
        while t >= duration or t < 0.0:
            if t >= duration:
                overshoot = t - duration
                self._loopCount += 1
                if self._loopCount >= self.repetitions:
                    self.time = duration if self._direction > 0 else 0.0
                    return self.time, True
                if ping_pong:
                    self._direction = -self._direction
                    t = duration - overshoot
                else:
                    t = overshoot
            else:
                overshoot = -t
                self._loopCount += 1
                if self._loopCount >= self.repetitions:
                    self.time = 0.0 if self._direction < 0 else duration
                    return self.time, True
                if ping_pong:
                    self._direction = -self._direction
                    t = overshoot
                else:
                    t = duration - overshoot
        self.time = t
        return t, False

    def _update(self, deltaTime):
        """Accumulate this action's contribution for one mixer tick.
        Returns True when the action finished on this tick."""
        if not self.enabled:
            return False
        delta = 0.0 if self.paused else deltaTime * self.timeScale
        clip_time, finished = self._updateTime(delta)
        weight = self.weight
        if weight > 0.0 and self._clip is not None:
            tracks = self._clip.tracks
            for i, track in enumerate(tracks):
                value = track._evaluate(clip_time)
                self._propertyMixers[i].accumulate(value, weight)
        return finished

    def __repr__(self):
        cname = self._clip.name if self._clip is not None else None
        return "AnimationAction(%r, t=%.3g)" % (cname, self.time)


class AnimationMixer:
    def __init__(self, root):
        self._root = root
        self.time = 0.0
        self.timeScale = 1.0
        self._actions = []          # all known actions
        self._nActiveActions = 0    # prefix of _actions that is active
        self._bindings = {}         # (id(node), track name) -> _PropertyMixer
        self._actionsByClip = {}    # (id(clip), id(root)) -> AnimationAction
        self._pending_finish = []   # finished last tick; deactivated next tick

    # -- action factory ------------------------------------------------------
    def clipAction(self, clip, optionalRoot=None, blendMode=None):
        root = optionalRoot if optionalRoot is not None else self._root
        key = (id(clip), id(root))
        action = self._actionsByClip.get(key)
        if action is None:
            action = AnimationAction(self, clip, root, blendMode)
            self._actionsByClip[key] = action
            self._actions.append(action)
        return action

    def existingAction(self, clip, optionalRoot=None):
        root = optionalRoot if optionalRoot is not None else self._root
        return self._actionsByClip.get((id(clip), id(root)))

    # -- transport -----------------------------------------------------------
    def update(self, deltaTime=0.0):
        delta = float(deltaTime) * self.timeScale
        self.time += delta

        # Actions that finished on the previous tick are deactivated now,
        # so their end pose was visible for one full frame (three.js
        # restores the original state on deactivation).
        if self._pending_finish:
            for action in self._pending_finish:
                self._deactivateAction(action)
            self._pending_finish = []

        active = self._actions[:self._nActiveActions]
        for action in active:
            finished = action._update(delta)
            if finished and not action.clampWhenFinished:
                if action not in self._pending_finish:
                    self._pending_finish.append(action)

        for prop_mixer in self._bindings.values():
            if prop_mixer.cumulativeWeight > 0.0:
                prop_mixer.apply()
        return self

    def stopAllAction(self):
        for action in list(self._actions):
            self._deactivateAction(action)
            action.reset()
        self._pending_finish = []
        return self

    def setTime(self, timeInSeconds):
        # three.js semantics: rewind everything, then advance once.
        self.time = 0.0
        for action in self._actions:
            action.time = 0.0
            action._loopCount = 0
            action._direction = 1
        return self.update(timeInSeconds)

    def getRoot(self):
        return self._root

    # -- cache management ------------------------------------------------------
    def uncacheClip(self, clip):
        for key in [k for k in self._actionsByClip if k[0] == id(clip)]:
            action = self._actionsByClip.pop(key)
            self._deactivateAction(action)
            if action in self._actions:
                self._actions.remove(action)
            self._releaseActionBindings(action)

    def uncacheAction(self, clip, optionalRoot=None):
        root = optionalRoot if optionalRoot is not None else self._root
        action = self._actionsByClip.pop((id(clip), id(root)), None)
        if action is not None:
            self._deactivateAction(action)
            if action in self._actions:
                self._actions.remove(action)
            self._releaseActionBindings(action)

    def uncacheRoot(self, root):
        for key in [k for k in self._actionsByClip if k[1] == id(root)]:
            action = self._actionsByClip.pop(key)
            self._deactivateAction(action)
            if action in self._actions:
                self._actions.remove(action)
            self._releaseActionBindings(action)

    # -- activation bookkeeping ------------------------------------------------
    def _isActiveAction(self, action):
        idx = self._actions.index(action) if action in self._actions else -1
        return 0 <= idx < self._nActiveActions

    def _activateAction(self, action):
        if action not in self._actions:
            self._actions.append(action)
            self._actionsByClip[
                (id(action._clip), id(action._localRoot))] = action
        if not self._isActiveAction(action):
            idx = self._actions.index(action)
            self._actions.pop(idx)
            self._actions.insert(self._nActiveActions, action)
            self._nActiveActions += 1
            self._bindAction(action)
            for prop_mixer in action._propertyMixers:
                if prop_mixer.useCount == 0:
                    prop_mixer.saveOriginalState()
                prop_mixer.useCount += 1

    def _deactivateAction(self, action):
        if not self._isActiveAction(action):
            return
        idx = self._actions.index(action)
        last = self._nActiveActions - 1
        self._actions.pop(idx)
        self._actions.insert(last, action)
        self._nActiveActions -= 1
        if action in self._pending_finish:
            self._pending_finish.remove(action)
        for prop_mixer in action._propertyMixers:
            prop_mixer.useCount -= 1
            if prop_mixer.useCount == 0:
                prop_mixer.restoreOriginalState()

    def _bindAction(self, action):
        if action._bound:
            return
        action._bound = True
        root = action._localRoot
        mixers = []
        tracks = action._clip.tracks if action._clip is not None else []
        for track in tracks:
            binding = PropertyBinding(root, track.name)
            binding.bind()
            node_key = id(binding.node) if binding.node is not None else id(root)
            key = (node_key, track.name)
            prop_mixer = self._bindings.get(key)
            if prop_mixer is None:
                prop_mixer = _PropertyMixer(
                    binding, track.ValueTypeName, track.getValueSize())
                self._bindings[key] = prop_mixer
            prop_mixer.referenceCount += 1
            mixers.append(prop_mixer)
        action._propertyMixers = mixers

    def _releaseActionBindings(self, action):
        for prop_mixer in action._propertyMixers:
            prop_mixer.referenceCount -= 1
            if prop_mixer.referenceCount <= 0:
                for key, value in list(self._bindings.items()):
                    if value is prop_mixer:
                        del self._bindings[key]
        action._propertyMixers = []
        action._bound = False


# ---------------------------------------------------------------------------
# LineSegments, helpers, Sprite, Ring / Lathe geometries
# ---------------------------------------------------------------------------

def _as_vector3(value, default=None):
    """Coerce a Vector3-like / sequence / dict into a Vector3."""
    if value is None:
        return default if default is not None else Vector3()
    if isinstance(value, Vector3):
        return value
    x = getattr(value, "x", None)
    if x is not None:
        try:
            return Vector3(float(x), float(value.y), float(value.z))
        except Exception:
            pass
    if isinstance(value, dict):
        return Vector3(
            float(value.get("x", 0.0)),
            float(value.get("y", 0.0)),
            float(value.get("z", 0.0)),
        )
    try:
        seq = list(value)
        return Vector3(float(seq[0]), float(seq[1]), float(seq[2]))
    except Exception:
        return default if default is not None else Vector3()


class LineSegments(Line):
    """Independent segment pairs (three.js ``LineSegments``): vertices
    (0, 1), (2, 3), ... form separate segments instead of one strip."""

    isLineSegments = True


class RingGeometry(BufferGeometry):
    """Flat ring in the XY plane facing +Z (three.js ``RingGeometry``)."""

    def __init__(self, innerRadius=0.5, outerRadius=1, thetaSegments=16,
                 phiSegments=1, thetaStart=0, thetaLength=2.0 * math.pi):
        super().__init__()
        r_in = float(innerRadius)
        r_out = float(outerRadius)
        tseg = max(3, int(thetaSegments))
        pseg = max(1, int(phiSegments))
        positions = []
        normals = []
        uvs = []
        idx = []
        for j in range(pseg + 1):
            radius = r_in + (r_out - r_in) * (j / pseg)
            for i in range(tseg + 1):
                theta = float(thetaStart) + float(thetaLength) * i / tseg
                x = radius * math.cos(theta)
                y = radius * math.sin(theta)
                positions.append([x, y, 0.0])
                normals.append([0.0, 0.0, 1.0])
                if r_out != 0.0:
                    uvs.append([x / r_out * 0.5 + 0.5,
                                y / r_out * 0.5 + 0.5])
                else:
                    uvs.append([0.5, 0.5])
        grid = tseg + 1
        for j in range(pseg):
            for i in range(tseg):
                a = i + grid * j
                b = i + grid * (j + 1)
                c = (i + 1) + grid * (j + 1)
                d = (i + 1) + grid * j
                idx += [a, b, d, b, c, d]
        self.attributes["position"] = np.array(positions, dtype=np.float32)
        self.attributes["normal"] = np.array(normals, dtype=np.float32)
        self.attributes["uv"] = np.array(uvs, dtype=np.float32)
        self.index = np.array(idx, dtype=np.int64)


def _point_xy(p):
    """(x, y) floats from a Vector2/Vector3-like, dict or sequence."""
    x = getattr(p, "x", None)
    if x is not None:
        return float(x), float(p.y)
    if isinstance(p, dict):
        return float(p.get("x", 0.0)), float(p.get("y", 0.0))
    seq = list(p)
    return float(seq[0]), float(seq[1])


class LatheGeometry(BufferGeometry):
    """Surface of revolution around the Y axis (three.js
    ``LatheGeometry``).  ``points`` is a sequence of Vector2-like points
    (x = radius, y = height), revolved from ``phiStart`` over
    ``phiLength``.  Vertex normals come from the profile tangent
    (perpendicular in the profile plane, rotated with the point), the
    same construction three.js uses."""

    def __init__(self, points=None, segments=12, phiStart=0.0,
                 phiLength=2.0 * math.pi):
        super().__init__()
        pts = []
        for p in (points if points is not None else []):
            px, py = _point_xy(p)
            pts.append((px, py))
        seg = max(1, int(segments))
        if len(pts) < 2:
            self.attributes["position"] = np.zeros((0, 3), dtype=np.float32)
            self.attributes["normal"] = np.zeros((0, 3), dtype=np.float32)
            self.attributes["uv"] = np.zeros((0, 2), dtype=np.float32)
            self.index = np.zeros((0,), dtype=np.int64)
            return
        # Profile normals: perpendicular to the tangent in the (radius,
        # height) plane, pointing away from the axis for a profile that
        # runs bottom-to-top (tangent (0, 1) -> normal (1, 0)).
        prof_n = []
        last = len(pts) - 1
        for j in range(len(pts)):
            if j == 0:
                tx = pts[1][0] - pts[0][0]
                ty = pts[1][1] - pts[0][1]
            elif j == last:
                tx = pts[last][0] - pts[last - 1][0]
                ty = pts[last][1] - pts[last - 1][1]
            else:
                tx = pts[j + 1][0] - pts[j - 1][0]
                ty = pts[j + 1][1] - pts[j - 1][1]
            nx, ny = ty, -tx
            ln = math.hypot(nx, ny)
            if ln <= 0.0:
                nx, ny, ln = 1.0, 0.0, 1.0
            prof_n.append((nx / ln, ny / ln))
        positions = []
        normals = []
        uvs = []
        idx = []
        for i in range(seg + 1):
            phi = float(phiStart) + float(phiLength) * i / seg
            sin_p = math.sin(phi)
            cos_p = math.cos(phi)
            for j, (px, py) in enumerate(pts):
                positions.append([px * sin_p, py, px * cos_p])
                nx, ny = prof_n[j]
                normals.append([nx * sin_p, ny, nx * cos_p])
                uvs.append([i / seg, j / last])
        stride = len(pts)
        for i in range(seg):
            for j in range(last):
                base = j + i * stride
                a = base
                b = base + stride
                c = base + stride + 1
                d = base + 1
                idx += [a, b, d, b, c, d]
        self.attributes["position"] = np.array(positions, dtype=np.float32)
        self.attributes["normal"] = np.array(normals, dtype=np.float32)
        self.attributes["uv"] = np.array(uvs, dtype=np.float32)
        self.index = np.array(idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Sprite
# ---------------------------------------------------------------------------

class SpriteMaterial(Material):
    """Material for ``Sprite`` (three.js ``SpriteMaterial``): ``color``
    tints ``map``; ``rotation`` rotates the billboard in the view plane
    (radians); ``sizeAttenuation`` (default True) lets the depth shrink
    the sprite, False keeps a constant screen size."""

    isSpriteMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        self.rotation = float(p.get("rotation", 0.0))
        self.sizeAttenuation = bool(p.get("sizeAttenuation", True))


class Sprite(Object3D):
    """A camera-facing billboard quad (three.js ``Sprite``).  The
    renderer draws it in a dedicated pass; ``center`` (a Vector2,
    default (0.5, 0.5)) is the anchor point of the quad that sits on
    the sprite's position."""

    isSprite = True

    def __init__(self, material=None, name=""):
        super().__init__(name)
        self.material = (
            material if material is not None else SpriteMaterial()
        )
        self.center = Vector2(0.5, 0.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class GridHelper(LineSegments):
    """A square grid of lines on the XZ plane (three.js
    ``GridHelper``), drawn with per-vertex colours: the two centre
    lines use ``colorCenterLine``, the rest ``colorGrid``."""

    isGridHelper = True

    def __init__(self, size=10, divisions=10, colorCenterLine=0x444444,
                 colorGrid=0x888888):
        size = float(size)
        divisions = max(1, int(divisions))
        half = size / 2.0
        step = size / divisions
        center_col = _as_color(colorCenterLine, Color(0x444444))
        grid_col = _as_color(colorGrid, Color(0x888888))
        positions = []
        colors = []
        for i in range(divisions + 1):
            p = -half + step * i
            col = center_col if i == divisions / 2 else grid_col
            rgb = (col.r, col.g, col.b)
            positions += [
                [-half, 0.0, p], [half, 0.0, p],
                [p, 0.0, -half], [p, 0.0, half],
            ]
            colors += [rgb, rgb, rgb, rgb]
        geometry = BufferGeometry()
        geometry.setAttribute(
            "position",
            Float32BufferAttribute(
                np.array(positions, dtype=np.float32), 3),
        )
        geometry.setAttribute(
            "color",
            Float32BufferAttribute(np.array(colors, dtype=np.float32), 3),
        )
        super().__init__(
            geometry, LineBasicMaterial({"vertexColors": True})
        )
        self.size = size
        self.divisions = divisions


class AxesHelper(LineSegments):
    """The three coordinate axes as coloured segments from the origin
    (three.js ``AxesHelper``): +X red, +Y green, +Z blue."""

    isAxesHelper = True

    def __init__(self, size=1):
        s = float(size)
        positions = np.array(
            [
                [0.0, 0.0, 0.0], [s, 0.0, 0.0],
                [0.0, 0.0, 0.0], [0.0, s, 0.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, s],
            ],
            dtype=np.float32,
        )
        colors = np.array(
            [
                [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        geometry = BufferGeometry()
        geometry.setAttribute(
            "position", Float32BufferAttribute(positions, 3)
        )
        geometry.setAttribute(
            "color", Float32BufferAttribute(colors, 3)
        )
        super().__init__(
            geometry, LineBasicMaterial({"vertexColors": True})
        )


class ArrowHelper(Object3D):
    """An arrow from ``origin`` toward ``dir`` (three.js
    ``ArrowHelper``): a Line shaft plus a cone Mesh head, as children.
    ``dir`` is expected to be normalised (it is normalised here for
    safety)."""

    isArrowHelper = True

    def __init__(self, dir=None, origin=None, length=1, color=0xFFFF00,
                 headLength=None, headWidth=None):
        super().__init__()
        if dir is None:
            dir = Vector3(0.0, 0.0, 1.0)
        if origin is None:
            origin = Vector3(0.0, 0.0, 0.0)
        length = float(length)
        if headLength is None:
            headLength = 0.2 * length
        if headWidth is None:
            headWidth = 0.2 * float(headLength)
        self.position.copy(_as_vector3(origin))
        line_geo = BufferGeometry()
        line_geo.setAttribute(
            "position",
            Float32BufferAttribute(
                np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                3,
            ),
        )
        self.line = Line(line_geo, LineBasicMaterial({"color": color}))
        self.add(self.line)
        # Cone with its tip at the local origin and its base at y = -1,
        # so setLength can park the tip exactly at the arrow's end
        # (the three.js construction).
        cone_geo = ConeGeometry(0.5, 1, 5)
        cone_geo.translate(0.0, -0.5, 0.0)
        self.cone = Mesh(cone_geo, MeshBasicMaterial({"color": color}))
        self.add(self.cone)
        self.setDirection(dir)
        self.setLength(length, headLength, headWidth)

    def setDirection(self, dir):
        d = _as_vector3(dir).clone()
        if d.lengthSq() > 0.0:
            d.normalize()
        if d.y > 0.99999:
            self.quaternion.set(0.0, 0.0, 0.0, 1.0)
        elif d.y < -0.99999:
            self.quaternion.set(1.0, 0.0, 0.0, 0.0)
        else:
            axis = Vector3(d.z, 0.0, -d.x)
            if axis.lengthSq() <= 0.0:
                axis.set(1.0, 0.0, 0.0)
            else:
                axis.normalize()
            angle = math.acos(_clamp(d.y, -1.0, 1.0))
            self.quaternion.setFromAxisAngle(axis, angle)
        return self

    def setLength(self, length, headLength=None, headWidth=None):
        length = float(length)
        if headLength is None:
            headLength = 0.2 * length
        headLength = float(headLength)
        if headWidth is None:
            headWidth = 0.2 * headLength
        headWidth = float(headWidth)
        self.line.scale.set(1.0, length - headLength, 1.0)
        self.cone.scale.set(headWidth, headLength, headWidth)
        self.cone.position.y = length
        return self

    def setColor(self, color):
        self.line.material.color.set(color)
        self.cone.material.color.set(color)
        return self


def _object_world_aabb(root):
    """(lo, hi) numpy arrays: the world-space AABB over every
    descendant's geometry positions (root included), or None when no
    geometry with positions exists below ``root``."""
    root.updateWorldMatrix(True, True)
    lo = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    hi = -lo.copy()
    found = False
    nodes = []
    root.traverse(nodes.append)
    for node in nodes:
        geometry = getattr(node, "geometry", None)
        if geometry is None:
            continue
        attrs = getattr(geometry, "attributes", None)
        if not isinstance(attrs, dict):
            continue
        pos = attrs.get("position")
        if pos is None:
            continue
        arr = np.asarray(pos, dtype=np.float64)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, 3)
        m = node.matrixWorld._a
        world = arr @ m[:3, :3].T + m[:3, 3]
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
        found = True
    return (lo, hi) if found else None


class BoxHelper(LineSegments):
    """A wireframe box around an object's world AABB (three.js
    ``BoxHelper``).  The box corners live in the geometry in world
    space; call ``update()`` after the object moves (three.js has the
    same contract)."""

    isBoxHelper = True

    # three.js BoxHelper edge list over the 8 AABB corners
    _EDGES = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    def __init__(self, object=None, color=0xFFFF00):
        geometry = BufferGeometry()
        geometry.setAttribute(
            "position",
            Float32BufferAttribute(np.zeros((24, 3), dtype=np.float32), 3),
        )
        super().__init__(
            geometry, LineBasicMaterial({"color": color})
        )
        self.object = object
        self.update(object)

    def update(self, object=None):
        if object is not None:
            self.object = object
        pos = np.zeros((24, 3), dtype=np.float32)
        box = (
            _object_world_aabb(self.object)
            if self.object is not None
            else None
        )
        if box is not None:
            lo, hi = box
            corners = np.array(
                [
                    [lo[0], lo[1], lo[2]],
                    [lo[0], lo[1], hi[2]],
                    [lo[0], hi[1], hi[2]],
                    [lo[0], hi[1], lo[2]],
                    [hi[0], lo[1], lo[2]],
                    [hi[0], lo[1], hi[2]],
                    [hi[0], hi[1], hi[2]],
                    [hi[0], hi[1], lo[2]],
                ],
                dtype=np.float64,
            )
            for k, (a, b) in enumerate(self._EDGES):
                pos[2 * k] = corners[a]
                pos[2 * k + 1] = corners[b]
        self.geometry.setAttribute(
            "position", Float32BufferAttribute(pos, 3)
        )
        return self


class PointLightHelper(Mesh):
    """three.js ``PointLightHelper``: a wireframe sphere (4x2
    segments) marking its light's world position, coloured by the
    light (or by the explicit ``color`` argument).

    three.js shares the light's matrixWorld as the helper's matrix;
    this port's renderer composes world matrices from local
    transforms, so the helper instead syncs its local position to
    the light's world position on every scene-graph refresh."""

    isPointLightHelper = True

    def __init__(self, light, sphereSize=None, color=None):
        geometry = SphereGeometry(
            1.0 if sphereSize is None else float(sphereSize), 4, 2)
        material = MeshBasicMaterial({"wireframe": True, "fog": False})
        super().__init__(geometry, material)
        self.type = "PointLightHelper"
        self.light = light
        self.color = color
        self.update()

    def update(self):
        if self.color is not None:
            self.material.color.set(self.color)
        elif self.light is not None:
            self.material.color.copy(self.light.color)
        return self

    def updateMatrixWorld(self, force=False):
        if self.light is not None:
            wp = self.light.getWorldPosition()
            self.position.set(wp.x, wp.y, wp.z)
        return super().updateMatrixWorld(force)


# ---------------------------------------------------------------------------
# Core math / texture classes the manual lessons reach for (squad 2):
# Matrix3, Plane, Sphere, Box3, Frustum, DataTexture and the image
# loaders, plus the remaining light/scene helpers.
# ---------------------------------------------------------------------------

class Matrix3:
    """3x3 matrix (three.js ``Matrix3``): ``elements`` is a flat
    column-major list of 9, as in the original."""

    isMatrix3 = True

    def __init__(self, *args):
        if len(args) == 9:
            self.elements = [float(v) for v in args]
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            a = np.asarray(args[0], dtype=np.float64).reshape(3, 3)
            self.elements = [float(v) for v in a.flatten(order="F")]
        else:
            self.elements = [1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0]

    def _a(self):
        return np.array(self.elements, dtype=np.float64).reshape(
            3, 3, order="F")

    def _set_a(self, a):
        self.elements = [float(v) for v in
                         np.asarray(a, dtype=np.float64).flatten(order="F")]
        return self

    def set(self, n11, n12, n13, n21, n22, n23, n31, n32, n33):
        # arguments are row by row, as in three.js
        self.elements = [float(n11), float(n21), float(n31),
                         float(n12), float(n22), float(n32),
                         float(n13), float(n23), float(n33)]
        return self

    def identity(self):
        self.elements = [1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0]
        return self

    def copy(self, m):
        self.elements = list(m.elements)
        return self

    def clone(self):
        out = Matrix3()
        out.elements = list(self.elements)
        return out

    def fromArray(self, array, offset=0):
        offset = int(offset)
        self.elements = [float(array[offset + i]) for i in range(9)]
        return self

    def toArray(self, array=None, offset=0):
        if array is None:
            return list(self.elements)
        offset = int(offset)
        for i, value in enumerate(self.elements):
            array[offset + i] = value
        return array

    def multiply(self, m):
        return self.multiplyMatrices(self, m)

    def premultiply(self, m):
        return self.multiplyMatrices(m, self)

    def multiplyMatrices(self, a, b):
        return self._set_a(a._a() @ b._a())

    def multiplyScalar(self, s):
        self.elements = [v * float(s) for v in self.elements]
        return self

    def determinant(self):
        return float(np.linalg.det(self._a()))

    def invert(self):
        det = self.determinant()
        if det == 0.0:
            return self.identity()
        return self._set_a(np.linalg.inv(self._a()))

    def transpose(self):
        return self._set_a(self._a().T)

    def getNormalMatrix(self, matrix4):
        """Set this matrix to the inverse-transpose of the upper 3x3
        of ``matrix4`` (three.js Matrix3.getNormalMatrix)."""
        a = np.asarray(matrix4._a, dtype=np.float64)[:3, :3]
        try:
            normal = np.linalg.inv(a).T
        except np.linalg.LinAlgError:
            normal = np.identity(3)
        return self._set_a(normal)

    def setFromMatrix4(self, m):
        a = np.asarray(m._a, dtype=np.float64)[:3, :3]
        return self._set_a(a)

    def equals(self, m):
        return list(self.elements) == list(m.elements)


class Plane:
    """A plane ``normal . p + constant = 0`` (three.js ``Plane``)."""

    isPlane = True

    def __init__(self, normal=None, constant=0.0):
        self.normal = (normal.clone() if normal is not None
                       else Vector3(0.0, 0.0, 1.0))
        self.constant = float(constant)

    def set(self, normal, constant):
        self.normal.copy(normal)
        self.constant = float(constant)
        return self

    def setComponents(self, x, y, z, w):
        self.normal.set(float(x), float(y), float(z))
        self.constant = float(w)
        return self

    def setFromNormalAndCoplanarPoint(self, normal, point):
        self.normal.copy(normal)
        self.constant = -point.dot(self.normal)
        return self

    def setFromCoplanarPoints(self, a, b, c):
        ab = Vector3().subVectors(b, a)
        ac = Vector3().subVectors(c, a)
        normal = Vector3().crossVectors(ac, ab)
        if normal.lengthSq() > 0.0:
            normal.normalize()
        return self.setFromNormalAndCoplanarPoint(normal, a)

    def copy(self, plane):
        self.normal.copy(plane.normal)
        self.constant = float(plane.constant)
        return self

    def clone(self):
        return Plane(self.normal.clone(), self.constant)

    def normalize(self):
        length = self.normal.length()
        if length > 0.0:
            inv = 1.0 / length
            self.normal.multiplyScalar(inv)
            self.constant *= inv
        return self

    def negate(self):
        self.constant *= -1.0
        self.normal.negate()
        return self

    def distanceToPoint(self, point):
        return self.normal.dot(point) + self.constant

    def distanceToSphere(self, sphere):
        center, radius = _sphere_parts(sphere)
        return self.distanceToPoint(center) - radius

    def projectPoint(self, point, target=None):
        if target is None:
            target = Vector3()
        distance = self.distanceToPoint(point)
        return target.copy(point).addScaledVector(self.normal, -distance)

    def coplanarPoint(self, target=None):
        if target is None:
            target = Vector3()
        return target.copy(self.normal).multiplyScalar(-self.constant)

    def applyMatrix4(self, matrix, normalMatrix=None):
        if normalMatrix is None:
            normalMatrix = Matrix3().getNormalMatrix(matrix)
        reference = self.coplanarPoint()
        ref = Vector3(reference.x, reference.y, reference.z)
        normal = Vector3(
            self.normal.x, self.normal.y, self.normal.z)
        a = normalMatrix._a()
        n = a @ np.array([normal.x, normal.y, normal.z])
        length = float(np.linalg.norm(n))
        if length > 0.0:
            n = n / length
        ref.applyMatrix4(matrix)
        self.normal.set(float(n[0]), float(n[1]), float(n[2]))
        self.constant = -ref.dot(self.normal)
        return self

    def translate(self, offset):
        self.constant -= offset.dot(self.normal)
        return self

    def equals(self, plane):
        return (self.normal.equals(plane.normal)
                and self.constant == plane.constant)


def _sphere_parts(sphere):
    """Normalise the port's sphere representations -- a Sphere, a
    ``(center, radius)`` tuple (BufferGeometry.boundingSphere), or a
    dict -- to ``(Vector3 center, float radius)``."""
    if isinstance(sphere, Sphere):
        return sphere.center, float(sphere.radius)
    if isinstance(sphere, dict):
        return sphere.get("center"), float(sphere.get("radius", 0.0))
    center, radius = sphere
    return center, float(radius)


class Sphere:
    """A sphere (three.js ``Sphere``); empty when radius < 0."""

    isSphere = True

    def __init__(self, center=None, radius=-1.0):
        self.center = (center.clone() if center is not None
                       else Vector3())
        self.radius = float(radius)

    def set(self, center, radius):
        self.center.copy(center)
        self.radius = float(radius)
        return self

    def setFromPoints(self, points, optionalCenter=None):
        if optionalCenter is not None:
            self.center.copy(optionalCenter)
        else:
            box = Box3().setFromPoints(points)
            box.getCenter(self.center)
        max_radius = 0.0
        for point in points:
            max_radius = max(max_radius,
                             self.center.distanceToSquared(point))
        self.radius = math.sqrt(max_radius)
        return self

    def copy(self, sphere):
        self.center.copy(sphere.center)
        self.radius = float(sphere.radius)
        return self

    def clone(self):
        return Sphere(self.center.clone(), self.radius)

    def isEmpty(self):
        return self.radius < 0.0

    def makeEmpty(self):
        self.radius = -1.0
        return self

    def containsPoint(self, point):
        return point.distanceToSquared(self.center) <= self.radius ** 2

    def distanceToPoint(self, point):
        return point.distanceTo(self.center) - self.radius

    def intersectsSphere(self, sphere):
        center, radius = _sphere_parts(sphere)
        return (center.distanceToSquared(self.center)
                <= (self.radius + radius) ** 2)

    def intersectsBox(self, box):
        return box.intersectsSphere(self)

    def intersectsPlane(self, plane):
        return abs(plane.distanceToPoint(self.center)) <= self.radius

    def clampPoint(self, point, target=None):
        if target is None:
            target = Vector3()
        delta = Vector3().subVectors(point, self.center)
        length_sq = delta.lengthSq()
        if length_sq > self.radius ** 2:
            if length_sq > 0.0:
                delta.multiplyScalar(self.radius / math.sqrt(length_sq))
            target.copy(self.center).add(delta)
        else:
            target.copy(point)
        return target

    def getBoundingBox(self, target=None):
        if target is None:
            target = Box3()
        if self.isEmpty():
            return target.makeEmpty()
        target.set(
            Vector3(self.center.x - self.radius,
                    self.center.y - self.radius,
                    self.center.z - self.radius),
            Vector3(self.center.x + self.radius,
                    self.center.y + self.radius,
                    self.center.z + self.radius),
        )
        return target

    def applyMatrix4(self, matrix):
        self.center.applyMatrix4(matrix)
        scale = _max_scale_on_axis(matrix)
        self.radius = self.radius * scale
        return self

    def translate(self, offset):
        self.center.add(offset)
        return self

    def expandByPoint(self, point):
        if self.isEmpty():
            return self.set(point.clone(), 0.0)
        delta = Vector3().subVectors(point, self.center)
        length = delta.length()
        if length > self.radius:
            # three.js expands the sphere to just contain the point.
            half = (length - self.radius) / 2.0
            if length > 0.0:
                self.center.addScaledVector(delta, half / length)
            self.radius += half
        return self

    def union(self, sphere):
        other_center, other_radius = _sphere_parts(sphere)
        if self.isEmpty():
            return self.set(other_center.clone(), other_radius)
        if other_radius < 0.0:
            return self
        distance = self.center.distanceTo(other_center)
        if distance + other_radius <= self.radius:
            return self
        if distance + self.radius <= other_radius:
            return self.set(other_center.clone(), other_radius)
        half = (distance + self.radius + other_radius) / 2.0
        if distance > 0.0:
            direction = Vector3().subVectors(
                other_center, self.center).multiplyScalar(1.0 / distance)
            self.center.addScaledVector(
                direction, half - self.radius)
        self.radius = half
        return self

    def equals(self, sphere):
        return (self.center.equals(sphere.center)
                and self.radius == sphere.radius)


def _max_scale_on_axis(matrix):
    a = np.asarray(matrix._a, dtype=np.float64)
    sx = float(np.linalg.norm(a[:3, 0]))
    sy = float(np.linalg.norm(a[:3, 1]))
    sz = float(np.linalg.norm(a[:3, 2]))
    return max(sx, sy, sz)


_INF = float("inf")


class Box3:
    """An axis-aligned bounding box (three.js ``Box3``): empty when
    ``min`` is +inf / ``max`` is -inf componentwise, as in the
    original."""

    isBox3 = True

    def __init__(self, min=None, max=None):
        self.min = (min.clone() if min is not None
                    else Vector3(_INF, _INF, _INF))
        self.max = (max.clone() if max is not None
                    else Vector3(-_INF, -_INF, -_INF))

    def set(self, min, max):
        self.min.copy(min)
        self.max.copy(max)
        return self

    def setFromPoints(self, points):
        self.makeEmpty()
        for point in points:
            self.expandByPoint(point)
        return self

    def setFromCenterAndSize(self, center, size):
        half = Vector3(size.x / 2.0, size.y / 2.0, size.z / 2.0)
        self.min.copy(center).sub(half)
        self.max.copy(center).add(half)
        return self

    def setFromObject(self, object, precise=False):
        self.makeEmpty()
        return self.expandByObject(object, precise)

    def expandByObject(self, object, precise=False):
        object.updateWorldMatrix(False, True)
        stack = [object]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            geometry = getattr(node, "geometry", None)
            if geometry is None:
                continue
            if getattr(geometry, "boundingBox", None) is None:
                try:
                    geometry.computeBoundingBox()
                except Exception:
                    continue
            bounds = getattr(geometry, "boundingBox", None)
            if bounds is None:
                continue
            lo, hi = bounds
            corners = [
                Vector3(x, y, z)
                for x in (lo.x, hi.x)
                for y in (lo.y, hi.y)
                for z in (lo.z, hi.z)
            ]
            for corner in corners:
                corner.applyMatrix4(node.matrixWorld)
                self.expandByPoint(corner)
        return self

    def setFromBufferAttribute(self, attribute):
        arr = np.asarray(attribute.array
                         if hasattr(attribute, "array") else attribute,
                         dtype=np.float64)
        if arr.size == 0:
            return self.makeEmpty()
        arr = arr.reshape(-1, 3)
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        self.min.set(float(lo[0]), float(lo[1]), float(lo[2]))
        self.max.set(float(hi[0]), float(hi[1]), float(hi[2]))
        return self

    def clone(self):
        return Box3(self.min.clone(), self.max.clone())

    def copy(self, box):
        self.min.copy(box.min)
        self.max.copy(box.max)
        return self

    def makeEmpty(self):
        self.min.set(_INF, _INF, _INF)
        self.max.set(-_INF, -_INF, -_INF)
        return self

    def isEmpty(self):
        return (self.max.x < self.min.x or self.max.y < self.min.y
                or self.max.z < self.min.z)

    def getCenter(self, target=None):
        if target is None:
            target = Vector3()
        if self.isEmpty():
            return target.set(0.0, 0.0, 0.0)
        return target.addVectors(self.min, self.max).multiplyScalar(0.5)

    def getSize(self, target=None):
        if target is None:
            target = Vector3()
        if self.isEmpty():
            return target.set(0.0, 0.0, 0.0)
        return target.subVectors(self.max, self.min)

    def expandByPoint(self, point):
        self.min.set(min(self.min.x, point.x),
                     min(self.min.y, point.y),
                     min(self.min.z, point.z))
        self.max.set(max(self.max.x, point.x),
                     max(self.max.y, point.y),
                     max(self.max.z, point.z))
        return self

    def expandByVector(self, vector):
        self.min.sub(vector)
        self.max.add(vector)
        return self

    def expandByScalar(self, scalar):
        scalar = float(scalar)
        self.min.addScalar(-scalar)
        self.max.addScalar(scalar)
        return self

    def containsPoint(self, point):
        return (point.x >= self.min.x and point.x <= self.max.x
                and point.y >= self.min.y and point.y <= self.max.y
                and point.z >= self.min.z and point.z <= self.max.z)

    def containsBox(self, box):
        return (self.min.x <= box.min.x and box.max.x <= self.max.x
                and self.min.y <= box.min.y and box.max.y <= self.max.y
                and self.min.z <= box.min.z and box.max.z <= self.max.z)

    def intersectsBox(self, box):
        return not (box.max.x < self.min.x or box.min.x > self.max.x
                    or box.max.y < self.min.y or box.min.y > self.max.y
                    or box.max.z < self.min.z or box.min.z > self.max.z)

    def intersectsSphere(self, sphere):
        center, radius = _sphere_parts(sphere)
        closest = Vector3(
            _clamp(center.x, self.min.x, self.max.x),
            _clamp(center.y, self.min.y, self.max.y),
            _clamp(center.z, self.min.z, self.max.z),
        )
        return closest.distanceToSquared(center) <= radius * radius

    def intersectsPlane(self, plane):
        # three.js Box3.intersectsPlane via the closest/farthest
        # corners along the plane normal.
        normal = plane.normal
        if normal.x >= 0:
            near_x, far_x = self.max.x, self.min.x
        else:
            near_x, far_x = self.min.x, self.max.x
        if normal.y >= 0:
            near_y, far_y = self.max.y, self.min.y
        else:
            near_y, far_y = self.min.y, self.max.y
        if normal.z >= 0:
            near_z, far_z = self.max.z, self.min.z
        else:
            near_z, far_z = self.min.z, self.max.z
        near = plane.distanceToPoint(Vector3(near_x, near_y, near_z))
        far = plane.distanceToPoint(Vector3(far_x, far_y, far_z))
        return near <= 0.0 <= far or (near <= 0.0 and far <= 0.0) \
            or near * far <= 0.0

    def clampPoint(self, point, target=None):
        if target is None:
            target = Vector3()
        return target.copy(point).clamp(self.min, self.max)

    def distanceToPoint(self, point):
        closest = self.clampPoint(point)
        return closest.sub(point).length()

    def getBoundingSphere(self, target=None):
        if target is None:
            target = Sphere()
        self.getCenter(target.center)
        target.radius = (self.getSize(Vector3()).length() / 2.0
                         if not self.isEmpty() else 0.0)
        return target

    def intersect(self, box):
        self.min.set(max(self.min.x, box.min.x),
                     max(self.min.y, box.min.y),
                     max(self.min.z, box.min.z))
        self.max.set(min(self.max.x, box.max.x),
                     min(self.max.y, box.max.y),
                     min(self.max.z, box.max.z))
        if self.isEmpty():
            self.makeEmpty()
        return self

    def union(self, box):
        self.min.set(min(self.min.x, box.min.x),
                     min(self.min.y, box.min.y),
                     min(self.min.z, box.min.z))
        self.max.set(max(self.max.x, box.max.x),
                     max(self.max.y, box.max.y),
                     max(self.max.z, box.max.z))
        return self

    def applyMatrix4(self, matrix):
        if self.isEmpty():
            return self
        points = [
            Vector3(x, y, z)
            for x in (self.min.x, self.max.x)
            for y in (self.min.y, self.max.y)
            for z in (self.min.z, self.max.z)
        ]
        self.makeEmpty()
        for point in points:
            point.applyMatrix4(matrix)
            self.expandByPoint(point)
        return self

    def translate(self, offset):
        self.min.add(offset)
        self.max.add(offset)
        return self

    def equals(self, box):
        return self.min.equals(box.min) and self.max.equals(box.max)


class Frustum:
    """A view frustum of six planes (three.js ``Frustum``)."""

    isFrustum = True

    def __init__(self, p0=None, p1=None, p2=None, p3=None, p4=None,
                 p5=None):
        given = [p0, p1, p2, p3, p4, p5]
        self.planes = [p.clone() if p is not None else Plane()
                       for p in given]

    def set(self, p0, p1, p2, p3, p4, p5):
        for i, plane in enumerate([p0, p1, p2, p3, p4, p5]):
            self.planes[i].copy(plane)
        return self

    def copy(self, frustum):
        for i in range(6):
            self.planes[i].copy(frustum.planes[i])
        return self

    def clone(self):
        out = Frustum()
        out.copy(self)
        return out

    def setFromProjectionMatrix(self, m):
        e = m.elements  # column-major
        rows = [
            (e[0], e[4], e[8], e[12]),
            (e[1], e[5], e[9], e[13]),
            (e[2], e[6], e[10], e[14]),
            (e[3], e[7], e[11], e[15]),
        ]
        combos = [
            (rows[3], rows[0], 1.0),   # left
            (rows[3], rows[0], -1.0),  # right
            (rows[3], rows[1], 1.0),   # bottom
            (rows[3], rows[1], -1.0),  # top
            (rows[3], rows[2], 1.0),   # near
            (rows[3], rows[2], -1.0),  # far
        ]
        for plane, (base, other, sign) in zip(self.planes, combos):
            plane.setComponents(
                base[0] + sign * other[0],
                base[1] + sign * other[1],
                base[2] + sign * other[2],
                base[3] + sign * other[3],
            ).normalize()
        return self

    def intersectsObject(self, object):
        geometry = getattr(object, "geometry", None)
        if geometry is not None:
            if getattr(geometry, "boundingSphere", None) is None:
                geometry.computeBoundingSphere()
            center, radius = _sphere_parts(geometry.boundingSphere)
            sphere = Sphere(center.clone(), radius)
            sphere.applyMatrix4(object.matrixWorld)
            return self.intersectsSphere(sphere)
        sphere = Sphere(object.getWorldPosition(), 0.0)
        return self.intersectsSphere(sphere)

    def intersectsSprite(self, sprite):
        return self.intersectsObject(sprite)

    def intersectsSphere(self, sphere):
        center, radius = _sphere_parts(sphere)
        for plane in self.planes:
            if plane.distanceToPoint(center) < -radius:
                return False
        return True

    def intersectsBox(self, box):
        for plane in self.planes:
            normal = plane.normal
            px = box.max.x if normal.x >= 0 else box.min.x
            py = box.max.y if normal.y >= 0 else box.min.y
            pz = box.max.z if normal.z >= 0 else box.min.z
            if plane.distanceToPoint(Vector3(px, py, pz)) < 0.0:
                return False
        return True

    def containsPoint(self, point):
        for plane in self.planes:
            if plane.distanceToPoint(point) < 0.0:
                return False
        return True


# Legacy numeric encodings (the manual's bundled three is r132-era):
# materials and textures written against those builds read and write
# ``texture.encoding`` with these values.  They coexist with the
# r187 colour-space strings; the renderer keys off the string form
# and treats the numeric form as metadata.
LinearEncoding = 1000
sRGBEncoding = 1001
GammaEncoding = 1002
RGBEEncoding = 1003
LogLuvEncoding = 1004
RGBM7Encoding = 1005
RGBM16Encoding = 1006
RGBDEncoding = 1007


class DataTexture(Texture):
    """A texture built from raw typed data (three.js
    ``DataTexture``).  ``data`` may be a typed array (the
    interpreter's JSTypedArray), a JS array, or a numpy array; the
    image is materialised as the (h, w, 3) float array the renderer
    samples, and the raw data stays available as ``.data``."""

    isDataTexture = True

    def __init__(self, data=None, width=1, height=1, format=None,
                 type=None, mapping=None, wrapS=None, wrapT=None,
                 magFilter=None, minFilter=None, anisotropy=1,
                 colorSpace=None):
        super().__init__(None)
        self.width = int(width)
        self.height = int(height)
        self.format = RGBAFormat if format is None else format
        self.type = type if type is not None else UnsignedByteType
        if mapping is not None:
            self.mapping = mapping
        if wrapS is not None:
            self.wrapS = wrapS
        if wrapT is not None:
            self.wrapT = wrapT
        if magFilter is not None:
            self.magFilter = magFilter
        if minFilter is not None:
            self.minFilter = minFilter
        self.data = data
        self.image = {"data": data, "width": self.width,
                      "height": self.height}
        self._materialise()
        self.needsUpdate = True

    def _flat_data(self):
        data = self.data
        if data is None:
            return None
        raw = getattr(data, "_data", None)  # JSTypedArray storage
        if raw is not None:
            return np.asarray(list(raw))
        try:
            return np.asarray(data)
        except Exception:
            return None

    def _materialise(self):
        arr = self._flat_data()
        if arr is None or arr.size == 0:
            self._image_array = None
            return
        channels = 4 if self.format == RGBAFormat else \
            3 if self.format == RGBFormat else 1
        count = self.width * self.height * channels
        flat = arr.reshape(-1)
        if flat.size < count:
            padded = np.zeros(count, dtype=flat.dtype)
            padded[:flat.size] = flat
            flat = padded
        flat = flat[:count]
        if np.issubdtype(flat.dtype, np.floating):
            if flat.size and float(np.max(flat)) <= 1.0:
                flat = flat * 255.0
            flat = np.clip(flat, 0, 255)
        grid = flat.reshape(self.height, self.width, channels)
        rgb = grid[:, :, :3] if channels >= 3 else np.repeat(
            grid, 3, axis=2)
        self._image_array = np.ascontiguousarray(
            rgb.astype(np.float64) / 255.0)

    @property
    def image(self):
        return self._image_array

    @image.setter
    def image(self, value):
        # The Texture constructor assigns .image; keep the raw-data
        # dict shape three.js exposes without clobbering the
        # materialised array (set by _materialise afterwards).
        if isinstance(value, dict) and "data" in value:
            self._image_dict = value
        elif value is not None:
            coerced = _coerce_texture_image(value)
            self._image_array = coerced


class _LoadedImage(Texture):
    """What ImageLoader hands back: a Texture that also carries the
    DOM-image surface page code reads (``width`` / ``height``)."""

    def __init__(self, image_array):
        super().__init__(image_array)
        if image_array is not None:
            self.height, self.width = image_array.shape[:2]
        else:
            self.width = 0
            self.height = 0


class ImageLoader:
    """three.js ``ImageLoader``: resolves an image by name from the
    same on-device registry TextureLoader uses and hands back an
    image object synchronously (the bytes are already local)."""

    isImageLoader = True

    def __init__(self, manager=None):
        self.manager = manager if manager is not None \
            else DefaultLoadingManager
        self.crossOrigin = "anonymous"
        self.path = ""

    def setCrossOrigin(self, crossOrigin):
        self.crossOrigin = crossOrigin
        return self

    def setPath(self, path):
        self.path = str(path)
        return self

    def load(self, name, onLoad=None, onProgress=None, onError=None):
        url = str(name)
        self.manager.itemStart(url)
        texture = TextureLoader().load(url)
        image = _LoadedImage(texture.image)
        if onLoad is not None:
            onLoad(image)
        self.manager.itemEnd(url)
        return image


class CubeTextureLoader:
    """three.js ``CubeTextureLoader``: six image urls -> a
    CubeTexture, each face resolved from the TextureLoader registry
    (the px/nx/py/ny/pz/nz order of the original)."""

    isCubeTextureLoader = True

    def __init__(self, manager=None):
        self.manager = manager if manager is not None \
            else DefaultLoadingManager
        self.crossOrigin = "anonymous"
        self.path = ""

    def setCrossOrigin(self, crossOrigin):
        self.crossOrigin = crossOrigin
        return self

    def setPath(self, path):
        self.path = str(path)
        return self

    def load(self, urls, onLoad=None, onProgress=None, onError=None):
        loader = TextureLoader()
        faces = []
        for url in list(urls):
            full = self.path + str(url)
            self.manager.itemStart(full)
            texture = loader.load(full)
            faces.append(texture.image if texture is not None else None)
            self.manager.itemEnd(full)
        while len(faces) < 6:
            faces.append(None)
        cube = CubeTexture(faces[:6])
        if onLoad is not None:
            onLoad(cube)
        return cube


def _line_segments(positions, color, vertex_colors=None):
    """Build a LineSegments from a flat position list (helper
    boilerplate shared by the helpers below)."""
    geometry = BufferGeometry()
    geometry.setAttribute(
        "position",
        Float32BufferAttribute(
            np.array(positions, dtype=np.float32).reshape(-1, 3), 3),
    )
    if vertex_colors is not None:
        geometry.setAttribute(
            "color",
            Float32BufferAttribute(
                np.array(vertex_colors, dtype=np.float32).reshape(-1, 3),
                3),
        )
        material = LineBasicMaterial({"vertexColors": True})
    else:
        material = LineBasicMaterial({"color": color})
    return LineSegments(geometry, material)


class DirectionalLightHelper(LineSegments):
    """three.js ``DirectionalLightHelper``: a square on the light's
    plane plus a line toward the target.  The geometry is rebuilt by
    ``update()`` from the light's world position/direction (the port
    composes world matrices locally, so the helper bakes world-space
    points and sits at the origin, like CameraHelper)."""

    isDirectionalLightHelper = True

    def __init__(self, light, size=1, color=None):
        self.light = light
        self.size = float(size)
        if color is None and light is not None:
            color = light.color.getHex()
        super().__init__(BufferGeometry(),
                         LineBasicMaterial({"color": color or 0xffffff}))
        self.type = "DirectionalLightHelper"
        self.update()

    def update(self):
        if self.light is None:
            return self
        position = self.light.getWorldPosition()
        target = (self.light.target.getWorldPosition()
                  if self.light.target is not None else Vector3())
        direction = Vector3().subVectors(position, target)
        length = direction.length()
        if length > 0.0:
            direction.multiplyScalar(1.0 / length)
        # Build an orthonormal basis around the direction.
        up = Vector3(0.0, 1.0, 0.0)
        if abs(direction.dot(up)) > 0.99:
            up = Vector3(1.0, 0.0, 0.0)
        side = Vector3().crossVectors(direction, up).normalize()
        upv = Vector3().crossVectors(side, direction).normalize()
        half = self.size / 2.0
        corners = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corner = position.clone()
            corner.addScaledVector(side, sx * half)
            corner.addScaledVector(upv, sy * half)
            corners.append(corner)
        positions = []
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            positions += [[a.x, a.y, a.z], [b.x, b.y, b.z]]
        positions += [[position.x, position.y, position.z],
                      [target.x, target.y, target.z]]
        self.geometry.setAttribute(
            "position",
            Float32BufferAttribute(
                np.array(positions, dtype=np.float32), 3))
        return self


class SpotLightHelper(LineSegments):
    """three.js ``SpotLightHelper``: a wire cone from the light
    toward its target with the light's angle.  World-space points
    baked by ``update()``, like DirectionalLightHelper."""

    isSpotLightHelper = True

    def __init__(self, light, color=None):
        self.light = light
        if color is None and light is not None:
            color = light.color.getHex()
        super().__init__(BufferGeometry(),
                         LineBasicMaterial({"color": color or 0xffffff}))
        self.type = "SpotLightHelper"
        self.update()

    def update(self):
        if self.light is None:
            return self
        position = self.light.getWorldPosition()
        target = (self.light.target.getWorldPosition()
                  if self.light.target is not None else Vector3())
        direction = Vector3().subVectors(target, position)
        length = direction.length()
        if length <= 0.0:
            length = 1.0
            direction = Vector3(0.0, -1.0, 0.0)
        else:
            direction.multiplyScalar(1.0 / length)
        radius = math.tan(float(self.light.angle)) * length
        up = Vector3(0.0, 1.0, 0.0)
        if abs(direction.dot(up)) > 0.99:
            up = Vector3(1.0, 0.0, 0.0)
        side = Vector3().crossVectors(direction, up).normalize()
        upv = Vector3().crossVectors(side, direction).normalize()
        base_center = position.clone().addScaledVector(direction, length)
        segments = 8
        ring = []
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            point = base_center.clone()
            point.addScaledVector(side, math.cos(angle) * radius)
            point.addScaledVector(upv, math.sin(angle) * radius)
            ring.append(point)
        positions = []
        for i in range(segments):
            a = ring[i]
            b = ring[(i + 1) % segments]
            positions += [[a.x, a.y, a.z], [b.x, b.y, b.z]]
            positions += [[position.x, position.y, position.z],
                          [a.x, a.y, a.z]]
        self.geometry.setAttribute(
            "position",
            Float32BufferAttribute(
                np.array(positions, dtype=np.float32), 3))
        return self


class HemisphereLightHelper(Mesh):
    """three.js ``HemisphereLightHelper``: a small octahedron at the
    light's position, sky colour on top / ground colour below."""

    isHemisphereLightHelper = True

    def __init__(self, light, size=1, color=None):
        geometry = OctahedronGeometry(float(size))
        material = MeshBasicMaterial({
            "vertexColors": True, "wireframe": True, "fog": False,
        })
        super().__init__(geometry, material)
        self.type = "HemisphereLightHelper"
        self.light = light
        self.color = color
        self.update()

    def update(self):
        if self.light is not None:
            sky = self.light.color
            ground = self.light.groundColor
            pos = self.geometry.attributes.get("position")
            if pos is not None:
                arr = np.asarray(pos.array if hasattr(pos, "array")
                                 else pos, dtype=np.float32)
                colors = np.where(
                    (arr[:, 1:2] >= 0.0),
                    np.array([sky.r, sky.g, sky.b], dtype=np.float32),
                    np.array([ground.r, ground.g, ground.b],
                             dtype=np.float32),
                )
                self.geometry.setAttribute(
                    "color", Float32BufferAttribute(colors, 3))
            wp = self.light.getWorldPosition()
            self.position.set(wp.x, wp.y, wp.z)
        return self

    def updateMatrixWorld(self, force=False):
        if self.light is not None:
            wp = self.light.getWorldPosition()
            self.position.set(wp.x, wp.y, wp.z)
        return super().updateMatrixWorld(force)


class CameraHelper(LineSegments):
    """three.js ``CameraHelper``: the camera's frustum as line
    segments.  Points are baked in world space by ``update()`` (the
    helper itself sits at the origin), from the camera's inverse
    view-projection matrix."""

    isCameraHelper = True

    def __init__(self, camera):
        self.camera = camera
        super().__init__(
            BufferGeometry(),
            LineBasicMaterial({"vertexColors": True, "toneMapped": False}))
        self.type = "CameraHelper"
        self.pointMap = {}
        self.update()

    def update(self):
        if self.camera is None:
            return self
        camera = self.camera
        camera.updateMatrixWorld()
        projection = getattr(camera, "projectionMatrix", None)
        if projection is None:
            return self
        view = Matrix4().copy(camera.matrixWorld).invert()
        mvp = Matrix4().copy(projection).multiply(view)
        inverse = mvp.clone().invert()
        ndc_corners = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
        ]
        world = []
        for x, y, z in ndc_corners:
            v = Vector3(float(x), float(y), float(z))
            v.applyMatrix4(inverse)
            world.append(v)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        positions = []
        colors = []
        colour = (1.0, 1.0, 1.0)
        for a, b in edges:
            positions += [[world[a].x, world[a].y, world[a].z],
                          [world[b].x, world[b].y, world[b].z]]
            colors += [colour, colour]
        # Up indicator on the near plane, as in the original.
        mid_top = world[2].clone().add(world[3]).multiplyScalar(0.5)
        above = mid_top.clone()
        above.y += (world[2].distanceTo(world[3])) * 0.25
        positions += [[mid_top.x, mid_top.y, mid_top.z],
                      [above.x, above.y, above.z]]
        colors += [colour, colour]
        self.geometry.setAttribute(
            "position",
            Float32BufferAttribute(
                np.array(positions, dtype=np.float32), 3))
        self.geometry.setAttribute(
            "color",
            Float32BufferAttribute(
                np.array(colors, dtype=np.float32), 3))
        return self


class PolarGridHelper(LineSegments):
    """three.js ``PolarGridHelper``: concentric circles and radial
    spokes on the XZ plane with per-vertex colours."""

    isPolarGridHelper = True

    def __init__(self, radius=10, sectors=16, rings=8, divisions=64,
                 color1=0x444444, color2=0x888888):
        radius = float(radius)
        sectors = max(1, int(sectors))
        rings = max(1, int(rings))
        divisions = max(8, int(divisions))
        c1 = _as_color(color1, Color(0x444444))
        c2 = _as_color(color2, Color(0x888888))
        positions = []
        colors = []
        for ring in range(1, rings + 1):
            r = radius * ring / rings
            col = c1 if ring == rings else c2
            rgb = (col.r, col.g, col.b)
            for i in range(divisions):
                a0 = 2.0 * math.pi * i / divisions
                a1 = 2.0 * math.pi * (i + 1) / divisions
                positions += [
                    [math.cos(a0) * r, 0.0, math.sin(a0) * r],
                    [math.cos(a1) * r, 0.0, math.sin(a1) * r],
                ]
                colors += [rgb, rgb]
        rgb = (c1.r, c1.g, c1.b)
        for i in range(sectors):
            angle = 2.0 * math.pi * i / sectors
            positions += [
                [0.0, 0.0, 0.0],
                [math.cos(angle) * radius, 0.0,
                 math.sin(angle) * radius],
            ]
            colors += [rgb, rgb]
        geometry = BufferGeometry()
        geometry.setAttribute(
            "position",
            Float32BufferAttribute(
                np.array(positions, dtype=np.float32), 3))
        geometry.setAttribute(
            "color",
            Float32BufferAttribute(
                np.array(colors, dtype=np.float32), 3))
        super().__init__(geometry,
                         LineBasicMaterial({"vertexColors": True}))
        self.radius = radius
        self.sectors = sectors
        self.rings = rings


def _box_outline_positions(box):
    lo, hi = box.min, box.max
    corners = [
        Vector3(x, y, z)
        for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)
    ]
    # corner index bits: x = bit0, y = bit1, z = bit2
    edges = [(0, 1), (2, 3), (4, 5), (6, 7),
             (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    positions = []
    for a, b in edges:
        positions += [[corners[a].x, corners[a].y, corners[a].z],
                      [corners[b].x, corners[b].y, corners[b].z]]
    return positions


class Box3Helper(LineSegments):
    """three.js ``Box3Helper``: the outline of a Box3."""

    isBox3Helper = True

    def __init__(self, box, color=0xffff00):
        self.box = box
        positions = (_box_outline_positions(box)
                     if box is not None and not box.isEmpty() else [])
        helper = _line_segments(positions, color)
        super().__init__(helper.geometry, helper.material)
        self.type = "Box3Helper"

    def updateMatrixWorld(self, force=False):
        box = self.box
        if box is not None and not box.isEmpty():
            self.geometry.setAttribute(
                "position",
                Float32BufferAttribute(
                    np.array(_box_outline_positions(box),
                             dtype=np.float32), 3))
        return super().updateMatrixWorld(force)


class BoxHelper(LineSegments):
    """three.js ``BoxHelper``: the world-space Box3 outline of an
    object, refreshed by ``update()``."""

    isBoxHelper = True

    def __init__(self, object, color=0xffff00):
        self.object = object
        box = Box3().setFromObject(object) if object is not None \
            else Box3()
        positions = ([] if box.isEmpty()
                     else _box_outline_positions(box))
        helper = _line_segments(positions, color)
        super().__init__(helper.geometry, helper.material)
        self.type = "BoxHelper"

    def update(self, object=None):
        if object is not None:
            self.object = object
        if self.object is None:
            return self
        box = Box3().setFromObject(self.object)
        if not box.isEmpty():
            self.geometry.setAttribute(
                "position",
                Float32BufferAttribute(
                    np.array(_box_outline_positions(box),
                             dtype=np.float32), 3))
        return self

    def setFromObject(self, object):
        self.object = object
        return self.update()


class ArrowHelper(Object3D):
    """three.js ``ArrowHelper``: a line plus a cone head along a
    direction."""

    isArrowHelper = True

    def __init__(self, dir=None, origin=None, length=1, color=0xffff00,
                 headLength=None, headWidth=None):
        super().__init__()
        self.type = "ArrowHelper"
        length = float(length)
        if headLength is None:
            headLength = min(length * 0.2, 1.0)
        if headWidth is None:
            headWidth = headLength * 0.4
        line_positions = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        line_geometry = BufferGeometry()
        line_geometry.setAttribute(
            "position",
            Float32BufferAttribute(
                np.array(line_positions, dtype=np.float32), 3))
        self.line = Line(line_geometry,
                         LineBasicMaterial({"color": color,
                                            "toneMapped": False}))
        cone_geometry = CylinderGeometry(0.0, 0.5, 1.0, 12, 1, True)
        cone_geometry.translate(0.0, -0.5, 0.0)
        self.cone = Mesh(cone_geometry,
                         MeshBasicMaterial({"color": color,
                                            "toneMapped": False}))
        self.add(self.line)
        self.add(self.cone)
        if origin is not None:
            self.position.copy(origin)
        self._color = color
        if dir is not None:
            self.setDirection(dir)
        self.setLength(length, headLength, headWidth)

    def setDirection(self, dir):
        length = dir.length()
        if length > 0.0:
            direction = dir.clone().multiplyScalar(1.0 / length)
        else:
            direction = Vector3(0.0, 1.0, 0.0)
        if direction.y > 0.99999:
            self.quaternion.set(0.0, 0.0, 0.0, 1.0)
        elif direction.y < -0.99999:
            self.quaternion.set(1.0, 0.0, 0.0, 0.0)
        else:
            axis = Vector3(direction.z, 0.0, -direction.x)
            if axis.lengthSq() > 0.0:
                axis.normalize()
            angle = math.acos(_clamp(direction.y, -1.0, 1.0))
            self.quaternion.setFromAxisAngle(axis, angle)
        return self

    def setLength(self, length, headLength=None, headWidth=None):
        if headLength is None:
            headLength = min(float(length) * 0.2, 1.0)
        if headWidth is None:
            headWidth = headLength * 0.4
        self.line.scale.set(1.0, float(length) - float(headLength), 1.0)
        self.cone.scale.set(float(headWidth), float(headLength), 1.0)
        self.cone.position.y = float(length) - float(headLength)
        return self

    def setColor(self, color):
        self.line.material.color.set(color)
        self.cone.material.color.set(color)
        return self


class EventDispatcher:
    """three.js ``EventDispatcher``: the listener registry base
    class (also subclassed directly by page code).  Listeners live
    in ``_listeners`` {type: [callback]}, the shape the
    interpreter's dispatch_event reads."""

    isEventDispatcher = True

    def __init__(self):
        self._listeners = {}

    def addEventListener(self, type, listener, *args, **kwargs):
        self._listeners.setdefault(str(type), []).append(listener)

    def hasEventListener(self, type, listener):
        return listener in self._listeners.get(str(type), [])

    def removeEventListener(self, type, listener, *args, **kwargs):
        callbacks = self._listeners.get(str(type), [])
        if listener in callbacks:
            callbacks.remove(listener)

    def removeAllListeners(self, type=None):
        if type is None:
            self._listeners = {}
        else:
            self._listeners.pop(str(type), None)

    def dispatchEvent(self, event):
        callbacks = []
        if isinstance(event, dict):
            callbacks = list(self._listeners.get(
                str(event.get("type", "")), []))
        for callback in callbacks:
            callback(event)
        return self


class TrackballControls:
    """A minimal TrackballControls (three/addons): the port renders
    on demand, so the control only needs to exist, accept its
    property bag, track the camera on ``update()`` with a simple
    orbit from pointer drags, and fire no events.  Drag handling is
    wired through the interpreter's listener registry when a DOM
    element is supplied."""

    isTrackballControls = True

    def __init__(self, camera, domElement=None):
        self.object = camera
        self.camera = camera
        self.domElement = domElement
        self.enabled = True
        self.screenSpacePanning = True
        self.rotateSpeed = 1.0
        self.zoomSpeed = 1.2
        self.panSpeed = 0.8
        self.noRotate = False
        self.noZoom = False
        self.noPan = False
        self.staticMoving = False
        self.dynamicDampingFactor = 0.2
        self.minDistance = 0.0
        self.maxDistance = float("inf")
        self.minZoom = 0.0
        self.maxZoom = float("inf")
        self.target = Vector3()
        self._listeners = {}

    def addEventListener(self, type, listener, *args):
        self._listeners.setdefault(str(type), []).append(listener)

    def removeEventListener(self, type, listener, *args):
        callbacks = self._listeners.get(str(type), [])
        if listener in callbacks:
            callbacks.remove(listener)

    def update(self, deltaTime=None):
        return True

    def reset(self):
        return self

    def dispose(self):
        return None


class VRButton:
    """three/addons VRButton: creates a DOM button in a browser.
    There is no DOM to attach to here; ``createButton`` returns a
    stub element (the WebVR lessons read the XR flags off the
    renderer and only optionally append the button)."""

    @staticmethod
    def createButton(renderer, options=None):
        from jsinterp import _StubElement
        return _StubElement()

    @staticmethod
    def registerSessionGrantedListener():
        return None


# ---------------------------------------------------------------------------
# MathUtils
# ---------------------------------------------------------------------------
# three.js ``MathUtils`` is an object literal of pure functions; here it is
# a class whose static methods play that role (``THREE.MathUtils.clamp``
# resolves through the module attribute, exactly like the original).

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def _imul32(a, b):
    """JavaScript ``Math.imul``: 32-bit integer multiplication, returned
    as an unsigned 32-bit bit pattern (downstream bitwise ops in the
    seeded generator only depend on the bit pattern)."""
    return (a * b) & 0xFFFFFFFF


class MathUtils:
    """The three.js math utility collection."""

    DEG2RAD = DEG2RAD
    RAD2DEG = RAD2DEG

    _seed = 1234567  # the original's initial seed

    @staticmethod
    def generateUUID():
        # The original's construction (version 4 / variant 10xx layout),
        # driven by the random module instead of Math.random.
        d0 = random.getrandbits(32)
        d1 = random.getrandbits(32)
        d2 = random.getrandbits(32)
        d3 = random.getrandbits(32)

        def h(v):
            return "%02x" % (v & 0xFF)

        return "".join(
            [
                h(d0), h(d0 >> 8), h(d0 >> 16), h(d0 >> 24), "-",
                h(d1), h(d1 >> 8), "-",
                h((d1 >> 16 & 0x0F) | 0x40), h(d1 >> 24), "-",
                h((d2 & 0x3F) | 0x80), h(d2 >> 8), "-",
                h(d2 >> 16), h(d2 >> 24),
                h(d3), h(d3 >> 8), h(d3 >> 16), h(d3 >> 24),
            ]
        )

    @staticmethod
    def clamp(value, min, max):
        # the original's Math.max(min, Math.min(max, value)), spelled out
        # because the parameter names shadow the Python builtins here.
        if value < min:
            return min
        if value > max:
            return max
        return value

    @staticmethod
    def euclideanModulo(n, m):
        return ((n % m) + m) % m

    @staticmethod
    def mapLinear(x, a1, a2, b1, b2):
        return b1 + (x - a1) * (b2 - b1) / (a2 - a1)

    @staticmethod
    def inverseLerp(x, y, value):
        if x != y:
            return (value - x) / (y - x)
        return 0

    @staticmethod
    def lerp(x, y, t):
        return (1 - t) * x + t * y

    @staticmethod
    def damp(x, y, lam, dt):
        return MathUtils.lerp(x, y, 1 - math.exp(-lam * dt))

    @staticmethod
    def pingpong(x, length=1):
        return length - abs(MathUtils.euclideanModulo(x, length * 2) - length)

    @staticmethod
    def smoothstep(x, min, max):
        if x <= min:
            return 0
        if x >= max:
            return 1
        x = (x - min) / (max - min)
        return x * x * (3 - 2 * x)

    @staticmethod
    def smootherstep(x, min, max):
        if x <= min:
            return 0
        if x >= max:
            return 1
        x = (x - min) / (max - min)
        return x * x * x * (x * (x * 6 - 15) + 10)

    @staticmethod
    def randInt(low, high):
        return low + math.floor(random.random() * (high - low + 1))

    @staticmethod
    def randFloat(low, high):
        return low + random.random() * (high - low)

    @staticmethod
    def randFloatSpread(range):
        return range * (0.5 - random.random())

    @staticmethod
    def seededRandom(s=None):
        # Mulberry32, bit-faithful to the original (32-bit ops masked).
        if s is not None:
            MathUtils._seed = int(s) & 0xFFFFFFFF
        MathUtils._seed = (MathUtils._seed + 0x6D2B79F5) & 0xFFFFFFFF
        t = MathUtils._seed
        t = _imul32(t ^ (t >> 15), t | 1)
        t = t ^ ((t + _imul32(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF)
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    @staticmethod
    def degToRad(degrees):
        return degrees * DEG2RAD

    @staticmethod
    def radToDeg(radians):
        return radians * RAD2DEG

    @staticmethod
    def isPowerOfTwo(value):
        return (
            value > 0
            and float(value).is_integer()
            and 2 ** round(math.log2(value)) == value
        )

    @staticmethod
    def ceilPowerOfTwo(value):
        return 2 ** math.ceil(math.log(value) / math.log(2))

    @staticmethod
    def floorPowerOfTwo(value):
        return 2 ** math.floor(math.log(value) / math.log(2))

    @staticmethod
    def setQuaternionFromProperEuler(q, a, b, c, order):
        cos = math.cos
        sin = math.sin
        c2 = cos(b / 2)
        s2 = sin(b / 2)
        c13 = cos((a + c) / 2)
        s13 = sin((a + c) / 2)
        c1_3 = cos((a - c) / 2)
        s1_3 = sin((a - c) / 2)
        c3_1 = cos((c - a) / 2)
        s3_1 = sin((c - a) / 2)
        if order == "XYX":
            q.set(c2 * s13, s2 * c1_3, s2 * s1_3, c2 * c13)
        elif order == "YZY":
            q.set(s2 * s1_3, c2 * s13, s2 * c1_3, c2 * c13)
        elif order == "ZXZ":
            q.set(s2 * c1_3, s2 * s1_3, c2 * s13, c2 * c13)
        elif order == "XZX":
            q.set(c2 * s13, s2 * s3_1, s2 * c3_1, c2 * c13)
        elif order == "YXY":
            q.set(s2 * c3_1, c2 * s13, s2 * s3_1, c2 * c13)
        elif order == "ZYZ":
            q.set(s2 * s3_1, s2 * c3_1, c2 * s13, c2 * c13)
        # unknown order: the original only warns; q is left unchanged.

    @staticmethod
    def getNormalizationScale(array):
        arr = getattr(array, "array", array)
        dtype = getattr(arr, "dtype", None)
        if dtype is None:
            return None
        if dtype == np.uint32:
            return 4294967295.0
        if dtype == np.uint16:
            return 65535.0
        if dtype == np.uint8:
            return 255.0
        if dtype == np.int32:
            return 2147483647.0
        if dtype == np.int16:
            return 32767.0
        if dtype == np.int8:
            return 127.0
        return None

    @staticmethod
    def denormalize(value, array):
        arr = getattr(array, "array", array)
        if getattr(arr, "dtype", None) == np.float32:
            return value
        scale = MathUtils.getNormalizationScale(array)
        if scale is None:
            raise ValueError("THREE.MathUtils: Invalid component type.")
        return max(value / scale, -1.0)

    @staticmethod
    def normalize(value, array):
        arr = getattr(array, "array", array)
        if getattr(arr, "dtype", None) == np.float32:
            return value
        scale = MathUtils.getNormalizationScale(array)
        if scale is None:
            raise ValueError("THREE.MathUtils: Invalid component type.")
        return round(value * scale)


__all__ += [
    "DEG2RAD",
    "RAD2DEG",
    "MathUtils",
]


# ---------------------------------------------------------------------------
# OBJLoader -- facade; the parser lives in obj.py
# ---------------------------------------------------------------------------

class MaterialCreator:
    """three.js ``MTLLoader.MaterialCreator`` stand-in: the parsed
    materials of one .mtl library, in the shape page code touches --
    ``materials`` (name -> Material), ``preload()`` (a no-op here: the
    port resolves everything synchronously) and ``createMaterial()``.
    Only the diffuse colour ``Kd`` is honoured, as in obj.py."""

    isMaterialCreator = True

    def __init__(self, rgb_by_name=None):
        self._rgb = dict(rgb_by_name or {})
        self.materials = {}
        for name, rgb in self._rgb.items():
            hex_value = 0
            for component in rgb:
                hex_value = (hex_value << 8) | int(round(
                    min(1.0, max(0.0, float(component))) * 255.0))
            self.materials[name] = MeshLambertMaterial(
                {"color": hex_value})
        self.materialsInfo = {
            name: {"kd": list(rgb)} for name, rgb in self._rgb.items()
        }

    def preload(self):
        return self

    def createMaterial(self, name):
        material = self.materials.get(str(name))
        if material is not None:
            return material
        return MeshLambertMaterial()

    def getAsArray(self):
        return list(self.materials.values())


def _registry_text(registry, name):
    """Fetch text registered under ``name`` (basename fallback), the
    obj.py ``_registry_lookup`` rule, for the loader facades."""
    key = str(name)
    data = registry.get(key)
    if data is None and ("/" in key or "\\" in key):
        data = registry.get(key.replace("\\", "/").split("/")[-1])
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data).decode("utf-8", errors="replace")
    return data


class MTLLoader:
    """three.js ``MTLLoader`` facade: ``load(name, onLoad)`` parses a
    dropped ``.mtl`` (registered by the host alongside ``.obj`` files)
    and calls ``onLoad`` with a :class:`MaterialCreator`, which page
    code hands to ``OBJLoader.setMaterials``.  Synchronous, like the
    other loader facades."""

    isMTLLoader = True

    _registry = {}

    def __init__(self, manager=None):
        self.manager = manager

    @classmethod
    def register(cls, name, data):
        if isinstance(data, (bytes, bytearray, memoryview)):
            cls._registry[str(name)] = bytes(data).decode(
                "utf-8", errors="replace")
            return True
        if isinstance(data, str):
            cls._registry[str(name)] = data
            return True
        return False

    def load(self, name, onLoad=None, onProgress=None, onError=None):
        obj_module = OBJLoader._load_obj_module()
        manager = _manager_of(self)
        manager.itemStart(str(name))
        try:
            text = _registry_text(self._registry, name)
            if text is None:
                text = _registry_text(OBJLoader._registry, name)
            if text is None:
                raise obj_module.OBJError(
                    "no dropped MTL file named %r" % (str(name),))
            creator = MaterialCreator(obj_module.parse_mtl(text))
        except Exception as exc:
            manager.itemError(str(name))
            manager.itemEnd(str(name))
            if onError is not None:
                onError(exc)
                return None
            raise
        if onLoad is not None:
            onLoad(creator)
        manager.itemEnd(str(name))
        return creator

    def parse(self, text, path=""):
        obj_module = OBJLoader._load_obj_module()
        return MaterialCreator(obj_module.parse_mtl(text))

    def setPath(self, path):
        return self

    def setResourcePath(self, path):
        return self

    def setCrossOrigin(self, value):
        return self

    def loadAsync(self, name, onProgress=None):
        try:
            res = self.load(name, onProgress=onProgress)
            return SyncPromise.resolved(res)
        except Exception as exc:
            return SyncPromise.rejected(exc)


class OBJLoader:
    """Minimal Wavefront OBJ loader facade (three.js ``OBJLoader``
    shape).

    The host (main.py) registers the text of dropped ``.obj`` (and
    ``.mtl``) files by file name; ``load(name, onLoad)`` parses them
    with the sibling module ``obj.py`` and calls ``onLoad`` with the
    resulting ``Group`` (the real OBJLoader hands the group itself to
    the callback, unlike GLTFLoader's result object).  Loading is
    synchronous, so ``onLoad`` runs before ``load`` returns; the
    group is also returned.
    """

    isOBJLoader = True

    _registry = {}

    def __init__(self, manager=None):
        self.manager = manager
        self._materials_override = None

    def setMaterials(self, materials):
        """three.js ``OBJLoader.setMaterials``: accept a
        MaterialCreator (from MTLLoader), a plain {name: Material}
        mapping, or a {name: (r, g, b)} mapping; the materials then
        take precedence over the .obj's own ``mtllib`` resolution."""
        rgb = None
        if materials is not None:
            rgb_map = getattr(materials, "_rgb", None)
            if isinstance(rgb_map, dict):
                rgb = dict(rgb_map)
            else:
                source = getattr(materials, "materials", materials)
                if isinstance(source, dict):
                    rgb = {}
                    for name, value in source.items():
                        color = getattr(value, "color", None)
                        if color is not None:
                            rgb[str(name)] = (color.r, color.g, color.b)
                        elif isinstance(value, (tuple, list)) \
                                and len(value) >= 3:
                            rgb[str(name)] = (
                                float(value[0]), float(value[1]),
                                float(value[2]))
        self._materials_override = rgb
        return self

    @classmethod
    def register(cls, name, data):
        if isinstance(data, (bytes, bytearray, memoryview)):
            cls._registry[str(name)] = bytes(data).decode(
                "utf-8", errors="replace"
            )
            return True
        if isinstance(data, str):
            cls._registry[str(name)] = data
            return True
        return False

    @classmethod
    def unregister(cls, name):
        cls._registry.pop(str(name), None)

    @classmethod
    def registered_names(cls):
        return sorted(cls._registry.keys())

    def load(self, name, onLoad=None, onProgress=None, onError=None):
        obj_module = self._load_obj_module()
        manager = _manager_of(self)
        manager.itemStart(str(name))
        try:
            group = obj_module.load_named(
                name, materials=self._materials_override)
        except Exception as exc:
            manager.itemError(str(name))
            manager.itemEnd(str(name))
            if onError is not None:
                onError(exc)
                return None
            raise
        if onLoad is not None:
            onLoad(group)
        manager.itemEnd(str(name))
        return group

    def loadAsync(self, name, onProgress=None):
        """Synchronous bridge for three.js ``Loader.loadAsync``
       : runs the synchronous ``load`` now and returns an
        already-settled SyncPromise (rejected when the load fails).
        Code that depends on the real asynchronous ordering is out
        of scope."""
        try:
            return SyncPromise.resolved(self.load(name))
        except Exception as exc:
            return SyncPromise.rejected(exc)

    def parse(self, data, onLoad=None, onError=None):
        """Parse OBJ text directly (three.js OBJLoader.parse shape)."""
        obj_module = self._load_obj_module()
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data).decode("utf-8", errors="replace")
        try:
            group = obj_module.load_text(data)
        except Exception as exc:
            if onError is not None:
                onError(exc)
                return None
            raise
        if onLoad is not None:
            onLoad(group)
        return group

    # three.js also exposes setPath(); paths are meaningless here (the
    # registry is keyed by plain names), so it is accepted and ignored.
    def setPath(self, path):
        return self

    @staticmethod
    def _load_obj_module():
        try:
            import obj as obj_module

            return obj_module
        except ImportError as exc:
            raise ImportError(
                "three.py: sibling module 'obj.py' is required for "
                "OBJLoader but could not be imported"
            ) from exc


# ---------------------------------------------------------------------------
# Paths and shapes: Path / Shape, triangulation,
# ShapeGeometry / ExtrudeGeometry
# ---------------------------------------------------------------------------

class _LineCurve:
    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1

    def getPoint(self, t):
        if t == 1.0:
            return self.v1.clone()
        return Vector2(
            self.v0.x + (self.v1.x - self.v0.x) * t,
            self.v0.y + (self.v1.y - self.v0.y) * t,
        )

    def getPoints(self, divisions=5):
        divisions = int(divisions)
        return [self.getPoint(i / divisions) for i in range(divisions + 1)]


class _QuadraticBezierCurve:
    def __init__(self, v0, v1, v2):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def getPoint(self, t):
        u = 1.0 - t
        return Vector2(
            u * u * self.v0.x + 2 * u * t * self.v1.x + t * t * self.v2.x,
            u * u * self.v0.y + 2 * u * t * self.v1.y + t * t * self.v2.y,
        )

    def getPoints(self, divisions=5):
        divisions = int(divisions)
        return [self.getPoint(i / divisions) for i in range(divisions + 1)]


class _CubicBezierCurve:
    def __init__(self, v0, v1, v2, v3):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def getPoint(self, t):
        u = 1.0 - t
        a = u * u * u
        b = 3 * u * u * t
        c = 3 * u * t * t
        d = t * t * t
        return Vector2(
            a * self.v0.x + b * self.v1.x + c * self.v2.x + d * self.v3.x,
            a * self.v0.y + b * self.v1.y + c * self.v2.y + d * self.v3.y,
        )

    def getPoints(self, divisions=5):
        divisions = int(divisions)
        return [self.getPoint(i / divisions) for i in range(divisions + 1)]


class _EllipseCurve:
    def __init__(self, aX, aY, xRadius, yRadius, aStartAngle, aEndAngle,
                 aClockwise=False, aRotation=0.0):
        self.aX = float(aX)
        self.aY = float(aY)
        self.xRadius = float(xRadius)
        self.yRadius = float(yRadius)
        self.aStartAngle = float(aStartAngle)
        self.aEndAngle = float(aEndAngle)
        self.aClockwise = bool(aClockwise)
        self.aRotation = float(aRotation)

    def getPoint(self, t):
        two_pi = 2.0 * math.pi
        delta = self.aEndAngle - self.aStartAngle
        same_points = abs(delta) < 1e-12
        while delta < 0.0:
            delta += two_pi
        while delta > two_pi:
            delta -= two_pi
        if delta < 1e-12:
            delta = 0.0 if same_points else two_pi
        if self.aClockwise and not same_points:
            delta = -two_pi if delta == two_pi else delta - two_pi
        angle = self.aStartAngle + t * delta
        x = self.aX + self.xRadius * math.cos(angle)
        y = self.aY + self.yRadius * math.sin(angle)
        if self.aRotation != 0.0:
            cos_r = math.cos(self.aRotation)
            sin_r = math.sin(self.aRotation)
            dx = x - self.aX
            dy = y - self.aY
            x = self.aX + dx * cos_r - dy * sin_r
            y = self.aY + dx * sin_r + dy * cos_r
        return Vector2(x, y)

    def getPoints(self, divisions=5):
        divisions = int(divisions)
        return [self.getPoint(i / divisions) for i in range(divisions + 1)]


class _SplineCurve:
    """2D Catmull-Rom spline through its points (three.js SplineCurve)."""

    def __init__(self, points):
        self.points = list(points)

    def getPoint(self, t):
        pts = self.points
        n = len(pts)
        if n == 0:
            return Vector2()
        if n == 1:
            return pts[0].clone()
        p = (n - 1) * t
        i = int(math.floor(p))
        if i > n - 2:
            i = n - 2
        weight = p - i
        p0 = pts[i - 1] if i > 0 else pts[0]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < n else pts[n - 1]
        w2 = weight * weight
        w3 = w2 * weight

        def _cr(v0, v1, v2, v3):
            return (
                0.5 * (2.0 * v1)
                + 0.5 * (v2 - v0) * weight
                + (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v3) * w2
                + (-v0 + 3.0 * v1 - 3.0 * v2 + v3) * w3
            )

        return Vector2(
            _cr(p0.x, p1.x, p2.x, p3.x),
            _cr(p0.y, p1.y, p2.y, p3.y),
        )

    def getPoints(self, divisions=5):
        divisions = int(divisions)
        return [self.getPoint(i / divisions) for i in range(divisions + 1)]


class Path:
    """A 2D path of lines and curves (three.js ``Path``)."""

    isPath = True

    def __init__(self, points=None):
        self.curves = []
        self.currentPoint = Vector2(0.0, 0.0)
        if points is not None:
            self.setFromPoints(points)

    def setFromPoints(self, points):
        seq = list(points)
        if not seq:
            return self
        first = seq[0]
        fx, fy = _point_xy(first)
        self.moveTo(fx, fy)
        for p in seq[1:]:
            x, y = _point_xy(p)
            self.lineTo(x, y)
        return self

    def moveTo(self, x, y):
        self.currentPoint = Vector2(float(x), float(y))
        return self

    def closePath(self):
        if self.curves:
            start = self.curves[0].getPoint(0.0)
            if (abs(start.x - self.currentPoint.x) > 1e-9
                    or abs(start.y - self.currentPoint.y) > 1e-9):
                self.lineTo(start.x, start.y)
        return self

    def lineTo(self, x, y):
        end = Vector2(float(x), float(y))
        if end.x != self.currentPoint.x or end.y != self.currentPoint.y:
            self.curves.append(_LineCurve(self.currentPoint.clone(), end))
            self.currentPoint = end
        return self

    def quadraticCurveTo(self, aCPx, aCPy, aX, aY):
        end = Vector2(float(aX), float(aY))
        self.curves.append(
            _QuadraticBezierCurve(
                self.currentPoint.clone(),
                Vector2(float(aCPx), float(aCPy)),
                end,
            )
        )
        self.currentPoint = end
        return self

    def bezierCurveTo(self, aCP1x, aCP1y, aCP2x, aCP2y, aX, aY):
        end = Vector2(float(aX), float(aY))
        self.curves.append(
            _CubicBezierCurve(
                self.currentPoint.clone(),
                Vector2(float(aCP1x), float(aCP1y)),
                Vector2(float(aCP2x), float(aCP2y)),
                end,
            )
        )
        self.currentPoint = end
        return self

    def splineThru(self, points):
        pts = [self.currentPoint.clone()]
        for p in points:
            x, y = _point_xy(p)
            pts.append(Vector2(x, y))
        curve = _SplineCurve(pts)
        self.curves.append(curve)
        self.currentPoint = pts[-1].clone()
        return self

    def absarc(self, aX, aY, aRadius, aStartAngle, aEndAngle,
               aClockwise=False):
        return self.absellipse(
            aX, aY, aRadius, aRadius, aStartAngle, aEndAngle, aClockwise, 0.0
        )

    def arc(self, aX, aY, aRadius, aStartAngle, aEndAngle,
            aClockwise=False):
        return self.absarc(aX, aY, aRadius, aStartAngle, aEndAngle,
                           aClockwise)

    def absellipse(self, aX, aY, xRadius, yRadius, aStartAngle, aEndAngle,
                   aClockwise=False, aRotation=0.0):
        curve = _EllipseCurve(
            aX, aY, xRadius, yRadius, aStartAngle, aEndAngle,
            aClockwise, aRotation,
        )
        if self.curves:
            first = curve.getPoint(0.0)
            if (abs(first.x - self.currentPoint.x) > 1e-9
                    or abs(first.y - self.currentPoint.y) > 1e-9):
                self.lineTo(first.x, first.y)
        self.curves.append(curve)
        self.currentPoint = curve.getPoint(1.0)
        return self

    def ellipse(self, aX, aY, xRadius, yRadius, aStartAngle, aEndAngle,
                aClockwise=False, aRotation=0.0):
        return self.absellipse(
            aX, aY, xRadius, yRadius, aStartAngle, aEndAngle,
            aClockwise, aRotation,
        )

    def getPoints(self, divisions=12):
        divisions = max(1, int(divisions))
        points = []
        last = None
        for curve in self.curves:
            if isinstance(curve, _EllipseCurve):
                resolution = divisions * 2
            elif isinstance(curve, _LineCurve):
                resolution = 1
            elif isinstance(curve, _SplineCurve):
                resolution = divisions * max(1, len(curve.points))
            else:
                resolution = divisions
            for p in curve.getPoints(resolution):
                if (last is not None and abs(p.x - last.x) < 1e-9
                        and abs(p.y - last.y) < 1e-9):
                    continue
                points.append(p)
                last = p
        if not points:
            points.append(self.currentPoint.clone())
        return points


class Shape(Path):
    """A 2D shape: a Path contour plus hole paths (three.js ``Shape``)."""

    isShape = True

    def __init__(self, points=None):
        super().__init__(points)
        self.holes = []

    def extractPoints(self, divisions=12):
        return {
            "shape": self.getPoints(divisions),
            "holes": [hole.getPoints(divisions) for hole in self.holes],
        }

    def getSpacedPoints(self, divisions=12):
        return self.getPoints(divisions)


# -- triangulation (ear clipping with hole bridging) ------------------------

def _shape_signed_area(pts):
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area * 0.5


def _shape_ring(points):
    """Sampled contour -> list of (x, y) tuples, no closing duplicate."""
    pts = []
    for p in points:
        x, y = _point_xy(p)
        if pts and abs(x - pts[-1][0]) < 1e-9 and abs(y - pts[-1][1]) < 1e-9:
            continue
        pts.append((x, y))
    if len(pts) > 1:
        if (abs(pts[0][0] - pts[-1][0]) < 1e-9
                and abs(pts[0][1] - pts[-1][1]) < 1e-9):
            pts.pop()
    return pts


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_proper_cross(a, b, c, d):
    """True when segments a-b and c-d cross at a point interior to both."""
    d1 = _orient(c, d, a)
    d2 = _orient(c, d, b)
    d3 = _orient(a, b, c)
    d4 = _orient(a, b, d)
    return (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
        and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    )


def _bridge_hole(poly, hole):
    """Splice a CW hole ring into a CCW polygon via a bridge edge.

    Returns a new weakly-simple polygon (bridge endpoints duplicated),
    still CCW overall, ready for ear clipping.
    """
    hi = max(range(len(hole)), key=lambda i: (hole[i][0], hole[i][1]))
    h = hole[hi]
    rings = (poly, hole)
    best = None
    for vi, v in enumerate(poly):
        if v == h:
            continue
        clear = True
        for ring in rings:
            n = len(ring)
            for i in range(n):
                p = ring[i]
                q = ring[(i + 1) % n]
                if p == h or q == h or p == v or q == v:
                    continue
                if _seg_proper_cross(h, v, p, q):
                    clear = False
                    break
            if not clear:
                break
        if not clear:
            continue
        d2 = (v[0] - h[0]) ** 2 + (v[1] - h[1]) ** 2
        if best is None or d2 < best[0]:
            best = (d2, vi)
    if best is None:
        # Degenerate layout: bridge to the closest vertex regardless.
        vi = min(
            range(len(poly)),
            key=lambda i: (poly[i][0] - h[0]) ** 2 + (poly[i][1] - h[1]) ** 2,
        )
    else:
        vi = best[1]
    ring = hole[hi:] + hole[:hi]
    return poly[:vi + 1] + ring + [ring[0], poly[vi]] + poly[vi + 1:]


def _point_in_tri_strict(p, a, b, c):
    eps = 1e-12
    return (
        _orient(a, b, p) > eps
        and _orient(b, c, p) > eps
        and _orient(c, a, p) > eps
    )


def _ear_clip(poly):
    """Ear-clipping triangulation of a CCW (weakly) simple polygon.

    Returns a list of index triples into ``poly``.
    """
    n = len(poly)
    if n < 3:
        return []
    order = list(range(n))
    tris = []

    def _clip_pass(check_inside):
        m = len(order)
        for k in range(m):
            i_prev = order[k - 1]
            i_cur = order[k]
            i_next = order[(k + 1) % m]
            cr = _orient(poly[i_prev], poly[i_cur], poly[i_next])
            if cr <= 1e-12:
                continue
            if check_inside:
                a, b, c = poly[i_prev], poly[i_cur], poly[i_next]
                blocked = False
                for j in order:
                    if j == i_prev or j == i_cur or j == i_next:
                        continue
                    if _point_in_tri_strict(poly[j], a, b, c):
                        blocked = True
                        break
                if blocked:
                    continue
            tris.append((i_prev, i_cur, i_next))
            del order[k]
            return True
        return False

    while len(order) > 3:
        if _clip_pass(True):
            continue
        if _clip_pass(False):
            continue
        # Pathological remainder: fan from the first remaining vertex.
        for k in range(1, len(order) - 1):
            tris.append((order[0], order[k], order[k + 1]))
        order = order[:1]
        break
    if len(order) == 3:
        tris.append((order[0], order[1], order[2]))
    return tris


def _triangulate_shape(contour, holes):
    """Triangulate a CCW contour with CW holes.

    Returns ``(merged_points, triangles)`` where ``merged_points`` is
    the bridged vertex list the triangle indices refer to.
    """
    if _shape_signed_area(contour) < 0:
        contour = list(reversed(contour))
    merged = list(contour)
    for hole in holes:
        if len(hole) < 3:
            continue
        if _shape_signed_area(hole) > 0:
            hole = list(reversed(hole))
        merged = _bridge_hole(merged, hole)
    return merged, _ear_clip(merged)


def _shape_list(shapes):
    if shapes is None:
        return []
    if isinstance(shapes, (list, tuple)):
        return list(shapes)
    return [shapes]


def _extract_rings(shape, curve_segments):
    extracted = shape.extractPoints(curve_segments)
    contour = _shape_ring(extracted["shape"])
    holes = []
    for hole_pts in extracted["holes"]:
        ring = _shape_ring(hole_pts)
        if len(ring) >= 3:
            holes.append(ring)
    return contour, holes


class ShapeGeometry(BufferGeometry):
    """Flat triangulation of a Shape in the XY plane (three.js
    ``ShapeGeometry``): z = 0, normals +Z, uv = (x, y)."""

    def __init__(self, shapes=None, curveSegments=12):
        super().__init__()
        positions = []
        uvs = []
        indices = []
        for shape in _shape_list(shapes):
            contour, holes = _extract_rings(shape, curveSegments)
            if len(contour) < 3:
                continue
            merged, tris = _triangulate_shape(contour, holes)
            base = len(positions)
            for (x, y) in merged:
                positions.append([x, y, 0.0])
                uvs.append([x, y])
            for (a, b, c) in tris:
                indices += [base + a, base + b, base + c]
        count = len(positions)
        self.attributes["position"] = np.array(
            positions, dtype=np.float32
        ).reshape(-1, 3)
        self.attributes["normal"] = np.tile(
            np.array([0.0, 0.0, 1.0], dtype=np.float32), (count, 1)
        )
        self.attributes["uv"] = np.array(uvs, dtype=np.float32).reshape(-1, 2)
        self.index = np.array(indices, dtype=np.int64)


class ExtrudeGeometry(BufferGeometry):
    """A Shape extruded along +Z (three.js ``ExtrudeGeometry``).

    Supports contour + holes and ``depth`` (legacy ``amount``); bevel
    is not implemented.  Built non-indexed with flat face
    normals so the signed volume is exactly area x depth for a prism.
    """

    def __init__(self, shapes=None, options=None):
        super().__init__()
        opts = options if isinstance(options, dict) else {}
        depth = float(opts.get("depth", opts.get("amount", 1.0)))
        curve_segments = int(opts.get("curveSegments", 12))
        positions = []
        uvs = []

        def _push(p, uv):
            positions.append([p[0], p[1], p[2]])
            uvs.append([uv[0], uv[1]])

        for shape in _shape_list(shapes):
            contour, holes = _extract_rings(shape, curve_segments)
            if len(contour) < 3:
                continue
            if _shape_signed_area(contour) < 0:
                contour = list(reversed(contour))
            holes = [
                list(reversed(h)) if _shape_signed_area(h) > 0 else h
                for h in holes
            ]
            merged, tris = _triangulate_shape(contour, holes)
            # Back cap at z = 0 (winding flipped -> normal -Z).
            for (a, b, c) in tris:
                for idx in (a, c, b):
                    x, y = merged[idx]
                    _push((x, y, 0.0), (x, y))
            # Front cap at z = depth (normal +Z).
            for (a, b, c) in tris:
                for idx in (a, b, c):
                    x, y = merged[idx]
                    _push((x, y, depth), (x, y))
            # Side walls for the contour and each hole ring.  The same
            # construction serves both: outer rings are CCW, hole
            # rings are CW, and the resulting normals face away from
            # the material in both cases.
            for ring in [contour] + holes:
                m = len(ring)
                dist = 0.0
                for i in range(m):
                    ax, ay = ring[i]
                    bx, by = ring[(i + 1) % m]
                    seg_len = math.hypot(bx - ax, by - ay)
                    a0 = (ax, ay, 0.0)
                    b0 = (bx, by, 0.0)
                    b1 = (bx, by, depth)
                    a1 = (ax, ay, depth)
                    _push(a0, (dist, 0.0))
                    _push(b0, (dist + seg_len, 0.0))
                    _push(b1, (dist + seg_len, depth))
                    _push(a0, (dist, 0.0))
                    _push(b1, (dist + seg_len, depth))
                    _push(a1, (dist, depth))
                    dist += seg_len
        count = len(positions)
        self.attributes["position"] = np.array(
            positions, dtype=np.float32
        ).reshape(-1, 3)
        self.attributes["uv"] = np.array(uvs, dtype=np.float32).reshape(-1, 2)
        self.index = None
        if count:
            self.computeVertexNormals()


# ---------------------------------------------------------------------------
# Curves in 3D and TubeGeometry
# ---------------------------------------------------------------------------

class Curve:
    """Minimal three.js ``Curve`` base: subclasses define getPoint.
    The arc-length API (getLengths / getPointAt / getUtoTmapping)
    follows the original's sampled-table algorithm."""

    isCurve = True
    arcLengthDivisions = 200

    def getPoint(self, t, optionalTarget=None):
        raise NotImplementedError("Curve.getPoint is abstract")

    def getPoints(self, divisions=50):
        divisions = max(1, int(divisions))
        return [self.getPoint(i / divisions) for i in range(divisions + 1)]

    def getSpacedPoints(self, divisions=5):
        divisions = max(1, int(divisions))
        if hasattr(self, "getPointAt"):
            return [self.getPointAt(i / divisions) for i in range(divisions + 1)]
        return self.getPoints(divisions)

    def getLengths(self, divisions=None):
        if divisions is None:
            divisions = self.arcLengthDivisions
        divisions = max(1, int(divisions))
        lengths = [0.0]
        current = self.getPoint(0.0)
        total = 0.0
        for i in range(1, divisions + 1):
            point = self.getPoint(i / divisions)
            total += point.distanceTo(current)
            lengths.append(total)
            current = point
        self.cacheArcLengths = lengths
        return lengths

    def getLength(self):
        return self.getLengths()[-1]

    def getUtoTmapping(self, u, distance=None):
        lengths = self.getLengths()
        total = lengths[-1]
        target = (float(distance) if distance is not None
                  else float(u) * total)
        divisions = len(lengths) - 1
        if target <= 0.0:
            return 0.0
        if target >= total:
            return 1.0
        idx = 1
        while idx < divisions and lengths[idx] < target:
            idx += 1
        before = lengths[idx - 1]
        segment = lengths[idx] - before
        fraction = (0.0 if segment <= 0.0
                    else (target - before) / segment)
        return (idx - 1 + fraction) / divisions

    def getPointAt(self, u, optionalTarget=None):
        t = self.getUtoTmapping(u)
        return self.getPoint(t, optionalTarget)

    def getSpacedPoints(self, divisions=50):
        divisions = max(1, int(divisions))
        return [self.getPointAt(i / divisions)
                for i in range(divisions + 1)]

    def getTangent(self, t, optionalTarget=None):
        delta = 0.0001
        t1 = max(0.0, float(t) - delta)
        t2 = min(1.0, float(t) + delta)
        p1 = self.getPoint(t1)
        p2 = self.getPoint(t2)
        if optionalTarget is not None:
            tangent = optionalTarget
            tangent.copy(p2)
        else:
            tangent = p2.clone()
        tangent.sub(p1)
        if tangent.lengthSq() > 0.0:
            tangent.normalize()
        return tangent

    def getTangentAt(self, u, optionalTarget=None):
        return self.getTangent(self.getUtoTmapping(u), optionalTarget)


class SplineCurve(_SplineCurve, Curve):
    """three.js ``SplineCurve``: the 2D Catmull-Rom spline
    (_SplineCurve) under its public name, with the Curve arc-length
    API (getPointAt & friends, inherited from Curve)."""

    isSplineCurve = True

    def getPointAt(self, u, optionalTarget=None):
        point = self.getPoint(self.getUtoTmapping(u))
        if optionalTarget is not None:
            optionalTarget.set(point.x, point.y)
            return optionalTarget
        return point


def _catmull_rom_knots(p0, p1, p2, p3, alpha):
    def _pow_dist(a, b):
        d = math.sqrt(
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        )
        return d ** alpha

    t0 = 0.0
    t1 = t0 + _pow_dist(p0, p1)
    t2 = t1 + _pow_dist(p1, p2)
    t3 = t2 + _pow_dist(p2, p3)
    if t1 <= t0:
        t1 = t0 + 1e-4
    if t2 <= t1:
        t2 = t1 + 1e-4
    if t3 <= t2:
        t3 = t2 + 1e-4
    return t0, t1, t2, t3


class CatmullRomCurve3(Curve):
    """Catmull-Rom spline through Vector3 points (three.js
    ``CatmullRomCurve3``).  ``curveType`` is 'centripetal' (default),
    'chordal' or 'catmullrom' (uniform, using ``tension``)."""

    isCatmullRomCurve3 = True

    def __init__(self, points=None, closed=False, curveType="centripetal",
                 tension=0.5):
        self.points = []
        for p in (points if points is not None else []):
            if isinstance(p, Vector3):
                self.points.append(p)
            else:
                seq = list(p)
                self.points.append(
                    Vector3(float(seq[0]), float(seq[1]), float(seq[2]))
                )
        self.closed = bool(closed)
        self.curveType = str(curveType)
        self.tension = float(tension)

    def getPoint(self, t, optionalTarget=None):
        pts = self.points
        result = optionalTarget if optionalTarget is not None else Vector3()
        n = len(pts)
        if n == 0:
            return result.set(0.0, 0.0, 0.0)
        if n == 1:
            return result.copy(pts[0])
        tuples = [(p.x, p.y, p.z) for p in pts]
        span = n if self.closed else n - 1
        p = span * float(t)
        int_point = int(math.floor(p))
        weight = p - int_point
        if self.closed:
            int_point %= n
            i0 = (int_point - 1) % n
            i1 = int_point
            i2 = (int_point + 1) % n
            i3 = (int_point + 2) % n
        else:
            if int_point > n - 2:
                int_point = n - 2
                weight = float(t) * span - int_point
            i1 = int_point
            i2 = int_point + 1
            if int_point > 0:
                i0 = int_point - 1
                p0 = tuples[i0]
            else:
                a = tuples[0]
                b = tuples[1]
                p0 = (2 * a[0] - b[0], 2 * a[1] - b[1], 2 * a[2] - b[2])
            if int_point + 2 < n:
                i3 = int_point + 2
                p3 = tuples[i3]
            else:
                a = tuples[n - 1]
                b = tuples[n - 2]
                p3 = (2 * a[0] - b[0], 2 * a[1] - b[1], 2 * a[2] - b[2])
        p1 = tuples[i1]
        p2 = tuples[i2]
        if self.closed:
            p0 = tuples[i0]
            p3 = tuples[i3]

        if self.curveType in ("centripetal", "chordal"):
            alpha = 1.0 if self.curveType == "chordal" else 0.5
            t0, t1, t2, t3 = _catmull_rom_knots(p0, p1, p2, p3, alpha)
            tt = t1 + weight * (t2 - t1)
            out = []
            for k in range(3):
                a1 = (t1 - tt) / (t1 - t0) * p0[k] + (tt - t0) / (t1 - t0) * p1[k]
                a2 = (t2 - tt) / (t2 - t1) * p1[k] + (tt - t1) / (t2 - t1) * p2[k]
                a3 = (t3 - tt) / (t3 - t2) * p2[k] + (tt - t2) / (t3 - t2) * p3[k]
                b1 = (t2 - tt) / (t2 - t0) * a1 + (tt - t0) / (t2 - t0) * a2
                b2 = (t3 - tt) / (t3 - t1) * a2 + (tt - t1) / (t3 - t1) * a3
                out.append((t2 - tt) / (t2 - t1) * b1 + (tt - t1) / (t2 - t1) * b2)
            return result.set(out[0], out[1], out[2])

        # Uniform Catmull-Rom (three.js CubicPoly form): value =
        # c0 + t0*w + c2*w^2 + c3*w^3 with c0 = p1 and the tangents
        # t0 = tension*(p2-p0), t1 = tension*(p3-p1).
        tension = self.tension
        w = weight
        w2 = w * w
        w3 = w2 * w
        out = []
        for k in range(3):
            t0v = tension * (p2[k] - p0[k])
            t1v = tension * (p3[k] - p1[k])
            cc2 = -3.0 * p1[k] + 3.0 * p2[k] - 2.0 * t0v - t1v
            cc3 = 2.0 * p1[k] - 2.0 * p2[k] + t0v + t1v
            out.append(p1[k] + t0v * w + cc2 * w2 + cc3 * w3)
        return result.set(out[0], out[1], out[2])


class TubeGeometry(BufferGeometry):
    """A tube swept along a 3D curve (three.js ``TubeGeometry``).

    Frames are computed by parallel transport of an initial normal
    (the real implementation minimises twist by rotating between
    successive tangents; the projection used here is equivalent for
    the smooth curves this port targets).
    """

    def __init__(self, path=None, tubularSegments=64, radius=1.0,
                 radialSegments=8, closed=False):
        super().__init__()
        tubular = max(1, int(tubularSegments))
        radial = max(1, int(radialSegments))
        radius = float(radius)
        if path is None:
            path = CatmullRomCurve3(
                [Vector3(-1.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0)]
            )
        closed = bool(closed) or bool(getattr(path, "closed", False))

        frames_p = [path.getPoint(i / tubular) for i in range(tubular + 1)]
        if closed:
            frames_p[tubular] = frames_p[0].clone()

        def _tangent(i):
            if closed:
                a = frames_p[(i - 1) % tubular]
                b = frames_p[(i + 1) % tubular]
            elif i == 0:
                a = frames_p[0]
                b = frames_p[1]
            elif i == tubular:
                a = frames_p[tubular - 1]
                b = frames_p[tubular]
            else:
                a = frames_p[i - 1]
                b = frames_p[i + 1]
            t = b.clone().sub(a)
            if t.lengthSq() == 0.0:
                return Vector3(0.0, 0.0, 1.0)
            return t.normalize()

        tangents = [_tangent(i) for i in range(tubular + 1)]
        if closed:
            tangents[tubular] = tangents[0].clone()

        # Initial normal: perpendicular to the first tangent, chosen
        # via the tangent's smallest component axis (three.js method).
        t0 = tangents[0]
        axis = Vector3(1.0, 0.0, 0.0)
        smallest = abs(t0.x)
        if abs(t0.y) <= smallest:
            smallest = abs(t0.y)
            axis = Vector3(0.0, 1.0, 0.0)
        if abs(t0.z) <= smallest:
            axis = Vector3(0.0, 0.0, 1.0)
        vec = Vector3().crossVectors(t0, axis).normalize()
        normal = Vector3().crossVectors(t0, vec).normalize()

        normals = []
        for i in range(tubular + 1):
            t = tangents[i]
            n = normal.clone()
            d = n.dot(t)
            n = Vector3(n.x - t.x * d, n.y - t.y * d, n.z - t.z * d)
            if n.lengthSq() < 1e-12:
                # Tangent swung parallel to the normal: rebuild one.
                n = Vector3(1.0, 0.0, 0.0)
                if abs(t.x) > 0.9:
                    n = Vector3(0.0, 1.0, 0.0)
                d = n.dot(t)
                n = Vector3(n.x - t.x * d, n.y - t.y * d, n.z - t.z * d)
            n.normalize()
            normals.append(n)
            normal = n

        positions = []
        out_normals = []
        uvs = []
        for i in range(tubular + 1):
            center = frames_p[i]
            n = normals[i]
            b = Vector3().crossVectors(tangents[i], n).normalize()
            for j in range(radial + 1):
                v = j / radial * 2.0 * math.pi
                cx = -math.cos(v)
                sx = math.sin(v)
                nx = cx * n.x + sx * b.x
                ny = cx * n.y + sx * b.y
                nz = cx * n.z + sx * b.z
                positions.append([
                    center.x + radius * nx,
                    center.y + radius * ny,
                    center.z + radius * nz,
                ])
                out_normals.append([nx, ny, nz])
                uvs.append([i / tubular, j / radial])
        indices = []
        stride = radial + 1
        for i in range(tubular):
            for j in range(radial):
                a = i * stride + j
                b = (i + 1) * stride + j
                c = (i + 1) * stride + j + 1
                d = i * stride + j + 1
                indices += [a, b, d, b, c, d]
        self.attributes["position"] = np.array(positions, dtype=np.float32)
        self.attributes["normal"] = np.array(out_normals, dtype=np.float32)
        self.attributes["uv"] = np.array(uvs, dtype=np.float32)
        self.index = np.array(indices, dtype=np.int64)


__all__ += [
    "OBJLoader",
    "Path",
    "Shape",
    "ShapeGeometry",
    "ExtrudeGeometry",
    "Curve",
    "CatmullRomCurve3",
    "TubeGeometry",
]


# ---------------------------------------------------------------------------
# MeshStandardMaterial approximation and environment maps
# ---------------------------------------------------------------------------

class MeshStandardMaterial(Material):
    """three.js MeshStandardMaterial, approximated.

    Real PBR (GGX / Smith / energy conservation) is out of scope; the
    renderer interprets the two headline parameters inside the existing
    Lambert-diffuse + Blinn-Phong-specular framework:

    - ``roughness`` drives the specular exponent (0 = tight mirror-like
      highlight, 1 = broad and faint) and damps the specular strength.
    - ``metalness`` suppresses the diffuse term and tints the specular
      reflection with the base colour (F0 = mix(0.04 grey, color,
      metalness), the standard dielectric/conductor blend).

    ``flatShading`` is accepted and stored; this port's renderer always
    shades flat per face, so the flag is naturally satisfied.
    ``envMap`` (a CubeTexture) overrides ``scene.environment`` for this
    material, and ``envMapIntensity`` scales the reflection.
    """

    isMeshStandardMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        self.emissive = _as_color(p.get("emissive"), Color(0.0, 0.0, 0.0))
        self.emissiveIntensity = float(p.get("emissiveIntensity", 1.0))
        self.roughness = float(p.get("roughness", 1.0))
        self.metalness = float(p.get("metalness", 0.0))
        self.flatShading = bool(p.get("flatShading", False))
        self.envMap = p.get("envMap", None)
        self.envMapIntensity = float(p.get("envMapIntensity", 1.0))


def _coerce_cube_face(face, default_rgb):
    """Normalise one cube face to an (h, w, 3) float64 array in 0..1.

    A face may be a hex int, a Color, or an image array (uint8 0..255 or
    float 0..1); a solid colour becomes a 1x1 image.  ``None`` (or
    anything unusable) falls back to ``default_rgb``."""
    if face is None:
        return np.array([[default_rgb]], dtype=np.float64)
    if isinstance(face, (int, np.integer)):
        n = int(face)
        rgb = (
            ((n >> 16) & 255) / 255.0,
            ((n >> 8) & 255) / 255.0,
            (n & 255) / 255.0,
        )
        return np.array([[rgb]], dtype=np.float64)
    r = getattr(face, "r", None)
    if r is not None:
        try:
            rgb = (float(r), float(face.g), float(face.b))
            return np.array([[rgb]], dtype=np.float64)
        except Exception:
            pass
    arr = _coerce_texture_image(face)
    if arr is None:
        return np.array([[default_rgb]], dtype=np.float64)
    arr = arr.astype(np.float64)
    if arr.max() > 1.0:  # 0..255 style
        arr = arr / 255.0
    return np.ascontiguousarray(arr)


class CubeTexture:
    """A minimal cube map: six faces in three.js order
    [+x, -x, +y, -y, +z, -z], each a solid colour or a small image.

    This is the port's own representation, built from an array of six
    face colours/images (procedural; no PMREM, no image-file loading).
    Assign to ``scene.environment`` (or a material's ``envMap``) and the
    renderer mixes the reflection into Standard/Phong specular.

    With no faces given, a procedural sky is used: bright blue top,
    pale horizon sides, dark ground bottom.
    """

    isCubeTexture = True

    # (px, nx, py, ny, pz, nz) defaults: a simple sky/ground split.
    _DEFAULT_FACES = (
        (0.62, 0.72, 0.85),  # +x horizon
        (0.62, 0.72, 0.85),  # -x horizon
        (0.35, 0.55, 0.95),  # +y sky
        (0.30, 0.27, 0.24),  # -y ground
        (0.62, 0.72, 0.85),  # +z horizon
        (0.62, 0.72, 0.85),  # -z horizon
    )

    def __init__(self, faces=None, name=""):
        self.name = str(name)
        if faces is None:
            seq = [None] * 6
        elif isinstance(faces, (list, tuple)) and len(faces) == 6:
            seq = list(faces)
        else:
            # a single colour/image broadcasts to all six faces
            seq = [faces] * 6
        self.faces = [
            _coerce_cube_face(face, self._DEFAULT_FACES[i])
            for i, face in enumerate(seq)
        ]
        self.needsUpdate = True

    def sample_rgb(self, direction):
        """Sample the cube map along ``direction`` ((3,) or (N, 3)).

        Returns float RGB in 0..1 with the same leading shape.  The
        face is chosen by the major axis (OpenGL cube-map convention)
        and sampled nearest; image rows are read with row 0 as the top
        of the face."""
        d = np.asarray(direction, dtype=np.float64)
        single = d.ndim == 1
        D = d.reshape(1, 3) if single else d.reshape(-1, 3)
        out = np.zeros((len(D), 3), dtype=np.float64)
        if len(D) == 0:
            return out[0] if single else out
        ax = np.argmax(np.abs(D), axis=1)
        for axis in range(3):
            sel = ax == axis
            if not np.any(sel):
                continue
            comp = D[sel, axis]
            pos = comp >= 0.0
            # three.js face order per axis: (+, -) -> (2*axis, 2*axis+1)
            for sign, face_i in ((True, 2 * axis), (False, 2 * axis + 1)):
                idx = np.nonzero(sel)[0][pos if sign else ~pos]
                if len(idx) == 0:
                    continue
                img = self.faces[face_i]
                h, w = img.shape[:2]
                if h == 1 and w == 1:
                    out[idx] = img[0, 0]
                    continue
                sub = D[idx]
                ma = np.abs(sub[:, axis])
                ma = np.where(ma < 1e-12, 1.0, ma)
                others = [c for c in range(3) if c != axis]
                # OpenGL cube-map (sc, tc) per face.
                if face_i == 0:    # +x: sc=-z, tc=-y
                    sc, tc = -sub[:, 2], -sub[:, 1]
                elif face_i == 1:  # -x: sc=+z, tc=-y
                    sc, tc = sub[:, 2], -sub[:, 1]
                elif face_i == 2:  # +y: sc=+x, tc=+z
                    sc, tc = sub[:, 0], sub[:, 2]
                elif face_i == 3:  # -y: sc=+x, tc=-z
                    sc, tc = sub[:, 0], -sub[:, 2]
                elif face_i == 4:  # +z: sc=+x, tc=-y
                    sc, tc = sub[:, 0], -sub[:, 1]
                else:              # -z: sc=-x, tc=-y
                    sc, tc = -sub[:, 0], -sub[:, 1]
                u = 0.5 * (sc / ma + 1.0)
                v = 0.5 * (tc / ma + 1.0)
                cols = np.clip((u * w).astype(np.int64), 0, w - 1)
                rows_i = np.clip(((1.0 - v) * h).astype(np.int64), 0, h - 1)
                out[idx] = img[rows_i, cols]
        return out[0] if single else out

    def clone(self):
        tex = CubeTexture.__new__(CubeTexture)
        tex.name = self.name
        tex.faces = list(self.faces)
        tex.needsUpdate = True
        return tex

    def __repr__(self):
        return "CubeTexture(%r, 6 faces)" % (self.name,)


__all__ += [
    "MeshStandardMaterial",
    "CubeTexture",
]


# ---------------------------------------------------------------------------
# Addon receptacles: SyncPromise, Stats / lil-gui stubs,
# BufferGeometryUtils.  These live on the three module so jsinterp's
# addon registry can bind them for 'three/addons/libs/stats.module.js',
# 'lil-gui' and 'three/addons/utils/BufferGeometryUtils.js' imports.
# They are deliberately NOT added to __all__: they are not core
# 'three' exports.
# ---------------------------------------------------------------------------

class SyncPromise:
    """Minimal promise-like object that is already settled.

    The port's loaders work synchronously (their input bytes are
    registered on device before any JS runs), so ``loadAsync``
    returns one of these: ``then`` / ``catch`` invoke
    their callbacks *synchronously*, before returning.  That covers
    the common ``loader.loadAsync(url).then(...)`` shape, but code
    that depends on the real asynchronous ordering (other work
    happening between the call and the resolution, ``await``
    interleaving) is out of scope, as is ``async``/``await`` syntax
    itself.
    """

    def __init__(self, state, value=None):
        self._state = state        # "fulfilled" or "rejected"
        self._value = value

    @classmethod
    def resolved(cls, value=None):
        if isinstance(value, SyncPromise):
            return value
        return cls("fulfilled", value)

    @classmethod
    def rejected(cls, error):
        return cls("rejected", error)

    def then(self, on_fulfilled=None, on_rejected=None):
        if self._state == "fulfilled":
            if not callable(on_fulfilled):
                return SyncPromise("fulfilled", self._value)
            try:
                return SyncPromise.resolved(on_fulfilled(self._value))
            except Exception as exc:
                return SyncPromise("rejected", exc)
        # rejected
        if not callable(on_rejected):
            return SyncPromise("rejected", self._value)
        try:
            return SyncPromise.resolved(on_rejected(self._value))
        except Exception as exc:
            return SyncPromise("rejected", exc)

    def catch_(self, on_rejected=None):
        return self.then(None, on_rejected)

    def finally_(self, callback=None):
        if callable(callback):
            try:
                result = callback()
            except Exception as exc:
                return SyncPromise("rejected", exc)
            if isinstance(result, SyncPromise) and \
                    result._state == "rejected":
                return result
        return SyncPromise(self._state, self._value)

    def __repr__(self):
        return "SyncPromise(%s, %r)" % (self._state, self._value)


# JS spells two Promise methods with Python keywords / conventions
# that cannot be written with ``def``; bind them as attributes so
# getattr-based member access from jsinterp finds them.
SyncPromise.catch = SyncPromise.catch_
setattr(SyncPromise, "finally", SyncPromise.finally_)


class _StubElement:
    """Smallest DOM-element stand-in for stub ``.dom`` properties:
    attribute bag plus the child/style surface examples touch."""

    def __init__(self, tag="div"):
        self.tagName = str(tag).upper()
        self.style = {}
        self.children = []
        self.parentNode = None
        self.textContent = ""

    def appendChild(self, child):
        self.children.append(child)
        return child

    def removeChild(self, child):
        if child in self.children:
            self.children.remove(child)
        return child

    def setAttribute(self, *args):
        pass

    def getAttribute(self, *args):
        return None

    def addEventListener(self, *args):
        pass

    def removeEventListener(self, *args):
        pass

    def play(self):
        return None

    def pause(self):
        return None


class _StatsPanel:
    """stats.js custom panel stub (``stats.addPanel``)."""

    def __init__(self, name="", foreground="", background=""):
        self.name = name
        self.dom = _StubElement("canvas")

    def update(self, *args):
        pass


class Stats:
    """Stub for the addons ``stats.module.js``.

    Examples import it in ~140 of the official webgl examples just
    to show an FPS meter.  There is no display surface in this port,
    so the class accepts the whole usual call surface -- ``dom`` /
    ``domElement`` (a dummy element), ``begin`` / ``end`` /
    ``update`` / ``showPanel`` / ``addPanel`` / ``setMode`` -- and
    silently does nothing.  ``end`` returns a millisecond timestamp
    like the original (a monotonic clock reading)."""

    REVISION = 16

    def __init__(self):
        self.dom = _StubElement("div")
        self.domElement = self.dom
        self._panel = 0
        self._panels = []

    def begin(self):
        pass

    def end(self):
        return time.perf_counter() * 1000.0

    def update(self):
        pass

    def showPanel(self, panel_id):
        self._panel = panel_id

    def setMode(self, mode):
        self._panel = mode

    def addPanel(self, panel):
        self._panels.append(panel)
        return panel


def _gui_read(obj, prop):
    if isinstance(obj, dict):
        return obj.get(prop)
    return getattr(obj, prop, None)


def _gui_write(obj, prop, value):
    if isinstance(obj, dict):
        obj[prop] = value
        return True
    try:
        setattr(obj, prop, value)
        return True
    except Exception:
        return False


class GUIController:
    """lil-gui controller stub: holds the value and the registered
    callbacks, and accepts the full chaining surface examples use
    (``name`` / ``min`` / ``max`` / ``step`` / ``options`` /
    ``onChange`` / ``onFinishChange`` / ``listen`` /
    ``updateDisplay`` ...).  ``setValue`` writes through to the
    target object (best effort: setattr for Python objects, item
    assignment for dicts) and fires the registered ``onChange``
    callbacks, like the real GUI does when a value is committed."""

    def __init__(self, parent, obj, prop, value):
        self._parent = parent
        self.object = obj
        self.property = prop
        self.initialValue = value
        self._value = value
        self._on_change = []
        self._on_finish_change = []
        self._name = prop
        self._min = None
        self._max = None
        self._step = None
        self._options = None
        self._listening = False
        self._destroyed = False
        self.domElement = _StubElement("div")

    # -- value ------------------------------------------------------
    def getValue(self):
        return self._value

    def setValue(self, value):
        self._value = value
        _gui_write(self.object, self.property, value)
        for callback in list(self._on_change):
            callback(value)
        return self

    def reset(self):
        return self.setValue(self.initialValue)

    # -- chaining configuration (accepted, recorded, no display) ----
    def name(self, name=None):
        if name is not None:
            self._name = name
        return self

    def min(self, value):
        self._min = value
        return self

    def max(self, value):
        self._max = value
        return self

    def step(self, value):
        self._step = value
        return self

    def options(self, options):
        self._options = options
        return self

    def onChange(self, callback):
        if callable(callback):
            self._on_change.append(callback)
        return self

    def onFinishChange(self, callback):
        if callable(callback):
            self._on_finish_change.append(callback)
        return self

    def listen(self, *args):
        self._listening = True
        return self

    def updateDisplay(self):
        value = _gui_read(self.object, self.property)
        if value is not None:
            self._value = value
        return self

    def show(self, *args):
        return self

    def hide(self, *args):
        return self

    def enable(self, *args):
        return self

    def disable(self, *args):
        return self

    def destroy(self):
        self._destroyed = True
        parent = self._parent
        if parent is not None and self in parent.controllers:
            parent.controllers.remove(self)

    def __getattr__(self, name):
        # Absorb any less-common controller method as a chainable
        # no-op rather than failing the whole example at the GUI
        # call site.  (Only reached for attributes not defined
        # above.)
        if name.startswith("_"):
            raise AttributeError(name)

        def _noop(*args, **kwargs):
            return self

        return _noop


class GUI:
    """Stub for the ``lil-gui`` package's default export.

    ~133 official examples build a parameter panel with it.  There
    is no panel to show in this port, so ``add`` / ``addColor`` /
    ``addFolder`` build real controller objects that retain their
    values and callbacks (see GUIController) while every display
    operation is a no-op.  Folders are GUI instances, as in lil-gui.
    """

    def __init__(self, options=None, _parent=None):
        self._parent = _parent
        self.controllers = []
        self.folders = []
        self.children = []
        self.title = None
        self.domElement = _StubElement("div")
        self._destroyed = False
        if isinstance(options, dict):
            if options.get("title") is not None:
                self.title = options["title"]
            if options.get("parent") is not None:
                self._parent = options["parent"]

    # -- controllers --------------------------------------------------
    def add(self, obj, prop, *args):
        value = _gui_read(obj, prop)
        controller = GUIController(self, obj, prop, value)
        for arg in args:
            if isinstance(arg, (list, tuple, dict)):
                controller._options = arg
        nums = [a for a in args if isinstance(a, (int, float))
                and not isinstance(a, bool)]
        if len(nums) >= 1:
            controller._min = nums[0]
        if len(nums) >= 2:
            controller._max = nums[1]
        if len(nums) >= 3:
            controller._step = nums[2]
        self.controllers.append(controller)
        self.children.append(controller)
        return controller

    def addColor(self, obj, prop, *args):
        controller = self.add(obj, prop)
        controller._is_color = True
        return controller

    def addFolder(self, title):
        folder = GUI(_parent=self)
        folder.title = title
        self.folders.append(folder)
        self.children.append(folder)
        return folder

    # -- panel-level surface ------------------------------------------
    def open(self, *args):
        return self

    def close(self):
        return self

    def openAnimated(self, *args):
        return self

    def show(self, *args):
        return self

    def hide(self, *args):
        return self

    def onChange(self, callback):
        # lil-gui >= 0.19 GUI-level change hook: registration only.
        if callable(callback):
            for controller in self.controllers:
                controller._on_change.append(callback)
        return self

    def controllersRecursive(self):
        out = list(self.controllers)
        for folder in self.folders:
            out.extend(folder.controllersRecursive())
        return out

    def reset(self, *args):
        for controller in self.controllersRecursive():
            controller.reset()
        return self

    def destroy(self):
        self._destroyed = True
        for folder in list(self.folders):
            folder.destroy()

    def __getattr__(self, name):
        # Same absorb-unknown-calls policy as GUIController.
        if name.startswith("_"):
            raise AttributeError(name)

        def _noop(*args, **kwargs):
            return self

        return _noop


# ---------------------------------------------------------------------------
# BufferGeometryUtils -- real implementations of the main
# functions, following the original's merge semantics: all inputs
# must agree on index-ness and on the attribute name set, attribute
# item sizes must match, and any mismatch returns None (the original
# logs an error and returns null) instead of producing a corrupt
# geometry.  Attributes here are the port's numpy arrays.
# ---------------------------------------------------------------------------

def _attribute_item_size(arr):
    arr = np.asarray(arr)
    if arr.ndim >= 2:
        return int(arr.shape[1])
    return 1


def mergeAttributes(attributes):
    """Concatenate a list of same-item-size attribute arrays into
    one; returns None when the item sizes disagree (the original
    ``mergeAttributes`` returns null in that case)."""
    if not attributes:
        return None
    arrays = [np.asarray(a) for a in attributes]
    item_size = _attribute_item_size(arrays[0])
    for arr in arrays[1:]:
        if _attribute_item_size(arr) != item_size:
            return None
    return np.concatenate(arrays, axis=0)


def mergeGeometries(geometries, useGroups=False):
    """Merge BufferGeometries into one (original:
    ``BufferGeometryUtils.mergeGeometries``).

    All geometries must be indexed or all non-indexed, must carry
    exactly the same attribute names, and matching attributes must
    share an item size; otherwise None is returned.  With
    ``useGroups`` the merged geometry gains a ``groups`` list of
    ``{start, count, materialIndex}`` dicts, one per input.
    """
    geometries = list(geometries)
    if not geometries:
        return None
    first = geometries[0]
    is_indexed = first.index is not None
    attribute_names = set(first.attributes.keys())
    for geometry in geometries[1:]:
        if (geometry.index is not None) != is_indexed:
            return None
        if set(geometry.attributes.keys()) != attribute_names:
            return None
    merged = BufferGeometry()
    for name in first.attributes.keys():
        arrays = [g.attributes[name] for g in geometries]
        merged_attr = mergeAttributes(arrays)
        if merged_attr is None:
            return None
        merged.attributes[name] = merged_attr
    if is_indexed:
        index_parts = []
        offset = 0
        for geometry in geometries:
            index_parts.append(
                np.asarray(geometry.index, dtype=np.int64) + offset)
            offset += geometry._vertex_count()
        merged.setIndex(np.concatenate(index_parts, axis=0))
    if useGroups:
        groups = []
        start = 0
        for i, geometry in enumerate(geometries):
            if is_indexed:
                count = int(np.asarray(geometry.index).size)
            else:
                count = geometry._vertex_count()
            groups.append({
                "start": start,
                "count": count,
                "materialIndex": i,
            })
            start += count
        merged.groups = groups
    for geometry in geometries:
        merged.userData.update(geometry.userData)
    return merged


# r132-era name for the same merge (the manual's bundled three
# predates the rename to mergeGeometries).
mergeBufferGeometries = mergeGeometries


def mergeVertices(geometry, tolerance=1e-4):
    """Return an indexed copy of ``geometry`` with duplicate
    vertices folded together (original:
    ``BufferGeometryUtils.mergeVertices``).  Two vertices are the
    same when every attribute value agrees within ``tolerance``.
    """
    if not geometry.attributes:
        return geometry
    vertex_count = geometry._vertex_count()
    if vertex_count == 0:
        out = BufferGeometry()
        out.setIndex(np.zeros(0, dtype=np.int64))
        return out
    if geometry.index is not None:
        order = np.asarray(geometry.index, dtype=np.int64).tolist()
    else:
        order = list(range(vertex_count))
    shift = 10.0 ** int(round(-np.log10(tolerance))) if tolerance > 0 else 1.0
    arrays = {}
    for name, arr in geometry.attributes.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays[name] = arr
    hash_to_new = {}
    new_rows = {name: [] for name in arrays}
    new_index = []
    for vertex in order:
        key_parts = []
        for name in sorted(arrays.keys()):
            row = arrays[name][vertex]
            key_parts.append(
                tuple(np.round(row.astype(np.float64) * shift)
                      .astype(np.int64).tolist()))
        key = tuple(key_parts)
        new_pos = hash_to_new.get(key)
        if new_pos is None:
            new_pos = len(hash_to_new)
            hash_to_new[key] = new_pos
            for name in arrays:
                new_rows[name].append(arrays[name][vertex])
        new_index.append(new_pos)
    out = BufferGeometry()
    for name, rows in new_rows.items():
        arr = np.stack(rows, axis=0)
        original = np.asarray(geometry.attributes[name])
        if original.ndim == 1:
            arr = arr.reshape(-1)
        out.attributes[name] = arr
    out.setIndex(np.asarray(new_index, dtype=np.int64))
    out.userData.update(geometry.userData)
    return out


# ---------------------------------------------------------------------------
# Foundation fill-ins -- the 200 upstream constants the port was
# missing (values copied verbatim from the local three.js clone's
# src/constants.js; note this clone's REVISION reads '188dev', a
# post-r187 development snapshot), plus Timer, OrthographicCamera,
# TorusKnotGeometry and CanvasTexture with a minimal Canvas 2D.
# ---------------------------------------------------------------------------
REVISION = '188dev'
MOUSE = { 'LEFT': 0, 'MIDDLE': 1, 'RIGHT': 2, 'ROTATE': 0, 'DOLLY': 1, 'PAN': 2 }
TOUCH = { 'ROTATE': 0, 'PAN': 1, 'DOLLY_PAN': 2, 'DOLLY_ROTATE': 3 }
CullFaceNone = 0
CullFaceBack = 1
CullFaceFront = 2
CullFaceFrontBack = 3
BasicShadowMap = 0
PCFShadowMap = 1
PCFSoftShadowMap = 2
VSMShadowMap = 3
NoBlending = 0
NormalBlending = 1
AdditiveBlending = 2
SubtractiveBlending = 3
MultiplyBlending = 4
CustomBlending = 5
MaterialBlending = 6
AddEquation = 100
SubtractEquation = 101
ReverseSubtractEquation = 102
MinEquation = 103
MaxEquation = 104
ZeroFactor = 200
OneFactor = 201
SrcColorFactor = 202
OneMinusSrcColorFactor = 203
SrcAlphaFactor = 204
OneMinusSrcAlphaFactor = 205
DstAlphaFactor = 206
OneMinusDstAlphaFactor = 207
DstColorFactor = 208
OneMinusDstColorFactor = 209
SrcAlphaSaturateFactor = 210
ConstantColorFactor = 211
OneMinusConstantColorFactor = 212
ConstantAlphaFactor = 213
OneMinusConstantAlphaFactor = 214
NeverDepth = 0
AlwaysDepth = 1
LessDepth = 2
LessEqualDepth = 3
EqualDepth = 4
GreaterEqualDepth = 5
GreaterDepth = 6
NotEqualDepth = 7
MultiplyOperation = 0
MixOperation = 1
AddOperation = 2
NoToneMapping = 0
LinearToneMapping = 1
ReinhardToneMapping = 2
CineonToneMapping = 3
ACESFilmicToneMapping = 4
CustomToneMapping = 5
AgXToneMapping = 6
NeutralToneMapping = 7
AttachedBindMode = 'attached'
DetachedBindMode = 'detached'
UVMapping = 300
CubeReflectionMapping = 301
CubeRefractionMapping = 302
EquirectangularReflectionMapping = 303
EquirectangularRefractionMapping = 304
RepeatWrapping = 1000
ClampToEdgeWrapping = 1001
MirroredRepeatWrapping = 1002
NearestFilter = 1003
NearestMipmapNearestFilter = 1004
NearestMipMapNearestFilter = 1004
NearestMipmapLinearFilter = 1005
NearestMipMapLinearFilter = 1005
LinearFilter = 1006
LinearMipmapNearestFilter = 1007
LinearMipMapNearestFilter = 1007
LinearMipmapLinearFilter = 1008
LinearMipMapLinearFilter = 1008
UnsignedByteType = 1009
ByteType = 1010
ShortType = 1011
UnsignedShortType = 1012
IntType = 1013
UnsignedIntType = 1014
FloatType = 1015
HalfFloatType = 1016
UnsignedShort4444Type = 1017
UnsignedShort5551Type = 1018
UnsignedInt248Type = 1020
UnsignedInt5999Type = 35902
UnsignedInt101111Type = 35899
AlphaFormat = 1021
RGBFormat = 1022
RGBAFormat = 1023
DepthFormat = 1026
DepthStencilFormat = 1027
RedFormat = 1028
RedIntegerFormat = 1029
RGFormat = 1030
RGIntegerFormat = 1031
RGBIntegerFormat = 1032
RGBAIntegerFormat = 1033
RGB_S3TC_DXT1_Format = 33776
RGBA_S3TC_DXT1_Format = 33777
RGBA_S3TC_DXT3_Format = 33778
RGBA_S3TC_DXT5_Format = 33779
RGB_PVRTC_4BPPV1_Format = 35840
RGB_PVRTC_2BPPV1_Format = 35841
RGBA_PVRTC_4BPPV1_Format = 35842
RGBA_PVRTC_2BPPV1_Format = 35843
RGB_ETC1_Format = 36196
RGB_ETC2_Format = 37492
RGBA_ETC2_EAC_Format = 37496
R11_EAC_Format = 37488
SIGNED_R11_EAC_Format = 37489
RG11_EAC_Format = 37490
SIGNED_RG11_EAC_Format = 37491
RGBA_ASTC_4x4_Format = 37808
RGBA_ASTC_5x4_Format = 37809
RGBA_ASTC_5x5_Format = 37810
RGBA_ASTC_6x5_Format = 37811
RGBA_ASTC_6x6_Format = 37812
RGBA_ASTC_8x5_Format = 37813
RGBA_ASTC_8x6_Format = 37814
RGBA_ASTC_8x8_Format = 37815
RGBA_ASTC_10x5_Format = 37816
RGBA_ASTC_10x6_Format = 37817
RGBA_ASTC_10x8_Format = 37818
RGBA_ASTC_10x10_Format = 37819
RGBA_ASTC_12x10_Format = 37820
RGBA_ASTC_12x12_Format = 37821
RGBA_BPTC_Format = 36492
RGB_BPTC_SIGNED_Format = 36494
RGB_BPTC_UNSIGNED_Format = 36495
RED_RGTC1_Format = 36283
SIGNED_RED_RGTC1_Format = 36284
RED_GREEN_RGTC2_Format = 36285
SIGNED_RED_GREEN_RGTC2_Format = 36286
InterpolateBezier = 2303
ZeroCurvatureEnding = 2400
ZeroSlopeEnding = 2401
WrapAroundEnding = 2402
TrianglesDrawMode = 0
TriangleStripDrawMode = 1
TriangleFanDrawMode = 2
BasicDepthPacking = 3200
RGBADepthPacking = 3201
RGBDepthPacking = 3202
RGDepthPacking = 3203
TangentSpaceNormalMap = 0
ObjectSpaceNormalMap = 1
NoColorSpace = ''
SRGBColorSpace = 'srgb'
LinearSRGBColorSpace = 'srgb-linear'
LinearTransfer = 'linear'
SRGBTransfer = 'srgb'
NoNormalPacking = ''
NormalRGPacking = 'rg'
NormalGAPacking = 'ga'
ZeroStencilOp = 0
KeepStencilOp = 7680
ReplaceStencilOp = 7681
IncrementStencilOp = 7682
DecrementStencilOp = 7683
IncrementWrapStencilOp = 34055
DecrementWrapStencilOp = 34056
InvertStencilOp = 5386
NeverStencilFunc = 512
LessStencilFunc = 513
EqualStencilFunc = 514
LessEqualStencilFunc = 515
GreaterStencilFunc = 516
NotEqualStencilFunc = 517
GreaterEqualStencilFunc = 518
AlwaysStencilFunc = 519
NeverCompare = 512
LessCompare = 513
EqualCompare = 514
LessEqualCompare = 515
GreaterCompare = 516
NotEqualCompare = 517
GreaterEqualCompare = 518
AlwaysCompare = 519
StaticDrawUsage = 35044
DynamicDrawUsage = 35048
StreamDrawUsage = 35040
StaticReadUsage = 35045
DynamicReadUsage = 35049
StreamReadUsage = 35041
StaticCopyUsage = 35046
DynamicCopyUsage = 35050
StreamCopyUsage = 35042
GLSL1 = '100'
GLSL3 = '300 es'
WebGLCoordinateSystem = 2000
WebGPUCoordinateSystem = 2001
TimestampQuery = { 'COMPUTE': 'compute', 'RENDER': 'render' }
InterpolationSamplingType = { 'PERSPECTIVE': 'perspective', 'LINEAR': 'linear', 'FLAT': 'flat' }
InterpolationSamplingMode = { 'NORMAL': 'normal', 'CENTROID': 'centroid', 'SAMPLE': 'sample', 'FIRST': 'first', 'EITHER': 'either' }
Compatibility = { 'TEXTURE_COMPARE': 'depthTextureCompare' }
RenderObjectRefreshType = { 'NONE': 0, 'SHARED': 1, 'FULL': 2 }


def _doc_member(obj, name):
    """Read ``name`` from a DOM-ish object: attribute access first, then
    mapping access (the interpreter's document stub is a JSObject)."""
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is not None:
        return value
    if isinstance(obj, dict):
        return obj.get(name)
    try:
        return obj[name]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """three.js ``Timer`` (core/Timer.js): the post-``Clock`` timing API.

    Unlike ``Clock``, querying never advances the state: ``update()``
    (called once per simulation step) computes the delta/elapsed pair,
    and ``getDelta()`` / ``getElapsed()`` only read it, so repeated
    queries within one step agree.  Internally times are milliseconds
    from a monotonic origin (``time.perf_counter``, the
    ``performance.now`` analogue); the getters return seconds, as
    upstream.  ``connect(document)`` enables the Page Visibility API
    behaviour (large deltas suppressed while hidden) when the document
    exposes ``hidden`` and ``addEventListener``.
    """

    isTimer = True

    def __init__(self):
        self._previousTime = 0.0
        self._currentTime = 0.0
        self._startTime = time.perf_counter() * 1000.0
        self._delta = 0.0
        self._elapsed = 0.0
        self._timescale = 1.0
        self._document = None
        self._pageVisibilityHandler = None

    def _now(self):
        return time.perf_counter() * 1000.0

    def connect(self, document):
        self._document = document
        if _doc_member(document, "hidden") is not None:

            def handler():
                if _doc_member(self._document, "hidden") is False:
                    self.reset()

            add = _doc_member(document, "addEventListener")
            if callable(add):
                add("visibilitychange", handler, False)
                self._pageVisibilityHandler = handler

    def disconnect(self):
        if self._pageVisibilityHandler is not None:
            remove = _doc_member(self._document, "removeEventListener")
            if callable(remove):
                remove("visibilitychange", self._pageVisibilityHandler)
            self._pageVisibilityHandler = None
        self._document = None

    def getDelta(self):
        return self._delta / 1000.0

    def getElapsed(self):
        return self._elapsed / 1000.0

    def getTimescale(self):
        return self._timescale

    def setTimescale(self, timescale):
        self._timescale = float(timescale)
        return self

    def reset(self):
        self._currentTime = self._now() - self._startTime
        return self

    def dispose(self):
        self.disconnect()

    def update(self, timestamp=None):
        if (self._pageVisibilityHandler is not None
                and _doc_member(self._document, "hidden") is True):
            self._delta = 0.0
        else:
            self._previousTime = self._currentTime
            now = self._now() if timestamp is None else float(timestamp)
            self._currentTime = now - self._startTime
            self._delta = ((self._currentTime - self._previousTime)
                           * self._timescale)
            self._elapsed += self._delta
        return self


# ---------------------------------------------------------------------------
# OrthographicCamera
# ---------------------------------------------------------------------------

class OrthographicCamera(Camera):
    """three.js ``OrthographicCamera``: parallel projection.

    The renderer consumes ``projectionMatrix`` generically, so no
    renderer change is needed.  ``Raycaster.setFromCamera`` carries the
    orthographic branch upstream specifies (rays parallel to the
    camera's -Z, starting from the near plane).
    """

    isOrthographicCamera = True

    def __init__(self, left=-1, right=1, top=1, bottom=-1, near=0.1,
                 far=2000, name=""):
        super().__init__(name)
        self.type = "OrthographicCamera"
        self.zoom = 1.0
        self.view = None
        self.left = float(left)
        self.right = float(right)
        self.top = float(top)
        self.bottom = float(bottom)
        self.near = float(near)
        self.far = float(far)
        self.coordinateSystem = WebGLCoordinateSystem
        self.reversedDepth = False
        self.updateProjectionMatrix()

    def setViewOffset(self, fullWidth, fullHeight, x, y, width, height):
        if self.view is None:
            self.view = {
                "enabled": True,
                "fullWidth": 1,
                "fullHeight": 1,
                "offsetX": 0,
                "offsetY": 0,
                "width": 1,
                "height": 1,
            }
        self.view["enabled"] = True
        self.view["fullWidth"] = fullWidth
        self.view["fullHeight"] = fullHeight
        self.view["offsetX"] = x
        self.view["offsetY"] = y
        self.view["width"] = width
        self.view["height"] = height
        self.updateProjectionMatrix()

    def clearViewOffset(self):
        if self.view is not None:
            self.view["enabled"] = False
        self.updateProjectionMatrix()

    def updateProjectionMatrix(self):
        dx = (self.right - self.left) / (2.0 * self.zoom)
        dy = (self.top - self.bottom) / (2.0 * self.zoom)
        cx = (self.right + self.left) / 2.0
        cy = (self.top + self.bottom) / 2.0

        left = cx - dx
        right = cx + dx
        top = cy + dy
        bottom = cy - dy

        if self.view is not None and self.view.get("enabled"):
            scale_w = ((self.right - self.left) / self.view["fullWidth"]
                       / self.zoom)
            scale_h = ((self.top - self.bottom) / self.view["fullHeight"]
                       / self.zoom)
            left += scale_w * self.view["offsetX"]
            right = left + scale_w * self.view["width"]
            top -= scale_h * self.view["offsetY"]
            bottom = top - scale_h * self.view["height"]

        self.projectionMatrix.makeOrthographic(
            left, right, top, bottom, self.near, self.far
        )
        return self


# ---------------------------------------------------------------------------
# TorusKnotGeometry
# ---------------------------------------------------------------------------

class TorusKnotGeometry(BufferGeometry):
    """three.js ``TorusKnotGeometry``: a tube wound around the torus
    knot curve defined by the coprime integers ``p`` and ``q``.
    Vertex, normal and uv generation follows the upstream formulas
    exactly (frame from two nearby curve points, normal = vertex minus
    curve point, normalised)."""

    def __init__(self, radius=1, tube=0.4, tubularSegments=64,
                 radialSegments=8, p=2, q=3):
        super().__init__()
        self.type = "TorusKnotGeometry"
        self.parameters = {
            "radius": radius,
            "tube": tube,
            "tubularSegments": tubularSegments,
            "radialSegments": radialSegments,
            "p": p,
            "q": q,
        }
        radius = float(radius)
        tube = float(tube)
        tubular_segments = int(math.floor(tubularSegments))
        radial_segments = int(math.floor(radialSegments))
        p = float(p)
        q = float(q)

        def position_on_curve(u):
            cu = math.cos(u)
            su = math.sin(u)
            qu_over_p = q / p * u
            cs = math.cos(qu_over_p)
            return (
                radius * (2.0 + cs) * 0.5 * cu,
                radius * (2.0 + cs) * 0.5 * su,
                radius * math.sin(qu_over_p) * 0.5,
            )

        positions = []
        normals = []
        uvs = []
        idx = []
        for i in range(tubular_segments + 1):
            u = i / tubular_segments * p * math.pi * 2.0
            p1 = position_on_curve(u)
            p2 = position_on_curve(u + 0.01)
            # Orthonormal basis, exactly as upstream:
            # T = P2 - P1, N = P2 + P1, B = T x N, N = B x T,
            # then B and N are normalised.
            t = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
            n = (p2[0] + p1[0], p2[1] + p1[1], p2[2] + p1[2])
            b = (t[1] * n[2] - t[2] * n[1],
                 t[2] * n[0] - t[0] * n[2],
                 t[0] * n[1] - t[1] * n[0])
            n = (b[1] * t[2] - b[2] * t[1],
                 b[2] * t[0] - b[0] * t[2],
                 b[0] * t[1] - b[1] * t[0])
            bl = math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
            b = (b[0] / bl, b[1] / bl, b[2] / bl)
            nl = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
            n = (n[0] / nl, n[1] / nl, n[2] / nl)
            for j in range(radial_segments + 1):
                v = j / radial_segments * math.pi * 2.0
                cx = -tube * math.cos(v)
                cy = tube * math.sin(v)
                vx = p1[0] + (cx * n[0] + cy * b[0])
                vy = p1[1] + (cx * n[1] + cy * b[1])
                vz = p1[2] + (cx * n[2] + cy * b[2])
                positions.append([vx, vy, vz])
                nx = vx - p1[0]
                ny = vy - p1[1]
                nz = vz - p1[2]
                il = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
                normals.append([nx * il, ny * il, nz * il])
                uvs.append([i / tubular_segments, j / radial_segments])
        for j in range(1, tubular_segments + 1):
            for i in range(1, radial_segments + 1):
                a = (radial_segments + 1) * (j - 1) + (i - 1)
                bq = (radial_segments + 1) * j + (i - 1)
                c = (radial_segments + 1) * j + i
                d = (radial_segments + 1) * (j - 1) + i
                idx += [a, bq, d, bq, c, d]
        self.attributes["position"] = np.array(positions, dtype=np.float32)
        self.attributes["normal"] = np.array(normals, dtype=np.float32)
        self.attributes["uv"] = np.array(uvs, dtype=np.float32)
        self.index = np.array(idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Canvas 2D + CanvasTexture
# ---------------------------------------------------------------------------

_NAMED_COLORS = {
    "black": (0, 0, 0), "silver": (192, 192, 192),
    "gray": (128, 128, 128), "grey": (128, 128, 128),
    "white": (255, 255, 255), "maroon": (128, 0, 0),
    "red": (255, 0, 0), "purple": (128, 0, 128),
    "fuchsia": (255, 0, 255), "magenta": (255, 0, 255),
    "green": (0, 128, 0), "lime": (0, 255, 0),
    "olive": (128, 128, 0), "yellow": (255, 255, 0),
    "navy": (0, 0, 128), "blue": (0, 0, 255),
    "teal": (0, 128, 128), "aqua": (0, 255, 255),
    "cyan": (0, 255, 255), "orange": (255, 165, 0),
    "pink": (255, 192, 203), "brown": (165, 42, 42),
}


def _parse_css_color(value):
    """Parse a CSS colour into an ``(r, g, b, a)`` tuple of ints 0..255.

    Accepts ``#rgb`` / ``#rrggbb`` / ``#rrggbbaa``, ``rgb()`` /
    ``rgba()``, common colour names, a ``Color``, an int ``0xRRGGBB``,
    or an ``(r, g, b[, a])`` sequence (0..1 floats or 0..255 ints).
    Unrecognised values fall back to opaque black.
    """
    if value is None:
        return (0, 0, 0, 255)
    if isinstance(value, Color):
        return (
            int(round(_clamp(value.r, 0.0, 1.0) * 255)),
            int(round(_clamp(value.g, 0.0, 1.0) * 255)),
            int(round(_clamp(value.b, 0.0, 1.0) * 255)),
            255,
        )
    if isinstance(value, (int, np.integer)):
        v = int(value)
        return ((v >> 16) & 255, (v >> 8) & 255, v & 255, 255)
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "transparent":
            return (0, 0, 0, 0)
        if s in _NAMED_COLORS:
            r, g, b = _NAMED_COLORS[s]
            return (r, g, b, 255)
        if s.startswith("#"):
            hexpart = s[1:]
            try:
                if len(hexpart) == 3:
                    r, g, b = (int(ch * 2, 16) for ch in hexpart)
                    return (r, g, b, 255)
                if len(hexpart) == 4:
                    r, g, b, a = (int(ch * 2, 16) for ch in hexpart)
                    return (r, g, b, a)
                if len(hexpart) == 6:
                    return (int(hexpart[0:2], 16), int(hexpart[2:4], 16),
                            int(hexpart[4:6], 16), 255)
                if len(hexpart) == 8:
                    return (int(hexpart[0:2], 16), int(hexpart[2:4], 16),
                            int(hexpart[4:6], 16), int(hexpart[6:8], 16))
            except ValueError:
                return (0, 0, 0, 255)
        if s.startswith("rgb"):
            inner = s[s.index("(") + 1:s.rindex(")")]
            parts = [p.strip() for p in inner.replace("/", ",").split(",")
                     if p.strip()]
            try:
                comps = []
                for part in parts[:3]:
                    if part.endswith("%"):
                        comps.append(
                            int(round(float(part[:-1]) / 100.0 * 255)))
                    else:
                        comps.append(int(round(float(part))))
                alpha = 255
                if len(parts) > 3:
                    apart = parts[3]
                    if apart.endswith("%"):
                        alpha = int(round(float(apart[:-1]) / 100.0 * 255))
                    else:
                        alpha = int(round(float(apart) * 255))
                while len(comps) < 3:
                    comps.append(0)
                return (comps[0], comps[1], comps[2], alpha)
            except ValueError:
                return (0, 0, 0, 255)
        return (0, 0, 0, 255)
    if isinstance(value, (list, tuple, np.ndarray)):
        seq = [float(c) for c in value]
        if len(seq) >= 3:
            if all(0.0 <= c <= 1.0 for c in seq):
                out = [int(round(c * 255)) for c in seq[:4]]
            else:
                out = [int(round(c)) for c in seq[:4]]
            while len(out) < 4:
                out.append(255)
            return tuple(out)
    return (0, 0, 0, 255)


class _CanvasGradient:
    """A linear or radial gradient usable as a fill/stroke style.

    Radial gradients are evaluated as concentric around the start
    circle's centre (t from radius0 to radius1); the examples use
    that form (a soft disc), and an offset end circle is not
    modelled."""

    def __init__(self, x0, y0, x1, y1, kind="linear", r0=0.0, r1=0.0):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.kind = kind
        self.r0 = float(r0)
        self.r1 = float(r1)
        self.stops = []

    def addColorStop(self, offset, color):
        self.stops.append(
            (_clamp(float(offset), 0.0, 1.0), _parse_css_color(color)))
        self.stops.sort(key=lambda stop: stop[0])

    def _lut(self):
        """256-entry RGBA lookup table (float32, 0..1)."""
        lut = np.zeros((256, 4), dtype=np.float32)
        stops = self.stops or [(0.0, (0, 0, 0, 255)),
                                (1.0, (0, 0, 0, 255))]
        ts = np.arange(256, dtype=np.float32) / 255.0
        cols = np.array(
            [[r / 255.0, g / 255.0, b / 255.0, a / 255.0]
             for _, (r, g, b, a) in stops],
            dtype=np.float32,
        )
        offs = np.array([o for o, _ in stops], dtype=np.float32)
        idx = np.searchsorted(offs, ts, side="right") - 1
        idx = np.clip(idx, 0, len(stops) - 1)
        nxt = np.clip(idx + 1, 0, len(stops) - 1)
        span = np.maximum(offs[nxt] - offs[idx], 1e-9)
        frac = np.clip((ts - offs[idx]) / span, 0.0, 1.0)[:, None]
        lut[:] = cols[idx] * (1.0 - frac) + cols[nxt] * frac
        before = ts < offs[0]
        after = ts > offs[-1]
        lut[before] = cols[0]
        lut[after] = cols[-1]
        return lut


class CanvasElement:
    """A minimal ``HTMLCanvasElement`` analogue: an RGBA pixel buffer
    (numpy ``uint8``, row 0 = top, HTML canvas order) plus one 2D
    context.  ``width`` / ``height`` are properties -- assigning either
    reallocates and clears the buffer, matching browser semantics.
    """

    def __init__(self, width=300, height=150):
        self._context = None
        self._width = 0
        self._height = 0
        self._buffer = np.zeros((0, 0, 4), dtype=np.uint8)
        # DOM surface page code touches on a canvas element: a style
        # bag (``canvas.style.display = ...`` in the offscreen-canvas
        # and multiple-scenes lessons) and event listeners
        # (``canvas.addEventListener('click', ...)`` in the tips /
        # debugging lessons).  Listeners live in a ``_listeners``
        # dict, the shape the interpreter's dispatch_event reads.
        self.style = {}
        self.dataset = {}
        self._listeners = {}
        self.width = width
        self.height = height

    def addEventListener(self, type, listener, *args, **kwargs):
        self._listeners.setdefault(str(type), []).append(listener)

    def removeEventListener(self, type, listener, *args, **kwargs):
        callbacks = self._listeners.get(str(type), [])
        if listener in callbacks:
            callbacks.remove(listener)

    @property
    def clientWidth(self):
        # No CSS layout model: the client size is the canvas size.
        return float(self._width)

    @property
    def clientHeight(self):
        return float(self._height)

    def getBoundingClientRect(self):
        w = float(self._width)
        h = float(self._height)
        return {
            "left": 0.0,
            "top": 0.0,
            "right": w,
            "bottom": h,
            "width": w,
            "height": h,
            "x": 0.0,
            "y": 0.0,
        }

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = max(0, int(value))
        self._reset_buffer()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = max(0, int(value))
        self._reset_buffer()

    def _reset_buffer(self):
        self._buffer = np.zeros(
            (self._height, self._width, 4), dtype=np.uint8)

    def getContext(self, kind="2d", *args):
        if str(kind) != "2d":
            return None
        if self._context is None:
            self._context = CanvasRenderingContext2D(self)
        return self._context


class CanvasRenderingContext2D:
    """Minimal Canvas 2D rasteriser writing into a ``CanvasElement``'s
    RGBA buffer with source-over compositing.

    Supported: ``fillRect`` / ``clearRect`` / ``strokeRect``,
    ``drawImage`` (canvas, Texture or numpy image), linear gradients
    (``createLinearGradient``) as styles, and simple path fills
    (``beginPath`` / ``moveTo`` / ``lineTo`` / ``closePath`` / ``arc`` /
    ``fill``).  Text is not rasterised (the port has no font engine);
    ``fillText`` / ``strokeText`` record the request in
    ``dropped_text`` and draw nothing.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self.fillStyle = "#000000"
        self.strokeStyle = "#000000"
        self.globalAlpha = 1.0
        self.lineWidth = 1.0
        self.dropped_text = []
        self._path = []
        # Current transform matrix [a, b, c, d, e, f] (the canvas
        # 2D affine) plus the save/restore stack.  Path points are
        # stored in user space and transformed when painted, as in a
        # browser; rects and images are transformed at call time.
        self._ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self._ctm_stack = []

    # -- transform state -------------------------------------------------
    def save(self):
        self._ctm_stack.append(list(self._ctm))

    def restore(self):
        if self._ctm_stack:
            self._ctm = self._ctm_stack.pop()

    def _multiply_ctm(self, a, b, c, d, e, f):
        A, B, C, D, E, F = self._ctm
        self._ctm = [
            A * a + C * b,
            B * a + D * b,
            A * c + C * d,
            B * c + D * d,
            A * e + C * f + E,
            B * e + D * f + F,
        ]

    def translate(self, x, y):
        self._multiply_ctm(1.0, 0.0, 0.0, 1.0, float(x), float(y))

    def scale(self, x, y=None):
        if y is None:
            y = x
        self._multiply_ctm(float(x), 0.0, 0.0, float(y), 0.0, 0.0)

    def rotate(self, angle):
        cos = math.cos(float(angle))
        sin = math.sin(float(angle))
        self._multiply_ctm(cos, sin, -sin, cos, 0.0, 0.0)

    def transform(self, a, b, c, d, e, f):
        self._multiply_ctm(float(a), float(b), float(c),
                           float(d), float(e), float(f))

    def setTransform(self, a=1.0, b=0.0, c=0.0, d=1.0, e=0.0, f=0.0):
        self._ctm = [float(a), float(b), float(c),
                     float(d), float(e), float(f)]

    def resetTransform(self):
        self._ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    def getImageData(self, sx=0, sy=0, sw=1, sh=1):
        buf = self.canvas._buffer
        h, w = buf.shape[:2]
        x0 = max(0, int(sx))
        y0 = max(0, int(sy))
        rw = max(0, int(sw))
        rh = max(0, int(sh))
        out = np.zeros((rh, rw, 4), dtype=np.uint8)
        cx1 = min(w, x0 + rw)
        cy1 = min(h, y0 + rh)
        if cx1 > x0 and cy1 > y0:
            out[: cy1 - y0, : cx1 - x0] = buf[y0:cy1, x0:cx1]
        return {
            "data": out.reshape(-1),
            "width": float(rw),
            "height": float(rh),
        }

    def putImageData(self, imageData, dx=0, dy=0):
        if not isinstance(imageData, dict):
            return
        data = imageData.get("data")
        rw = int(imageData.get("width", 0))
        rh = int(imageData.get("height", 0))
        if data is None or rw <= 0 or rh <= 0:
            return
        src = np.asarray(data, dtype=np.uint8).reshape(rh, rw, 4)
        buf = self.canvas._buffer
        h, w = buf.shape[:2]
        x0 = max(0, int(dx))
        y0 = max(0, int(dy))
        cx1 = min(w, x0 + rw)
        cy1 = min(h, y0 + rh)
        if cx1 > x0 and cy1 > y0:
            buf[y0:cy1, x0:cx1] = src[: cy1 - y0, : cx1 - x0]

    def getTransform(self):
        a, b, c, d, e, f = self._ctm
        return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}

    def _xform(self, x, y):
        a, b, c, d, e, f = self._ctm
        return (a * x + c * y + e, b * x + d * y + f)

    def _ctm_is_identity(self):
        return self._ctm == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    def _ctm_axis_aligned(self):
        return self._ctm[1] == 0.0 and self._ctm[2] == 0.0

    # -- compositing ---------------------------------------------------
    def _blend(self, xs, ys, rgba):
        """Source-over blend of a float RGBA array (h, w, 4, 0..1) into
        the buffer region whose top-left is (xs, ys)."""
        buf = self.canvas._buffer
        h, w = rgba.shape[:2]
        if h <= 0 or w <= 0:
            return
        region = buf[ys:ys + h, xs:xs + w].astype(np.float32) / 255.0
        src_a = rgba[:, :, 3:4] * float(self.globalAlpha)
        dst_a = region[:, :, 3:4]
        out_a = src_a + dst_a * (1.0 - src_a)
        safe_a = np.where(out_a > 0.0, out_a, 1.0)
        out_rgb = (rgba[:, :, :3] * src_a
                   + region[:, :, :3] * dst_a * (1.0 - src_a)) / safe_a
        out = np.concatenate([out_rgb, out_a], axis=2)
        out = np.where(out_a > 0.0, out, 0.0)
        buf[ys:ys + h, xs:xs + w] = np.clip(
            np.round(out * 255.0), 0, 255).astype(np.uint8)

    def _style_field(self, style, xs, ys, w, h):
        """An (h, w, 4) float RGBA field for a style over a rect."""
        if isinstance(style, _CanvasGradient):
            lut = style._lut()
            px = (np.arange(xs, xs + w, dtype=np.float32) + 0.5)
            py = (np.arange(ys, ys + h, dtype=np.float32) + 0.5)
            if style.kind == "radial":
                dist = np.sqrt((px[None, :] - style.x0) ** 2
                               + (py[:, None] - style.y0) ** 2)
                span = max(style.r1 - style.r0, 1e-9)
                t = np.clip((dist - style.r0) / span, 0.0, 1.0)
                return lut[(t * 255.0).astype(np.int64)]
            dx = style.x1 - style.x0
            dy = style.y1 - style.y0
            denom = dx * dx + dy * dy
            if denom == 0.0:
                t = np.zeros((h, w), dtype=np.float32)
            else:
                gx = ((px[None, :] - style.x0) * dx
                      + (py[:, None] - style.y0) * dy) / denom
                t = np.clip(gx, 0.0, 1.0)
            return lut[(t * 255.0).astype(np.int64)]
        r, g, b, a = _parse_css_color(style)
        field = np.empty((h, w, 4), dtype=np.float32)
        field[:, :] = (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        return field

    def _clip_rect(self, x, y, w, h):
        cw = self.canvas._width
        ch = self.canvas._height
        xs = max(0, int(math.floor(x)))
        ys = max(0, int(math.floor(y)))
        xe = min(cw, int(math.ceil(x + w)))
        ye = min(ch, int(math.ceil(y + h)))
        return xs, ys, max(0, xe - xs), max(0, ye - ys)

    # -- rect family ---------------------------------------------------
    def _rect_corners(self, x, y, w, h):
        return [self._xform(x, y), self._xform(x + w, y),
                self._xform(x + w, y + h), self._xform(x, y + h)]

    def fillRect(self, x, y, w, h):
        if not self._ctm_is_identity() and not self._ctm_axis_aligned():
            # Rotated/skewed rect: paint it as a filled polygon.
            self._fill_polys([self._rect_corners(x, y, w, h)],
                             self.fillStyle)
            return
        if not self._ctm_is_identity():
            corners = self._rect_corners(x, y, w, h)
            xs_all = [p[0] for p in corners]
            ys_all = [p[1] for p in corners]
            x, y = min(xs_all), min(ys_all)
            w = max(xs_all) - x
            h = max(ys_all) - y
        xs, ys, rw, rh = self._clip_rect(x, y, w, h)
        if rw <= 0 or rh <= 0:
            return
        self._blend(xs, ys, self._style_field(self.fillStyle, xs, ys,
                                              rw, rh))

    def clearRect(self, x, y, w, h):
        if not self._ctm_is_identity():
            corners = self._rect_corners(x, y, w, h)
            xs_all = [p[0] for p in corners]
            ys_all = [p[1] for p in corners]
            x, y = min(xs_all), min(ys_all)
            w = max(xs_all) - x
            h = max(ys_all) - y
        xs, ys, rw, rh = self._clip_rect(x, y, w, h)
        if rw <= 0 or rh <= 0:
            return
        self.canvas._buffer[ys:ys + rh, xs:xs + rw] = 0

    def strokeRect(self, x, y, w, h):
        lw = float(self.lineWidth)
        half = lw / 2.0
        # Border band: outer rect minus inner rect, drawn as 4 fills
        # in the stroke style.
        saved = self.fillStyle
        self.fillStyle = self.strokeStyle
        try:
            self.fillRect(x - half, y - half, w + lw, lw)
            self.fillRect(x - half, y + h - half, w + lw, lw)
            self.fillRect(x - half, y + half, lw, h - lw)
            self.fillRect(x + w - half, y + half, lw, h - lw)
        finally:
            self.fillStyle = saved

    # -- images ----------------------------------------------------------
    def drawImage(self, image, dx, dy, dw=None, dh=None):
        src = None
        if isinstance(image, CanvasElement):
            src = image._buffer
        elif isinstance(image, Texture):
            src = image.image
        else:
            arr = _coerce_texture_image(image)
            if arr is not None:
                src = arr
        if src is None or src.size == 0:
            return
        src = np.asarray(src)
        sh, sw = src.shape[:2]
        if dw is None:
            dw = sw
        if dh is None:
            dh = sh
        dw = int(round(dw))
        dh = int(round(dh))
        if dw <= 0 or dh <= 0:
            return
        if not self._ctm_is_identity():
            # The destination rect goes through the CTM; rotation is
            # approximated by the axis-aligned bounding box.
            corners = [self._xform(dx, dy), self._xform(dx + dw, dy),
                       self._xform(dx + dw, dy + dh),
                       self._xform(dx, dy + dh)]
            xs_all = [p[0] for p in corners]
            ys_all = [p[1] for p in corners]
            dx, dy = min(xs_all), min(ys_all)
            dw = int(round(max(xs_all) - dx))
            dh = int(round(max(ys_all) - dy))
            if dw <= 0 or dh <= 0:
                return
        rows = np.minimum(
            (np.arange(dh, dtype=np.float32) / dh * sh).astype(np.int64),
            sh - 1)
        cols = np.minimum(
            (np.arange(dw, dtype=np.float32) / dw * sw).astype(np.int64),
            sw - 1)
        scaled = src[rows][:, cols]
        if scaled.shape[2] == 3:
            alpha = np.full(scaled.shape[:2] + (1,), 255, dtype=np.uint8)
            scaled = np.concatenate([scaled, alpha], axis=2)
        ox = int(round(dx))
        oy = int(round(dy))
        xs = max(0, ox)
        ys = max(0, oy)
        xe = min(self.canvas._width, ox + dw)
        ye = min(self.canvas._height, oy + dh)
        if xe <= xs or ye <= ys:
            return
        sub = scaled[ys - oy:ye - oy, xs - ox:xe - ox]
        self._blend(xs, ys, sub.astype(np.float32) / 255.0)

    # -- gradients -------------------------------------------------------
    def createLinearGradient(self, x0, y0, x1, y1):
        return _CanvasGradient(x0, y0, x1, y1)

    def createRadialGradient(self, x0, y0, r0, x1, y1, r1):
        return _CanvasGradient(x0, y0, x1, y1, kind="radial",
                               r0=r0, r1=r1)

    # -- paths (simple polygon fill) --------------------------------------
    def beginPath(self):
        self._path = []

    def moveTo(self, x, y):
        self._path.append([(float(x), float(y))])

    def lineTo(self, x, y):
        if not self._path:
            self._path.append([])
        self._path[-1].append((float(x), float(y)))

    def closePath(self):
        pass  # fill() closes subpaths implicitly, as canvas does

    def arc(self, x, y, radius, startAngle, endAngle,
            counterclockwise=False):
        x = float(x)
        y = float(y)
        radius = float(radius)
        start = float(startAngle)
        end = float(endAngle)
        if counterclockwise:
            while end > start:
                end -= 2.0 * math.pi
        else:
            while end < start:
                end += 2.0 * math.pi
        steps = max(2, int(abs(end - start) / (math.pi / 16.0)) + 1)
        pts = []
        for i in range(steps + 1):
            a = start + (end - start) * i / steps
            pts.append((x + radius * math.cos(a),
                        y + radius * math.sin(a)))
        if not self._path or not self._path[-1]:
            self._path.append(pts)
        else:
            self._path[-1].extend(pts)

    def fill(self):
        polys = [[self._xform(px, py) for (px, py) in p]
                 for p in self._path if len(p) >= 2]
        self._fill_polys(polys, self.fillStyle)

    def _fill_polys(self, polys, style):
        buf = self.canvas._buffer
        ch, cw = buf.shape[:2]
        if ch == 0 or cw == 0:
            return
        mask = np.zeros((ch, cw), dtype=bool)
        if not polys:
            return
        py = np.arange(ch, dtype=np.float64) + 0.5
        for poly in polys:
            pts = np.asarray(poly, dtype=np.float64)
            xs = pts[:, 0]
            ys = pts[:, 1]
            # Even-odd rule via per-row edge crossings.
            for row in range(ch):
                yv = py[row]
                crossings = []
                n = len(pts)
                for i in range(n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i + 1) % n]
                    if (y1 <= yv < y2) or (y2 <= yv < y1):
                        crossings.append(
                            x1 + (yv - y1) / (y2 - y1) * (x2 - x1))
                crossings.sort()
                for k in range(0, len(crossings) - 1, 2):
                    xa = int(math.ceil(crossings[k] - 0.5))
                    xb = int(math.floor(crossings[k + 1] - 0.5))
                    xa = max(0, xa)
                    xb = min(cw - 1, xb)
                    if xb >= xa:
                        mask[row, xa:xb + 1] = True
        if not mask.any():
            return
        ys_idx, xs_idx = np.nonzero(mask)
        field = self._style_field(
            style, 0, 0, cw, ch)[ys_idx, xs_idx]
        region = np.zeros((ch, cw, 4), dtype=np.float32)
        region[ys_idx, xs_idx] = field
        region[:, :, 3] *= mask
        # Blend only the masked pixels: compose manually to avoid
        # touching the rest of the buffer.
        src_a = region[:, :, 3:4] * float(self.globalAlpha)
        dst = buf.astype(np.float32) / 255.0
        dst_a = dst[:, :, 3:4]
        out_a = src_a + dst_a * (1.0 - src_a)
        safe_a = np.where(out_a > 0.0, out_a, 1.0)
        out_rgb = (region[:, :, :3] * src_a
                   + dst[:, :, :3] * dst_a * (1.0 - src_a)) / safe_a
        out = np.concatenate([out_rgb, out_a], axis=2)
        out = np.where(mask[:, :, None], out, dst)
        buf[:] = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)

    def stroke(self):
        """Stroke the current path with ``strokeStyle`` / ``lineWidth``:
        each segment is rasterised as the set of pixels within
        lineWidth / 2 of it (round caps/joins fall out of the distance
        test)."""
        polys = [[self._xform(px, py) for (px, py) in p]
                 for p in self._path if len(p) >= 2]
        for poly in polys:
            for i in range(len(poly) - 1):
                self._stroke_segment(poly[i], poly[i + 1])

    def _stroke_segment(self, p1, p2):
        buf = self.canvas._buffer
        ch, cw = buf.shape[:2]
        if ch == 0 or cw == 0:
            return
        half = max(0.5, float(self.lineWidth) / 2.0)
        x1, y1 = p1
        x2, y2 = p2
        xs = max(0, int(math.floor(min(x1, x2) - half - 1)))
        xe = min(cw, int(math.ceil(max(x1, x2) + half + 1)))
        ys = max(0, int(math.floor(min(y1, y2) - half - 1)))
        ye = min(ch, int(math.ceil(max(y1, y2) + half + 1)))
        if xe <= xs or ye <= ys:
            return
        px = (np.arange(xs, xe, dtype=np.float64) + 0.5)
        py = (np.arange(ys, ye, dtype=np.float64) + 0.5)
        gx, gy = np.meshgrid(px, py)
        dx = x2 - x1
        dy = y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 0.0:
            dist = np.sqrt((gx - x1) ** 2 + (gy - y1) ** 2)
        else:
            t = ((gx - x1) * dx + (gy - y1) * dy) / seg_len2
            t = np.clip(t, 0.0, 1.0)
            dist = np.sqrt((gx - (x1 + t * dx)) ** 2
                           + (gy - (y1 + t * dy)) ** 2)
        mask = dist <= half
        if not mask.any():
            return
        field = self._style_field(
            self.strokeStyle, xs, ys, xe - xs, ye - ys)
        region = np.zeros((ye - ys, xe - xs, 4), dtype=np.float32)
        region[mask] = field[mask]
        self._blend(xs, ys, region)

    # -- text (not rasterised) --------------------------------------------
    def fillText(self, text, x, y, maxWidth=None):
        self.dropped_text.append(str(text))

    def strokeText(self, text, x, y, maxWidth=None):
        self.dropped_text.append(str(text))

    def measureText(self, text):
        return _MeasuredText(0.0)


class _MeasuredText:
    def __init__(self, width):
        self.width = width


def _canvas_image(canvas):
    """Extract an (h, w, 3) image from a canvas-like object, or None.

    Order: this module's ``CanvasElement`` (a live view of its buffer),
    an object exposing a pixel buffer, anything ``_coerce_texture_image``
    understands, and finally a DOM-stub element with numeric
    width/height, which yields a blank image of that size -- the same
    pixels an undrawn real canvas would produce.
    """
    if canvas is None:
        return None
    if isinstance(canvas, CanvasElement):
        if canvas._buffer.size == 0:
            return None
        return canvas._buffer[:, :, :3]
    for attr in ("_buffer", "pixels", "_pixels"):
        buf = getattr(canvas, attr, None)
        if buf is not None:
            arr = _coerce_texture_image(buf)
            if arr is not None:
                return arr
    arr = _coerce_texture_image(canvas)
    if arr is not None:
        return arr
    w = _doc_member(canvas, "width")
    h = _doc_member(canvas, "height")
    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
        w = int(w)
        h = int(h)
        if w > 0 and h > 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
    return None


class CanvasTexture(Texture):
    """three.js ``CanvasTexture``: a ``Texture`` whose image comes from
    a canvas.  Sets ``needsUpdate`` at construction, as upstream.  For
    a ``CanvasElement`` the image is a live view of the canvas buffer,
    so later 2D drawing appears without further work; after a canvas
    resize (which reallocates the buffer) call ``refresh()``.
    """

    isCanvasTexture = True

    def __init__(self, canvas=None, mapping=None, wrapS=None, wrapT=None,
                 magFilter=None, minFilter=None, format=None, type=None,
                 anisotropy=None, name=""):
        self.canvas = canvas
        super().__init__(_canvas_image(canvas), name=name)
        self.mapping = UVMapping if mapping is None else mapping
        self.wrapS = ClampToEdgeWrapping if wrapS is None else wrapS
        self.wrapT = ClampToEdgeWrapping if wrapT is None else wrapT
        self.magFilter = LinearFilter if magFilter is None else magFilter
        self.minFilter = (LinearMipmapLinearFilter if minFilter is None
                          else minFilter)
        self.format = RGBAFormat if format is None else format
        self.type = UnsignedByteType if type is None else type
        self.anisotropy = 1 if anisotropy is None else anisotropy
        self.needsUpdate = True

    def refresh(self):
        img = _canvas_image(self.canvas)
        if img is not None:
            self.image = img
        self.needsUpdate = True
        return self

    def clone(self):
        tex = CanvasTexture(self.canvas, name=self.name)
        tex.mapping = self.mapping
        tex.wrapS = self.wrapS
        tex.wrapT = self.wrapT
        return tex

__all__ += [
    "REVISION",
    "MOUSE",
    "TOUCH",
    "CullFaceNone",
    "CullFaceBack",
    "CullFaceFront",
    "CullFaceFrontBack",
    "BasicShadowMap",
    "PCFShadowMap",
    "PCFSoftShadowMap",
    "VSMShadowMap",
    "NoBlending",
    "NormalBlending",
    "AdditiveBlending",
    "SubtractiveBlending",
    "MultiplyBlending",
    "CustomBlending",
    "MaterialBlending",
    "AddEquation",
    "SubtractEquation",
    "ReverseSubtractEquation",
    "MinEquation",
    "MaxEquation",
    "ZeroFactor",
    "OneFactor",
    "SrcColorFactor",
    "OneMinusSrcColorFactor",
    "SrcAlphaFactor",
    "OneMinusSrcAlphaFactor",
    "DstAlphaFactor",
    "OneMinusDstAlphaFactor",
    "DstColorFactor",
    "OneMinusDstColorFactor",
    "SrcAlphaSaturateFactor",
    "ConstantColorFactor",
    "OneMinusConstantColorFactor",
    "ConstantAlphaFactor",
    "OneMinusConstantAlphaFactor",
    "NeverDepth",
    "AlwaysDepth",
    "LessDepth",
    "LessEqualDepth",
    "EqualDepth",
    "GreaterEqualDepth",
    "GreaterDepth",
    "NotEqualDepth",
    "MultiplyOperation",
    "MixOperation",
    "AddOperation",
    "NoToneMapping",
    "LinearToneMapping",
    "ReinhardToneMapping",
    "CineonToneMapping",
    "ACESFilmicToneMapping",
    "CustomToneMapping",
    "AgXToneMapping",
    "NeutralToneMapping",
    "AttachedBindMode",
    "DetachedBindMode",
    "UVMapping",
    "CubeReflectionMapping",
    "CubeRefractionMapping",
    "EquirectangularReflectionMapping",
    "EquirectangularRefractionMapping",
    "RepeatWrapping",
    "ClampToEdgeWrapping",
    "MirroredRepeatWrapping",
    "NearestFilter",
    "NearestMipmapNearestFilter",
    "NearestMipMapNearestFilter",
    "NearestMipmapLinearFilter",
    "NearestMipMapLinearFilter",
    "LinearFilter",
    "LinearMipmapNearestFilter",
    "LinearMipMapNearestFilter",
    "LinearMipmapLinearFilter",
    "LinearMipMapLinearFilter",
    "UnsignedByteType",
    "ByteType",
    "ShortType",
    "UnsignedShortType",
    "IntType",
    "UnsignedIntType",
    "FloatType",
    "HalfFloatType",
    "UnsignedShort4444Type",
    "UnsignedShort5551Type",
    "UnsignedInt248Type",
    "UnsignedInt5999Type",
    "UnsignedInt101111Type",
    "AlphaFormat",
    "RGBFormat",
    "RGBAFormat",
    "DepthFormat",
    "DepthStencilFormat",
    "RedFormat",
    "RedIntegerFormat",
    "RGFormat",
    "RGIntegerFormat",
    "RGBIntegerFormat",
    "RGBAIntegerFormat",
    "RGB_S3TC_DXT1_Format",
    "RGBA_S3TC_DXT1_Format",
    "RGBA_S3TC_DXT3_Format",
    "RGBA_S3TC_DXT5_Format",
    "RGB_PVRTC_4BPPV1_Format",
    "RGB_PVRTC_2BPPV1_Format",
    "RGBA_PVRTC_4BPPV1_Format",
    "RGBA_PVRTC_2BPPV1_Format",
    "RGB_ETC1_Format",
    "RGB_ETC2_Format",
    "RGBA_ETC2_EAC_Format",
    "R11_EAC_Format",
    "SIGNED_R11_EAC_Format",
    "RG11_EAC_Format",
    "SIGNED_RG11_EAC_Format",
    "RGBA_ASTC_4x4_Format",
    "RGBA_ASTC_5x4_Format",
    "RGBA_ASTC_5x5_Format",
    "RGBA_ASTC_6x5_Format",
    "RGBA_ASTC_6x6_Format",
    "RGBA_ASTC_8x5_Format",
    "RGBA_ASTC_8x6_Format",
    "RGBA_ASTC_8x8_Format",
    "RGBA_ASTC_10x5_Format",
    "RGBA_ASTC_10x6_Format",
    "RGBA_ASTC_10x8_Format",
    "RGBA_ASTC_10x10_Format",
    "RGBA_ASTC_12x10_Format",
    "RGBA_ASTC_12x12_Format",
    "RGBA_BPTC_Format",
    "RGB_BPTC_SIGNED_Format",
    "RGB_BPTC_UNSIGNED_Format",
    "RED_RGTC1_Format",
    "SIGNED_RED_RGTC1_Format",
    "RED_GREEN_RGTC2_Format",
    "SIGNED_RED_GREEN_RGTC2_Format",
    "InterpolateBezier",
    "ZeroCurvatureEnding",
    "ZeroSlopeEnding",
    "WrapAroundEnding",
    "TrianglesDrawMode",
    "TriangleStripDrawMode",
    "TriangleFanDrawMode",
    "BasicDepthPacking",
    "RGBADepthPacking",
    "RGBDepthPacking",
    "RGDepthPacking",
    "TangentSpaceNormalMap",
    "ObjectSpaceNormalMap",
    "NoColorSpace",
    "SRGBColorSpace",
    "LinearSRGBColorSpace",
    "LinearTransfer",
    "SRGBTransfer",
    "NoNormalPacking",
    "NormalRGPacking",
    "NormalGAPacking",
    "ZeroStencilOp",
    "KeepStencilOp",
    "ReplaceStencilOp",
    "IncrementStencilOp",
    "DecrementStencilOp",
    "IncrementWrapStencilOp",
    "DecrementWrapStencilOp",
    "InvertStencilOp",
    "NeverStencilFunc",
    "LessStencilFunc",
    "EqualStencilFunc",
    "LessEqualStencilFunc",
    "GreaterStencilFunc",
    "NotEqualStencilFunc",
    "GreaterEqualStencilFunc",
    "AlwaysStencilFunc",
    "NeverCompare",
    "LessCompare",
    "EqualCompare",
    "LessEqualCompare",
    "GreaterCompare",
    "NotEqualCompare",
    "GreaterEqualCompare",
    "AlwaysCompare",
    "StaticDrawUsage",
    "DynamicDrawUsage",
    "StreamDrawUsage",
    "StaticReadUsage",
    "DynamicReadUsage",
    "StreamReadUsage",
    "StaticCopyUsage",
    "DynamicCopyUsage",
    "StreamCopyUsage",
    "GLSL1",
    "GLSL3",
    "WebGLCoordinateSystem",
    "WebGPUCoordinateSystem",
    "TimestampQuery",
    "InterpolationSamplingType",
    "InterpolationSamplingMode",
    "Compatibility",
    "RenderObjectRefreshType",
    "Timer",
    "OrthographicCamera",
    "TorusKnotGeometry",
    "CanvasElement",
    "CanvasRenderingContext2D",
    "CanvasTexture",
]


# ---------------------------------------------------------------------------
# ShaderMaterial and post-processing
#
# ShaderMaterial / RawShaderMaterial carry GLSL source plus a uniforms
# dict; the actual compilation and execution live in render.py (which
# drives glsl.py's parser and glsl_eval.py's vector evaluator).  The
# EffectComposer family is the minimal pass chain:
# passes exchange float RGB frames through WebGLRenderTarget.frame_rgb,
# and a pass that renders "to screen" writes the palette-quantised
# result into renderer.frame, which is what the host displays.
# ---------------------------------------------------------------------------

class ShaderMaterial(Material):
    """three.js ShaderMaterial: user GLSL for both stages.

    ``uniforms`` maps names to ``{"value": ...}`` entries, exactly as
    in three.js; values may be scalars, Color / Vector2 / Vector3 /
    Matrix4, Texture, or lists (for uniform arrays).  ``defines`` is
    an optional {name: value} dict fed to the GLSL preprocessor.
    Fog and lights uniforms are not wired (the port does not inject
    the fog/lights chunks into user shaders); ``wireframe`` is
    accepted but shader materials are always drawn filled."""

    isShaderMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        uniforms = p.get("uniforms")
        self.uniforms = uniforms if uniforms is not None else {}
        self.vertexShader = str(p.get(
            "vertexShader",
            "void main() { gl_Position = projectionMatrix * "
            "modelViewMatrix * vec4( position, 1.0 ); }"))
        self.fragmentShader = str(p.get(
            "fragmentShader",
            "void main() { gl_FragColor = vec4( 1.0, 1.0, 1.0, 1.0 ); }"))
        defines = p.get("defines")
        self.defines = defines if defines is not None else {}


class RawShaderMaterial(ShaderMaterial):
    """ShaderMaterial without the automatic prefix (no built-in
    uniforms / attributes are declared; the shader declares its own,
    and the host still supplies values for the standard names)."""

    isRawShaderMaterial = True


class WebGLRenderTarget:
    """Minimal render target: a size plus a Texture placeholder.

    The port's passes keep the actual pixels in ``frame_rgb`` (an
    (h, w, 3) float array in 0..1, or None before the first write);
    ``texture.image`` is kept pointing at the same array so code that
    samples ``target.texture`` sees the current frame."""

    isWebGLRenderTarget = True

    def __init__(self, width=256, height=192, options=None):
        self.width = int(width)
        self.height = int(height)
        self.texture = Texture(
            np.zeros((self.height, self.width, 3), dtype=np.float64))
        self.frame_rgb = None
        self.depthBuffer = True
        self.stencilBuffer = False

    def setSize(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.frame_rgb = None
        self.texture = Texture(
            np.zeros((self.height, self.width, 3), dtype=np.float64))
        return self

    def clone(self):
        other = WebGLRenderTarget(self.width, self.height)
        return other

    def dispose(self):
        self.frame_rgb = None


class Pass:
    """Base of the composer passes (three/addons Pass)."""

    isPass = True

    def __init__(self):
        self.enabled = True
        self.needsSwap = True
        self.clear = False
        self.renderToScreen = False

    def setSize(self, width, height):
        pass

    def render(self, renderer, writeBuffer, readBuffer,
               deltaTime=0.0, maskActive=False):
        raise NotImplementedError


class RenderPass(Pass):
    """Renders a scene into the composer's read buffer (and, when it
    is the last enabled pass, the renderer's frame already holds the
    scene, which is the screen result)."""

    isRenderPass = True

    def __init__(self, scene, camera):
        super().__init__()
        self.scene = scene
        self.camera = camera
        self.needsSwap = False
        self.clear = True

    def render(self, renderer, writeBuffer, readBuffer,
               deltaTime=0.0, maskActive=False):
        renderer.render(self.scene, self.camera)
        render_module = WebGLRenderer._load_render_module()
        readBuffer.frame_rgb = render_module.palette_frame_to_rgb(
            renderer.frame)
        readBuffer.texture.image = readBuffer.frame_rgb


class ShaderPass(Pass):
    """Runs a fragment shader over the whole frame.

    ``shader`` is a ShaderMaterial or a shader dict of the form
    {uniforms, vertexShader, fragmentShader} (the CopyShader shape).
    The read buffer is bound to the ``textureID`` uniform ("tDiffuse"
    by default).  The vertex shader is not executed: the pass is a
    virtual fullscreen quad and the fragment stage receives the
    standard ``vUv`` varying directly (the pass-through vertex shaders
    used by copy/blur-style passes only forward uv)."""

    isShaderPass = True

    def __init__(self, shader, textureID="tDiffuse"):
        super().__init__()
        self.textureID = str(textureID)
        if isinstance(shader, ShaderMaterial):
            self.material = shader
            self.uniforms = shader.uniforms
        else:
            uniforms = {}
            for name, entry in (shader.get("uniforms") or {}).items():
                if isinstance(entry, dict):
                    uniforms[name] = {"value": entry.get("value")}
                else:
                    uniforms[name] = {"value": entry}
            self.uniforms = uniforms
            self.material = ShaderMaterial({
                "uniforms": uniforms,
                "vertexShader": shader.get("vertexShader", ""),
                "fragmentShader": shader.get("fragmentShader", ""),
            })
        self.needsSwap = True

    def render(self, renderer, writeBuffer, readBuffer,
               deltaTime=0.0, maskActive=False):
        entry = self.uniforms.get(self.textureID)
        if isinstance(entry, dict):
            entry["value"] = readBuffer
        else:
            self.uniforms[self.textureID] = {"value": readBuffer}
        render_module = WebGLRenderer._load_render_module()
        rgb, discard = render_module.render_shader_pass(
            self.material, renderer.width, renderer.height)
        if discard is not None and readBuffer.frame_rgb is not None:
            rgb = rgb.copy()
            rgb[discard] = readBuffer.frame_rgb[discard]
        if self.renderToScreen:
            renderer.frame = render_module.quantize_rgb(rgb)
        else:
            writeBuffer.frame_rgb = rgb
            writeBuffer.texture.image = rgb


class OutputPass(Pass):
    """three.js OutputPass applies tone mapping and the output colour
    space conversion.  Both are the identity in this port (the whole
    pipeline works in display space), so the pass forwards the read
    buffer unchanged."""

    isOutputPass = True

    def __init__(self):
        super().__init__()
        self.needsSwap = False

    def render(self, renderer, writeBuffer, readBuffer,
               deltaTime=0.0, maskActive=False):
        if readBuffer.frame_rgb is None:
            return
        if self.renderToScreen:
            render_module = WebGLRenderer._load_render_module()
            renderer.frame = render_module.quantize_rgb(
                readBuffer.frame_rgb)
        else:
            writeBuffer.frame_rgb = readBuffer.frame_rgb
            writeBuffer.texture.image = readBuffer.frame_rgb


class _IdentityEffectPass(Pass):
    """Base for the effect passes this port does not evaluate
    (bloom / film grain): the pass chain runs and the frame passes
    through unchanged, exactly OutputPass's contract.  The pass
    parameters are stored with their three.js names so page code
    that tweaks them (``bloomPass.strength = ...``) works; the
    visual effect itself is NOT applied -- recorded honestly in the
    coverage notes."""

    def render(self, renderer, writeBuffer, readBuffer,
               deltaTime=0.0, maskActive=False):
        if readBuffer.frame_rgb is None:
            return
        if self.renderToScreen:
            render_module = WebGLRenderer._load_render_module()
            renderer.frame = render_module.quantize_rgb(
                readBuffer.frame_rgb)
        else:
            writeBuffer.frame_rgb = readBuffer.frame_rgb
            writeBuffer.texture.image = readBuffer.frame_rgb


class UnrealBloomPass(_IdentityEffectPass):
    """three/addons UnrealBloomPass (identity pass-through; see
    _IdentityEffectPass)."""

    isUnrealBloomPass = True

    def __init__(self, resolution=None, strength=1.0, radius=0.0,
                 threshold=0.0):
        super().__init__()
        self.needsSwap = False
        self.resolution = resolution
        self.strength = float(strength)
        self.radius = float(radius)
        self.threshold = float(threshold)

    def setSize(self, width, height):
        pass


class BloomPass(_IdentityEffectPass):
    """The r132-era three/addons BloomPass (identity pass-through;
    see _IdentityEffectPass)."""

    isBloomPass = True

    def __init__(self, strength=1.0, kernelSize=25, sigma=4.0,
                 resolution=256):
        super().__init__()
        self.needsSwap = False
        self.strength = float(strength)
        self.kernelSize = kernelSize
        self.sigma = float(sigma)
        self.resolution = resolution

    def setSize(self, width, height):
        pass


class FilmPass(_IdentityEffectPass):
    """three/addons FilmPass (identity pass-through; see
    _IdentityEffectPass)."""

    isFilmPass = True

    def __init__(self, noiseIntensity=0.5, scanlinesIntensity=0.05,
                 scanlinesCount=2048, grayscale=True):
        super().__init__()
        self.needsSwap = False
        self.uniforms = {
            "noiseIntensity": {"value": float(noiseIntensity)},
            "scanlinesIntensity": {"value": float(scanlinesIntensity)},
            "scanlinesCount": {"value": float(scanlinesCount)},
            "grayscale": {"value": 1.0 if grayscale else 0.0},
        }

    def setSize(self, width, height):
        pass


# The three/addons CopyShader, in its original shape: a shader dict
# with a tDiffuse sampler and an opacity uniform.
CopyShader = {
    "uniforms": {
        "tDiffuse": {"value": None},
        "opacity": {"value": 1.0},
    },
    "vertexShader": (
        "varying vec2 vUv;\n"
        "void main() {\n"
        "  vUv = uv;\n"
        "  gl_Position = projectionMatrix * modelViewMatrix * "
        "vec4( position, 1.0 );\n"
        "}\n"
    ),
    "fragmentShader": (
        "uniform float opacity;\n"
        "uniform sampler2D tDiffuse;\n"
        "varying vec2 vUv;\n"
        "void main() {\n"
        "  vec4 texel = texture2D( tDiffuse, vUv );\n"
        "  gl_FragColor = opacity * texel;\n"
        "}\n"
    ),
}


class EffectComposer:
    """Minimal EffectComposer: an ordered pass chain
    over two WebGLRenderTargets.  ``render()`` runs the enabled
    passes; the last enabled pass renders to screen (into
    ``renderer.frame``).  The stencil/depth mask (maskActive) dance of
    the original is not modelled."""

    isEffectComposer = True

    def __init__(self, renderer, renderTarget=None):
        self.renderer = renderer
        self._pixelRatio = 1.0
        if renderTarget is not None:
            self.renderTarget1 = renderTarget
            self.renderTarget2 = renderTarget.clone()
        else:
            width = int(getattr(renderer, "width", 256))
            height = int(getattr(renderer, "height", 192))
            self.renderTarget1 = WebGLRenderTarget(width, height)
            self.renderTarget2 = WebGLRenderTarget(width, height)
        self.writeBuffer = self.renderTarget1
        self.readBuffer = self.renderTarget2
        self.renderToScreen = True
        self.passes = []
        self.copyPass = ShaderPass(CopyShader)

    def addPass(self, pass_):
        self.passes.append(pass_)
        pass_.setSize(self.renderTarget1.width,
                      self.renderTarget1.height)

    def insertPass(self, pass_, index):
        self.passes.insert(int(index), pass_)

    def removePass(self, pass_):
        if pass_ in self.passes:
            self.passes.remove(pass_)

    def isLastEnabledPass(self, index):
        return all(not p.enabled for p in self.passes[index + 1:])

    def render(self, deltaTime=0.0):
        write = self.writeBuffer
        read = self.readBuffer
        for i, pass_ in enumerate(self.passes):
            if not pass_.enabled:
                continue
            pass_.renderToScreen = (
                self.renderToScreen and self.isLastEnabledPass(i))
            pass_.render(self.renderer, write, read, deltaTime, False)
            if pass_.needsSwap:
                write, read = read, write
        self.writeBuffer = write
        self.readBuffer = read

    def setSize(self, width, height):
        self.renderTarget1.setSize(width, height)
        self.renderTarget2.setSize(width, height)
        for pass_ in self.passes:
            pass_.setSize(width, height)

    def getSize(self):
        return {"width": self.renderTarget1.width,
                "height": self.renderTarget1.height}

    def setPixelRatio(self, ratio):
        self._pixelRatio = float(ratio)

    def getPixelRatio(self):
        return self._pixelRatio

    def reset(self, renderTarget=None):
        if renderTarget is not None:
            self.renderTarget1 = renderTarget
            self.renderTarget2 = renderTarget.clone()
        else:
            self.renderTarget1.frame_rgb = None
            self.renderTarget2.frame_rgb = None
        self.writeBuffer = self.renderTarget1
        self.readBuffer = self.renderTarget2


class EdgesGeometry(BufferGeometry):
    """Extract hard edges from a triangle BufferGeometry for LineSegments
    (three.js EdgesGeometry)."""

    isEdgesGeometry = True
    type = "EdgesGeometry"

    def __init__(self, geometry=None, thresholdAngle=1.0):
        super().__init__()
        self.parameters = {
            "geometry": geometry,
            "thresholdAngle": float(thresholdAngle),
        }
        if geometry is None:
            return
        pos_attr = geometry.attributes.get("position")
        if pos_attr is None:
            return
        pos = np.asarray(pos_attr, dtype=np.float32).reshape(-1, 3)
        n_verts = int(pos.shape[0])
        if n_verts < 3:
            return
        idx = geometry._index
        if idx is not None:
            indices = np.asarray(idx, dtype=np.int32).reshape(-1)
            n_tris = int(indices.size // 3)
            tris = indices[: n_tris * 3].reshape(n_tris, 3)
        else:
            n_tris = n_verts // 3
            tris = np.arange(n_tris * 3, dtype=np.int32).reshape(n_tris, 3)
        if n_tris == 0:
            return

        threshold_dot = math.cos(math.radians(float(thresholdAngle)))
        precision = 10000.0

        def _vkey(v):
            return (
                int(round(float(v[0]) * precision)),
                int(round(float(v[1]) * precision)),
                int(round(float(v[2]) * precision)),
            )

        edge_map = {}
        for t in range(n_tris):
            i0 = int(tris[t, 0])
            i1 = int(tris[t, 1])
            i2 = int(tris[t, 2])
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= n_verts or i1 >= n_verts or i2 >= n_verts:
                continue
            p0 = pos[i0]
            p1 = pos[i1]
            p2 = pos[i2]
            normal = np.cross(p1 - p0, p2 - p0).astype(np.float64)
            nlen = float(np.linalg.norm(normal))
            if nlen < 1e-12:
                continue
            normal /= nlen

            verts = ((i0, p0), (i1, p1), (i2, p2))
            for e in range(3):
                ia, pa = verts[e]
                ib, pb = verts[(e + 1) % 3]
                ka = _vkey(pa)
                kb = _vkey(pb)
                if ka == kb:
                    continue
                if ka < kb:
                    key = (ka, kb)
                    pair = (pa, pb)
                else:
                    key = (kb, ka)
                    pair = (pb, pa)
                prev = edge_map.get(key)
                if prev is None:
                    edge_map[key] = (pair[0], pair[1], normal, False)
                else:
                    a0, b0, n0, emitted = prev
                    if not emitted:
                        if float(np.dot(n0, normal)) <= threshold_dot:
                            edge_map[key] = (a0, b0, n0, True)
                        else:
                            edge_map[key] = None

        segments = []
        for entry in edge_map.values():
            if entry is None:
                continue
            a0, b0, _, _ = entry
            segments.append(a0)
            segments.append(b0)

        if segments:
            self.attributes["position"] = np.asarray(segments, dtype=np.float32).reshape(-1, 3)
        else:
            self.attributes["position"] = np.zeros((0, 3), dtype=np.float32)


class InstancedMesh(Mesh):
    """Mesh rendered multiple times with per-instance transforms and optional
    per-instance colours (three.js InstancedMesh)."""

    isInstancedMesh = True

    def __init__(self, geometry=None, material=None, count=0):
        super().__init__(geometry, material)
        self.count = max(0, int(count))
        ident = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.float32
        )
        mats = np.tile(ident, self.count) if self.count > 0 else np.zeros(0, dtype=np.float32)
        self.instanceMatrix = BufferAttribute(mats, 16)
        self.instanceMatrix.needsUpdate = False
        self.instanceColor = None

    def setMatrixAt(self, index, matrix):
        i = int(index)
        if i < 0 or i >= self.count:
            return
        if hasattr(matrix, "_a"):
            vals = np.asarray(matrix._a, dtype=np.float32).T.reshape(16)
        elif hasattr(matrix, "elements"):
            vals = np.asarray(matrix.elements, dtype=np.float32).reshape(16)
        else:
            vals = np.asarray(matrix, dtype=np.float32).reshape(16)
        self.instanceMatrix.array[i * 16 : (i + 1) * 16] = vals

    def getMatrixAt(self, index, matrix):
        i = int(index)
        if i < 0 or i >= self.count:
            return
        vals = self.instanceMatrix.array[i * 16 : (i + 1) * 16]
        if hasattr(matrix, "_a"):
            matrix._a = vals.reshape(4, 4).T.astype(np.float64)
        elif hasattr(matrix, "fromArray"):
            matrix.fromArray(vals)

    def setColorAt(self, index, color):
        i = int(index)
        if i < 0 or i >= self.count:
            return
        if self.instanceColor is None:
            self.instanceColor = BufferAttribute(
                np.ones(self.count * 3, dtype=np.float32), 3
            )
            self.instanceColor.needsUpdate = False
        c = color if isinstance(color, Color) else Color(color)
        self.instanceColor.array[i * 3 : i * 3 + 3] = (
            np.float32(c.r),
            np.float32(c.g),
            np.float32(c.b),
        )

    def getColorAt(self, index, color):
        i = int(index)
        if self.instanceColor is None or i < 0 or i >= self.count:
            return
        rgb = self.instanceColor.array[i * 3 : i * 3 + 3]
        if hasattr(color, "setRGB"):
            color.setRGB(float(rgb[0]), float(rgb[1]), float(rgb[2]))

    def copy(self, source, recursive=True):
        super().copy(source, recursive)
        self.count = int(source.count)
        self.instanceMatrix = BufferAttribute(source.instanceMatrix.array.copy(), 16)
        self.instanceMatrix.needsUpdate = False
        if source.instanceColor is not None:
            self.instanceColor = BufferAttribute(source.instanceColor.array.copy(), 3)
            self.instanceColor.needsUpdate = False
        else:
            self.instanceColor = None
        return self

    def clone(self, recursive=True):
        return InstancedMesh(self.geometry, self.material, self.count).copy(self, recursive)

    def dispose(self):
        return None


class Vector4:
    """4D vector (three.js Vector4)."""

    isVector4 = True

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def set(self, x, y, z, w):
        self.x, self.y, self.z, self.w = float(x), float(y), float(z), float(w)
        return self

    def setScalar(self, s):
        s = float(s)
        self.x = self.y = self.z = self.w = s
        return self

    def copy(self, v):
        self.x, self.y, self.z, self.w = float(v.x), float(v.y), float(v.z), float(v.w)
        return self

    def clone(self):
        return Vector4(self.x, self.y, self.z, self.w)

    def add(self, v):
        self.x += v.x
        self.y += v.y
        self.z += v.z
        self.w += v.w
        return self

    def sub(self, v):
        self.x -= v.x
        self.y -= v.y
        self.z -= v.z
        self.w -= v.w
        return self

    def multiplyScalar(self, s):
        s = float(s)
        self.x *= s
        self.y *= s
        self.z *= s
        self.w *= s
        return self

    def divideScalar(self, s):
        s = float(s)
        return self.multiplyScalar(1.0 / s) if s != 0.0 else self.setScalar(0.0)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z + self.w * v.w

    def lengthSq(self):
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

    def length(self):
        return math.sqrt(self.lengthSq())

    def normalize(self):
        return self.divideScalar(self.length() or 1.0)

    def lerp(self, v, alpha):
        a = float(alpha)
        self.x += (v.x - self.x) * a
        self.y += (v.y - self.y) * a
        self.z += (v.z - self.z) * a
        self.w += (v.w - self.w) * a
        return self

    def equals(self, v):
        return v.x == self.x and v.y == self.y and v.z == self.z and v.w == self.w

    def fromArray(self, arr, offset=0):
        i = int(offset)
        self.x, self.y, self.z, self.w = float(arr[i]), float(arr[i + 1]), float(arr[i + 2]), float(arr[i + 3])
        return self

    def toArray(self, arr=None, offset=0):
        if arr is None:
            return [self.x, self.y, self.z, self.w]
        i = int(offset)
        arr[i], arr[i + 1], arr[i + 2], arr[i + 3] = self.x, self.y, self.z, self.w
        return arr

    def applyMatrix4(self, m):
        e = m._a
        x, y, z, w = self.x, self.y, self.z, self.w
        self.x = float(e[0, 0] * x + e[0, 1] * y + e[0, 2] * z + e[0, 3] * w)
        self.y = float(e[1, 0] * x + e[1, 1] * y + e[1, 2] * z + e[1, 3] * w)
        self.z = float(e[2, 0] * x + e[2, 1] * y + e[2, 2] * z + e[2, 3] * w)
        self.w = float(e[3, 0] * x + e[3, 1] * y + e[3, 2] * z + e[3, 3] * w)
        return self

    def negate(self):
        self.x, self.y, self.z, self.w = -self.x, -self.y, -self.z, -self.w
        return self


class Spherical:
    """Spherical coordinates (radius, phi, theta) matching three.js."""

    def __init__(self, radius=1.0, phi=0.0, theta=0.0):
        self.radius = float(radius)
        self.phi = float(phi)
        self.theta = float(theta)

    def set(self, radius, phi, theta):
        self.radius = float(radius)
        self.phi = float(phi)
        self.theta = float(theta)
        return self

    def copy(self, other):
        self.radius = float(other.radius)
        self.phi = float(other.phi)
        self.theta = float(other.theta)
        return self

    def clone(self):
        return Spherical(self.radius, self.phi, self.theta)

    def makeSafe(self):
        eps = 0.000001
        self.phi = max(eps, min(math.pi - eps, self.phi))
        return self

    def setFromVector3(self, v):
        return self.setFromCartesianCoords(v.x, v.y, v.z)

    def setFromCartesianCoords(self, x, y, z):
        self.radius = math.sqrt(x * x + y * y + z * z)
        if self.radius == 0.0:
            self.theta = 0.0
            self.phi = 0.0
        else:
            self.theta = math.atan2(x, z)
            self.phi = math.acos(max(-1.0, min(1.0, y / self.radius)))
        return self


class Layers:
    """Object visibility layer bitmask (three.js Layers)."""

    def __init__(self):
        self.mask = 1

    def set(self, channel):
        self.mask = (1 << int(channel)) & 0xFFFFFFFF

    def enable(self, channel):
        self.mask |= 1 << int(channel)

    def enableAll(self):
        self.mask = 0xFFFFFFFF

    def toggle(self, channel):
        self.mask ^= 1 << int(channel)

    def disable(self, channel):
        self.mask &= ~(1 << int(channel))

    def disableAll(self):
        self.mask = 0

    def test(self, layers):
        return (self.mask & int(layers.mask)) != 0

    def isEnabled(self, channel):
        return (self.mask & (1 << int(channel))) != 0


class DepthTexture(Texture):
    """Depth attachment texture stub (three.js DepthTexture)."""

    isDepthTexture = True

    def __init__(self, width=1, height=1, type=None, mapping=None,
                 wrapS=None, wrapT=None, magFilter=None, minFilter=None,
                 anisotropy=1, format=None):
        super().__init__(None)
        self.width = int(width)
        self.height = int(height)
        self.type = type
        self.format = format


class PMREMGenerator:
    """Stub PMREMGenerator returning environment textures for IBL callers."""

    def __init__(self, renderer=None):
        self._renderer = renderer

    def fromScene(self, scene, sigma=0.0, near=0.1, far=100.0):
        rt = WebGLRenderTarget(64, 64)
        rt.texture = CubeTexture()
        return rt

    def fromEquirectangular(self, equirectangular):
        rt = WebGLRenderTarget(64, 64)
        rt.texture = equirectangular
        return rt

    def fromCubemap(self, cubemap):
        rt = WebGLRenderTarget(64, 64)
        rt.texture = cubemap
        return rt

    def compileEquirectangularShader(self):
        return None

    def compileCubemapShader(self):
        return None

    def dispose(self):
        return None


class PlaneHelper(Object3D):
    """Visual helper for THREE.Plane (three.js PlaneHelper)."""

    type = "PlaneHelper"

    def __init__(self, plane=None, size=1.0, color=0xFFFF00):
        super().__init__()
        self.plane = plane
        self.size = float(size)
        self.color = _as_color(color, Color(0xFFFF00))

    def updateMatrixWorld(self, force=False):
        if self.plane is not None:
            n = self.plane.normal
            self.position.set(-n.x * self.plane.constant, -n.y * self.plane.constant, -n.z * self.plane.constant)
            self.lookAt(self.position.x + n.x, self.position.y + n.y, self.position.z + n.z)
        super().updateMatrixWorld(force)


class MeshPhysicalMaterial(MeshStandardMaterial):
    """three.js MeshPhysicalMaterial (rendered via MeshStandardMaterial path)."""

    isMeshPhysicalMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        self.clearcoat = float(p.get("clearcoat", 0.0))
        self.clearcoatRoughness = float(p.get("clearcoatRoughness", 0.0))
        self.ior = float(p.get("ior", 1.5))
        self.reflectivity = float(p.get("reflectivity", 0.5))
        self.sheen = float(p.get("sheen", 0.0))
        self.sheenRoughness = float(p.get("sheenRoughness", 1.0))
        self.sheenColor = _as_color(p.get("sheenColor"), Color(0.0, 0.0, 0.0))
        self.specularIntensity = float(p.get("specularIntensity", 1.0))
        self.specularColor = _as_color(p.get("specularColor"), Color(1.0, 1.0, 1.0))
        self.transmission = float(p.get("transmission", 0.0))
        self.thickness = float(p.get("thickness", 0.0))
        self.attenuationDistance = float(p.get("attenuationDistance", float("inf")))
        self.attenuationColor = _as_color(p.get("attenuationColor"), Color(1.0, 1.0, 1.0))
        self.iridescence = float(p.get("iridescence", 0.0))


class ShadowMaterial(Material):
    """three.js ShadowMaterial."""

    isShadowMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self.transparent = True


class MeshDepthMaterial(Material):
    """three.js MeshDepthMaterial."""

    isMeshDepthMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        self.depthPacking = p.get("depthPacking", 0)
        self.wireframe = bool(p.get("wireframe", False))


class InstancedBufferGeometry(BufferGeometry):
    """three.js InstancedBufferGeometry."""

    isInstancedBufferGeometry = True

    def __init__(self):
        super().__init__()
        self.instanceCount = float("inf")


class WebGLCubeRenderTarget(WebGLRenderTarget):
    """three.js WebGLCubeRenderTarget."""

    isWebGLCubeRenderTarget = True

    def __init__(self, size=1, options=None):
        super().__init__(size, size, options)
        self.texture = CubeTexture()

    def fromEquirectangularTexture(self, renderer, texture):
        self.texture = texture
        return self

    def clear(self, renderer, color=True, depth=True, stencil=True):
        return None


class ArrayCamera(PerspectiveCamera):
    """three.js ArrayCamera."""

    isArrayCamera = True

    def __init__(self, array=None):
        super().__init__()
        self.cameras = list(array) if array is not None else []


class FileLoader:
    """Minimal synchronous FileLoader stub."""

    def __init__(self, manager=None):
        self.manager = manager
        self.responseType = ""

    def setResponseType(self, value):
        self.responseType = value
        return self

    def load(self, url, onLoad=None, onProgress=None, onError=None):
        if callable(onError):
            onError(f"FileLoader: no local file {url!r}")
        return None

    def loadAsync(self, url, onProgress=None):
        return SyncPromise.rejected(f"FileLoader: no local file {url!r}")


class UniformsGroup:
    """Minimal three.js UniformsGroup."""

    isUniformsGroup = True

    def __init__(self):
        self.uniforms = []
        self.name = ""

    def add(self, uniform):
        self.uniforms.append(uniform)
        return self

    def remove(self, uniform):
        if uniform in self.uniforms:
            self.uniforms.remove(uniform)
        return self

    def setName(self, name):
        self.name = str(name)
        return self

    def clone(self):
        g = UniformsGroup()
        g.name = self.name
        g.uniforms = list(self.uniforms)
        return g


class RoomEnvironment(Scene):
    """Minimal three.js RoomEnvironment studio lighting scene."""

    def __init__(self):
        super().__init__()
        self.add(AmbientLight(0xFFFFFF, 0.75))
        d = DirectionalLight(0xFFFFFF, 1.0)
        d.position.set(5, 8, 6)
        self.add(d)

    def dispose(self):
        return None


class ImprovedNoise:
    """Ken Perlin Improved Noise (three/addons/math/ImprovedNoise.js)."""

    _P = [
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
        190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,
        135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
        223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
        251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    ]
    _PERM = _P + _P

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(t, a, b):
        return a + t * (b - a)

    @staticmethod
    def _grad(h, x, y, z):
        h = h & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise(self, x, y, z):
        p = self._PERM
        fx, fy, fz = float(x), float(y), float(z)
        X = int(math.floor(fx)) & 255
        Y = int(math.floor(fy)) & 255
        Z = int(math.floor(fz)) & 255
        x = fx - math.floor(fx)
        y = fy - math.floor(fy)
        z = fz - math.floor(fz)
        u = self._fade(x)
        v = self._fade(y)
        w = self._fade(z)
        A = p[X] + Y
        AA = p[A] + Z
        AB = p[A + 1] + Z
        B = p[X + 1] + Y
        BA = p[B] + Z
        BB = p[B + 1] + Z
        return self._lerp(
            w,
            self._lerp(
                v,
                self._lerp(u, self._grad(p[AA], x, y, z), self._grad(p[BA], x - 1, y, z)),
                self._lerp(u, self._grad(p[AB], x, y - 1, z), self._grad(p[BB], x - 1, y - 1, z)),
            ),
            self._lerp(
                v,
                self._lerp(u, self._grad(p[AA + 1], x, y, z - 1), self._grad(p[BA + 1], x - 1, y, z - 1)),
                self._lerp(u, self._grad(p[AB + 1], x, y - 1, z - 1), self._grad(p[BB + 1], x - 1, y - 1, z - 1)),
            ),
        )


class CubeCamera(Object3D):
    """three.js CubeCamera stub."""

    isCubeCamera = True

    def __init__(self, near=0.1, far=1000.0, renderTarget=None):
        super().__init__()
        self.type = "CubeCamera"
        self.renderTarget = renderTarget if renderTarget is not None else WebGLCubeRenderTarget(64)

    def update(self, renderer, scene):
        return None


class LOD(Object3D):
    """three.js Level of Detail (LOD) container."""

    isLOD = True

    def __init__(self):
        super().__init__()
        self.type = "LOD"
        self.levels = []
        self.autoUpdate = True

    def addLevel(self, obj, distance=0.0, hysteresis=0.0):
        dist = abs(float(distance))
        self.levels.append({"object": obj, "distance": dist, "hysteresis": float(hysteresis)})
        self.levels.sort(key=lambda lv: lv["distance"])
        self.add(obj)
        return self

    def getObjectForDistance(self, distance):
        if not self.levels:
            return None
        d = float(distance)
        chosen = self.levels[0]["object"]
        for lv in self.levels:
            if d >= lv["distance"]:
                chosen = lv["object"]
            else:
                break
        return chosen

    def update(self, camera):
        if len(self.levels) > 1 and camera is not None:
            cam_pos = Vector3().setFromMatrixPosition(camera.matrixWorld)
            obj_pos = Vector3().setFromMatrixPosition(self.matrixWorld)
            dist = cam_pos.distanceTo(obj_pos)
            self.levels[0]["object"].visible = True
            i = 1
            while i < len(self.levels):
                if dist >= self.levels[i]["distance"]:
                    self.levels[i - 1]["object"].visible = False
                    self.levels[i]["object"].visible = True
                else:
                    break
                i += 1
            while i < len(self.levels):
                self.levels[i]["object"].visible = False
                i += 1


class InterleavedBuffer:
    """three.js InterleavedBuffer."""

    isInterleavedBuffer = True

    def __init__(self, array, stride):
        self.array = np.asarray(array)
        self.stride = int(stride)
        self.count = int(self.array.size // self.stride) if self.stride > 0 else 0
        self.usage = 0
        self.needsUpdate = False

    def setUsage(self, value):
        self.usage = value
        return self

    def clone(self, data=None):
        return InterleavedBuffer(self.array.copy(), self.stride)


class InterleavedBufferAttribute:
    """three.js InterleavedBufferAttribute."""

    isInterleavedBufferAttribute = True

    def __init__(self, interleavedBuffer, itemSize, offset, normalized=False):
        self.data = interleavedBuffer
        self.itemSize = int(itemSize)
        self.offset = int(offset)
        self.normalized = bool(normalized)

    @property
    def count(self):
        return self.data.count

    @property
    def array(self):
        return self.data.array


class Data3DTexture(Texture):
    """three.js Data3DTexture stub."""

    isData3DTexture = True

    def __init__(self, data=None, width=1, height=1, depth=1):
        super().__init__(None)
        self.image = {"data": data, "width": int(width), "height": int(height), "depth": int(depth)}


class BufferGeometryLoader:
    """three.js BufferGeometryLoader stub."""

    def __init__(self, manager=None):
        self.manager = manager

    def load(self, url, onLoad=None, onProgress=None, onError=None):
        if callable(onError):
            onError(f"BufferGeometryLoader: no local file {url!r}")
        return BufferGeometry()

    def parse(self, json_data):
        return BufferGeometry()


class LightProbe(Light):
    """three.js LightProbe."""

    isLightProbe = True

    def __init__(self, sh=None, intensity=1.0):
        super().__init__(0xFFFFFF, intensity)
        self.sh = sh


class Sky(Mesh):
    """Minimal three/addons/objects/Sky.js dome with uniform bag."""

    def __init__(self):
        geo = BoxGeometry(1, 1, 1)
        mat = MeshBasicMaterial({"color": 0x87CEEB, "side": BackSide})
        mat.uniforms = {
            "turbidity": {"value": 2.0},
            "rayleigh": {"value": 1.0},
            "mieCoefficient": {"value": 0.005},
            "mieDirectionalG": {"value": 0.8},
            "sunPosition": {"value": Vector3(0, 1, 0)},
            "up": {"value": Vector3(0, 1, 0)},
            "cloudCoverage": {"value": 0.4},
            "cloudDensity": {"value": 0.4},
            "cloudElevation": {"value": 0.5},
            "showSunDisc": {"value": True},
        }
        super().__init__(geo, mat)
        self.isSky = True


AlwaysStencilFunc = 519
HalfFloatType = 1016


class Uniform:
    """three.js Uniform wrapper."""

    isUniform = True

    def __init__(self, value=None):
        self.value = value

    def clone(self):
        v = self.value
        if hasattr(v, "clone"):
            v = v.clone()
        elif isinstance(v, list):
            v = list(v)
        return Uniform(v)


class _UniformsUtils:
    @staticmethod
    def clone(uniforms):
        out = {}
        for k, u in (uniforms or {}).items():
            if isinstance(u, dict) and "value" in u:
                v = u["value"]
                if hasattr(v, "clone"):
                    v = v.clone()
                elif isinstance(v, list):
                    v = list(v)
                out[k] = {"value": v}
            elif hasattr(u, "clone"):
                out[k] = u.clone()
            else:
                out[k] = u
        return out

    @staticmethod
    def merge(uniforms_list):
        merged = {}
        for udict in (uniforms_list or []):
            for k, v in _UniformsUtils.clone(udict).items():
                merged[k] = v
        return merged


UniformsUtils = _UniformsUtils()


class InstancedBufferAttribute(BufferAttribute):
    """three.js InstancedBufferAttribute."""

    isInstancedBufferAttribute = True

    def __init__(self, array, itemSize, normalized=False, meshPerAttribute=1):
        super().__init__(array, itemSize, normalized)
        self.meshPerAttribute = int(meshPerAttribute)


class LineDashedMaterial(LineBasicMaterial):
    """three.js LineDashedMaterial."""

    isLineDashedMaterial = True

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        p = _split_params(params, kwargs)
        self.scale = float(p.get("scale", 1.0))
        self.dashSize = float(p.get("dashSize", 3.0))
        self.gapSize = float(p.get("gapSize", 1.0))


class VideoTexture(Texture):
    """three.js VideoTexture stub."""

    isVideoTexture = True

    def __init__(self, video=None, mapping=None, wrapS=None, wrapT=None,
                 magFilter=None, minFilter=None, format=None, type=None,
                 anisotropy=1):
        super().__init__(video)


class SkeletonHelper(Object3D):
    """three.js SkeletonHelper stub."""

    isSkeletonHelper = True

    def __init__(self, obj=None):
        super().__init__()
        self.root = obj
        self.bones = []

    def update(self):
        return None


class RectAreaLight(Light):
    """three.js RectAreaLight."""

    isRectAreaLight = True

    def __init__(self, color=0xFFFFFF, intensity=1.0, width=10.0, height=10.0):
        super().__init__(color, intensity)
        self.width = float(width)
        self.height = float(height)


class WireframeGeometry(BufferGeometry):
    """Extract all triangle edges for LineSegments (three.js WireframeGeometry)."""

    isWireframeGeometry = True

    def __init__(self, geometry=None):
        super().__init__()
        if geometry is None:
            return
        pos_attr = geometry.attributes.get("position")
        if pos_attr is None:
            return
        pos = np.asarray(pos_attr, dtype=np.float32).reshape(-1, 3)
        n_verts = int(pos.shape[0])
        if n_verts < 3:
            return
        idx = geometry._index
        if idx is not None:
            indices = np.asarray(idx, dtype=np.int32).reshape(-1)
            n_tris = int(indices.size // 3)
            tris = indices[: n_tris * 3].reshape(n_tris, 3)
        else:
            n_tris = n_verts // 3
            tris = np.arange(n_tris * 3, dtype=np.int32).reshape(n_tris, 3)
        if n_tris == 0:
            return
        e0 = tris[:, [0, 1]]
        e1 = tris[:, [1, 2]]
        e2 = tris[:, [2, 0]]
        edges = np.concatenate([e0, e1, e2], axis=0).reshape(-1)
        valid = edges[(edges >= 0) & (edges < n_verts)]
        if valid.size >= 2:
            valid = valid[: (valid.size // 2) * 2]
            self.attributes["position"] = pos[valid].copy()


class LineLoop(Line):
    """three.js LineLoop."""

    isLineLoop = True


class CapsuleGeometry(CylinderGeometry):
    """Approximate CapsuleGeometry using CylinderGeometry."""

    def __init__(self, radius=1.0, length=1.0, capSegments=4, radialSegments=8):
        r = float(radius)
        total_h = float(length) + 2.0 * r
        super().__init__(r, r, total_h, max(4, int(radialSegments)), 1)


class MeshToonMaterial(MeshLambertMaterial):
    """three.js MeshToonMaterial."""

    isMeshToonMaterial = True


class MeshMatcapMaterial(MeshLambertMaterial):
    """three.js MeshMatcapMaterial."""

    isMeshMatcapMaterial = True


class _ColorManagement:
    enabled = True
    legacyMode = False


ColorManagement = _ColorManagement()


class _AnimationUtils:
    @staticmethod
    def subclip(sourceClip, name, startFrame, endFrame, fps=30):
        return sourceClip

    @staticmethod
    def makeClipAdditive(targetClip, referenceFrame=0, referenceClip=None, fps=30):
        return targetClip


AnimationUtils = _AnimationUtils()


class _ShaderChunk(dict):
    def __missing__(self, key):
        return ""


ShaderChunk = _ShaderChunk()


def _decode_radiance_hdr(raw):
    """Decode Radiance .hdr (RGBE RLE / uncompressed) bytes into
    (width, height, uint8_rgba_flat) with Reinhard tone-mapping."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError("HDRLoader: expected bytes")
    data = bytes(raw)
    pos = 0
    n = len(data)
    # Read header until empty line
    while pos < n:
        nl = data.find(b"\n", pos)
        if nl < 0:
            raise ValueError("HDRLoader: truncated header")
        line = data[pos:nl].decode("ascii", errors="replace").strip()
        pos = nl + 1
        if line == "":
            break
    # Read resolution line: e.g. "-Y 512 +X 1024"
    nl = data.find(b"\n", pos)
    if nl < 0:
        raise ValueError("HDRLoader: missing resolution line")
    res_line = data[pos:nl].decode("ascii", errors="replace").strip()
    pos = nl + 1
    parts = res_line.split()
    if len(parts) != 4:
        raise ValueError(f"HDRLoader: bad resolution {res_line!r}")
    height = int(parts[1])
    width = int(parts[3])
    flip_y = parts[0].startswith("-")

    scanlines = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        if pos + 4 > n:
            break
        b0, b1, b2, b3 = data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
        if b0 == 2 and b1 == 2 and ((b2 << 8) | b3) == width and width >= 8 and width <= 0x7FFF:
            pos += 4
            row = np.zeros((width, 4), dtype=np.uint8)
            for ch in range(4):
                x = 0
                while x < width and pos < n:
                    cnt = data[pos]
                    pos += 1
                    if cnt > 128:
                        run = cnt - 128
                        val = data[pos]
                        pos += 1
                        row[x : x + run, ch] = val
                        x += run
                    else:
                        run = cnt
                        row[x : x + run, ch] = np.frombuffer(data, dtype=np.uint8, count=run, offset=pos)
                        pos += run
                        x += run
        else:
            row = np.frombuffer(data, dtype=np.uint8, count=width * 4, offset=pos).reshape(width, 4)
            pos += width * 4
        dst_y = (height - 1 - y) if flip_y else y
        scanlines[dst_y] = row

    rgbe = scanlines.astype(np.float32)
    exp = rgbe[:, :, 3]
    scale = np.where(exp > 0.0, np.ldexp(1.0, exp.astype(np.int32) - 136), 0.0)
    rgb = rgbe[:, :, :3] * scale[:, :, None]
    # Reinhard tone-mapping + gamma 2.2
    tm = rgb / (1.0 + rgb)
    srgb = np.power(np.clip(tm, 0.0, 1.0), 1.0 / 2.2)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(srgb * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba[:, :, 3] = 255
    return width, height, rgba.reshape(-1)


class HDRLoader:
    """Radiance .hdr loader (three/addons/loaders/HDRLoader.js & RGBELoader.js)."""

    _registry = {}

    def __init__(self, manager=None):
        self.manager = manager

    @classmethod
    def register(cls, name, data):
        if isinstance(data, (bytes, bytearray, memoryview)):
            cls._registry[str(name)] = bytes(data)
            return True
        return False

    def setDataType(self, dataType):
        self.type = dataType
        return self

    def setPath(self, path):
        return self

    def load(self, name, onLoad=None, onProgress=None, onError=None):
        manager = _manager_of(self)
        manager.itemStart(str(name))
        try:
            raw = self._registry.get(str(name))
            if raw is None:
                raw = self._registry.get(str(name).rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
            if raw is None:
                raise ValueError(f"HDRLoader: no dropped HDR file named {str(name)!r}")
            w, h, rgba = _decode_radiance_hdr(raw)
            tex = DataTexture(rgba, w, h, RGBAFormat)
            tex.needsUpdate = True
            if callable(onLoad):
                onLoad(tex)
            return tex
        except Exception as exc:
            if callable(onError):
                onError(exc)
                return None
            raise
        finally:
            manager.itemEnd(str(name))

    def loadAsync(self, name, onProgress=None):
        try:
            return SyncPromise.resolved(self.load(name, onProgress=onProgress))
        except Exception as exc:
            return SyncPromise.rejected(exc)


RGBELoader = HDRLoader


_PLY_TYPE_MAP = {
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}


def _parse_ply_bytes(raw):
    """Parse ASCII or binary PLY bytes into a BufferGeometry."""
    import struct as _struct
    data = bytes(raw)
    hdr_end = data.find(b"end_header\n")
    hdr_len = 11
    if hdr_end < 0:
        hdr_end = data.find(b"end_header\r\n")
        hdr_len = 12
    if hdr_end < 0:
        raise ValueError("PLYLoader: missing end_header")
    header_lines = data[:hdr_end].decode("ascii", errors="replace").splitlines()
    body = data[hdr_end + hdr_len :]

    fmt = "ascii"
    elements = []
    cur_elem = None
    for line in header_lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            cur_elem = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(cur_elem)
        elif parts[0] == "property" and cur_elem is not None:
            if parts[1] == "list":
                cur_elem["props"].append(("list", parts[2], parts[3], parts[4]))
            else:
                cur_elem["props"].append(("scalar", parts[1], parts[2]))

    positions = []
    normals = []
    colors = []
    indices = []

    if fmt == "ascii":
        text_lines = body.decode("ascii", errors="replace").splitlines()
        li = 0
        for el in elements:
            cnt = el["count"]
            if el["name"] == "vertex":
                pnames = [p[2] for p in el["props"]]
                ix = pnames.index("x") if "x" in pnames else -1
                iy = pnames.index("y") if "y" in pnames else -1
                iz = pnames.index("z") if "z" in pnames else -1
                inx = pnames.index("nx") if "nx" in pnames else -1
                iny = pnames.index("ny") if "ny" in pnames else -1
                inz = pnames.index("nz") if "nz" in pnames else -1
                ir = pnames.index("red") if "red" in pnames else -1
                ig = pnames.index("green") if "green" in pnames else -1
                ib = pnames.index("blue") if "blue" in pnames else -1
                for _ in range(cnt):
                    while li < len(text_lines) and not text_lines[li].strip():
                        li += 1
                    vals = text_lines[li].split()
                    li += 1
                    if ix >= 0 and iy >= 0 and iz >= 0:
                        positions.append((float(vals[ix]), float(vals[iy]), float(vals[iz])))
                    if inx >= 0 and iny >= 0 and inz >= 0:
                        normals.append((float(vals[inx]), float(vals[iny]), float(vals[inz])))
                    if ir >= 0 and ig >= 0 and ib >= 0:
                        colors.append((float(vals[ir]) / 255.0, float(vals[ig]) / 255.0, float(vals[ib]) / 255.0))
            elif el["name"] == "face":
                for _ in range(cnt):
                    while li < len(text_lines) and not text_lines[li].strip():
                        li += 1
                    vals = text_lines[li].split()
                    li += 1
                    k = int(vals[0])
                    if k >= 3:
                        v0 = int(vals[1])
                        for j in range(1, k - 1):
                            indices.extend([v0, int(vals[1 + j]), int(vals[2 + j])])
            else:
                li += cnt
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        pos = 0
        for el in elements:
            cnt = el["count"]
            if el["name"] == "vertex":
                pnames = [p[2] for p in el["props"]]
                sfmt = endian + "".join(_PLY_TYPE_MAP[p[1]][0] for p in el["props"])
                st = _struct.Struct(sfmt)
                ix = pnames.index("x") if "x" in pnames else -1
                iy = pnames.index("y") if "y" in pnames else -1
                iz = pnames.index("z") if "z" in pnames else -1
                inx = pnames.index("nx") if "nx" in pnames else -1
                iny = pnames.index("ny") if "ny" in pnames else -1
                inz = pnames.index("nz") if "nz" in pnames else -1
                ir = pnames.index("red") if "red" in pnames else -1
                ig = pnames.index("green") if "green" in pnames else -1
                ib = pnames.index("blue") if "blue" in pnames else -1
                for _ in range(cnt):
                    vals = st.unpack_from(body, pos)
                    pos += st.size
                    if ix >= 0 and iy >= 0 and iz >= 0:
                        positions.append((float(vals[ix]), float(vals[iy]), float(vals[iz])))
                    if inx >= 0 and iny >= 0 and inz >= 0:
                        normals.append((float(vals[inx]), float(vals[iny]), float(vals[inz])))
                    if ir >= 0 and ig >= 0 and ib >= 0:
                        colors.append((float(vals[ir]) / 255.0, float(vals[ig]) / 255.0, float(vals[ib]) / 255.0))
            elif el["name"] == "face":
                for _ in range(cnt):
                    for prop in el["props"]:
                        if prop[0] == "list":
                            cfmt, csz = _PLY_TYPE_MAP[prop[1]]
                            vfmt, vsz = _PLY_TYPE_MAP[prop[2]]
                            k = _struct.unpack_from(endian + cfmt, body, pos)[0]
                            pos += csz
                            verts = _struct.unpack_from(endian + vfmt * k, body, pos)
                            pos += vsz * k
                            if prop[3] in ("vertex_indices", "vertex_index") and k >= 3:
                                v0 = int(verts[0])
                                for j in range(1, k - 1):
                                    indices.extend([v0, int(verts[j]), int(verts[j + 1])])
                        else:
                            _, psz = _PLY_TYPE_MAP[prop[1]]
                            pos += psz
            else:
                for _ in range(cnt):
                    for prop in el["props"]:
                        if prop[0] == "list":
                            cfmt, csz = _PLY_TYPE_MAP[prop[1]]
                            _, vsz = _PLY_TYPE_MAP[prop[2]]
                            k = _struct.unpack_from(endian + cfmt, body, pos)[0]
                            pos += csz + vsz * k
                        else:
                            _, psz = _PLY_TYPE_MAP[prop[1]]
                            pos += psz

    geo = BufferGeometry()
    if positions:
        geo.attributes["position"] = np.asarray(positions, dtype=np.float32)
    if normals:
        geo.attributes["normal"] = np.asarray(normals, dtype=np.float32)
    if colors:
        geo.attributes["color"] = np.asarray(colors, dtype=np.float32)
    if indices:
        geo.index = np.asarray(indices, dtype=np.int32)
    if not normals and positions:
        geo.computeVertexNormals()
    return geo


class PLYLoader:
    """Stanford PLY loader (three/addons/loaders/PLYLoader.js)."""

    _registry = {}

    def __init__(self, manager=None):
        self.manager = manager

    @classmethod
    def register(cls, name, data):
        if isinstance(data, str):
            cls._registry[str(name)] = data.encode("utf-8")
            return True
        if isinstance(data, (bytes, bytearray, memoryview)):
            cls._registry[str(name)] = bytes(data)
            return True
        return False

    def parse(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        return _parse_ply_bytes(data)

    def load(self, name, onLoad=None, onProgress=None, onError=None):
        manager = _manager_of(self)
        manager.itemStart(str(name))
        try:
            raw = self._registry.get(str(name))
            if raw is None:
                raw = self._registry.get(str(name).rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
            if raw is None:
                raise ValueError(f"PLYLoader: no dropped PLY file named {str(name)!r}")
            geo = _parse_ply_bytes(raw)
            if callable(onLoad):
                onLoad(geo)
            return geo
        except Exception as exc:
            if callable(onError):
                onError(exc)
                return None
            raise
        finally:
            manager.itemEnd(str(name))

    def loadAsync(self, name, onProgress=None):
        try:
            return SyncPromise.resolved(self.load(name, onProgress=onProgress))
        except Exception as exc:
            return SyncPromise.rejected(exc)


class Face3:
    """Legacy three.js Face3 for Geometry."""

    isFace3 = True

    def __init__(self, a=0, b=0, c=0, normal=None, color=None, materialIndex=0):
        self.a = int(a)
        self.b = int(b)
        self.c = int(c)
        if isinstance(normal, list):
            self.normal = Vector3()
            self.vertexNormals = normal
        else:
            self.normal = normal if normal is not None else Vector3()
            self.vertexNormals = []
        if isinstance(color, list):
            self.color = Color(0xFFFFFF)
            self.vertexColors = color
        else:
            self.color = _as_color(color, Color(0xFFFFFF))
            self.vertexColors = []
        self.materialIndex = int(materialIndex)

    def clone(self):
        f = Face3(self.a, self.b, self.c, self.normal.clone(), self.color.clone(), self.materialIndex)
        f.vertexNormals = [n.clone() for n in self.vertexNormals]
        f.vertexColors = [c.clone() for c in self.vertexColors]
        return f


class Geometry(BufferGeometry):
    """Legacy three.js Geometry with vertices, faces, faceVertexUvs, and
    automatic sync to BufferGeometry attributes for render.py."""

    isGeometry = True

    def __init__(self):
        super().__init__()
        self.vertices = []
        self.colors = []
        self.faces = []
        self.faceVertexUvs = [[]]

    def computeFaceNormals(self):
        for f in self.faces:
            if f.a < len(self.vertices) and f.b < len(self.vertices) and f.c < len(self.vertices):
                va = self.vertices[f.a]
                vb = self.vertices[f.b]
                vc = self.vertices[f.c]
                cb = Vector3().subVectors(vc, vb)
                ab = Vector3().subVectors(va, vb)
                cb.cross(ab).normalize()
                f.normal.copy(cb)
        return self

    def computeVertexNormals(self, areaWeighted=True):
        verts = [Vector3() for _ in range(len(self.vertices))]
        if areaWeighted:
            for f in self.faces:
                if f.a < len(self.vertices) and f.b < len(self.vertices) and f.c < len(self.vertices):
                    va, vb, vc = self.vertices[f.a], self.vertices[f.b], self.vertices[f.c]
                    cb = Vector3().subVectors(vc, vb)
                    ab = Vector3().subVectors(va, vb)
                    cb.cross(ab)
                    verts[f.a].add(cb)
                    verts[f.b].add(cb)
                    verts[f.c].add(cb)
        else:
            self.computeFaceNormals()
            for f in self.faces:
                for idx in (f.a, f.b, f.c):
                    if idx < len(verts):
                        verts[idx].add(f.normal)
        for v in verts:
            v.normalize()
        for f in self.faces:
            f.vertexNormals = [
                verts[f.a].clone() if f.a < len(verts) else Vector3(0, 1, 0),
                verts[f.b].clone() if f.b < len(verts) else Vector3(0, 1, 0),
                verts[f.c].clone() if f.c < len(verts) else Vector3(0, 1, 0),
            ]
        return self

    def mergeVertices(self, precisionPoints=4):
        return 0

    def _sync_buffers(self):
        if not self.faces:
            return
        pos = []
        nrm = []
        col = []
        uv = []
        has_uv = bool(self.faceVertexUvs and self.faceVertexUvs[0])
        uv_list = self.faceVertexUvs[0] if has_uv else []
        has_col = bool(self.colors) or any(
            getattr(f, "vertexColors", None) or (f.color and (f.color.r, f.color.g, f.color.b) != (1.0, 1.0, 1.0))
            for f in self.faces
        )
        for fi, f in enumerate(self.faces):
            idxs = (f.a, f.b, f.c)
            if any(i < 0 or i >= len(self.vertices) for i in idxs):
                continue
            va, vb, vc = self.vertices[f.a], self.vertices[f.b], self.vertices[f.c]
            pos.extend([(va.x, va.y, va.z), (vb.x, vb.y, vb.z), (vc.x, vc.y, vc.z)])
            vns = getattr(f, "vertexNormals", None)
            if vns and len(vns) >= 3:
                nrm.extend([(vns[0].x, vns[0].y, vns[0].z), (vns[1].x, vns[1].y, vns[1].z), (vns[2].x, vns[2].y, vns[2].z)])
            elif f.normal is not None and f.normal.lengthSq() > 0:
                nn = (f.normal.x, f.normal.y, f.normal.z)
                nrm.extend([nn, nn, nn])
            else:
                cb = Vector3().subVectors(vc, vb)
                ab = Vector3().subVectors(va, vb)
                cb.cross(ab).normalize()
                nn = (cb.x, cb.y, cb.z)
                nrm.extend([nn, nn, nn])
            if has_col:
                vcs = getattr(f, "vertexColors", None)
                if vcs and len(vcs) >= 3:
                    col.extend([(vcs[0].r, vcs[0].g, vcs[0].b), (vcs[1].r, vcs[1].g, vcs[1].b), (vcs[2].r, vcs[2].g, vcs[2].b)])
                elif len(self.colors) == len(self.vertices):
                    ca, cb2, cc = self.colors[f.a], self.colors[f.b], self.colors[f.c]
                    col.extend([(ca.r, ca.g, ca.b), (cb2.r, cb2.g, cb2.b), (cc.r, cc.g, cc.b)])
                else:
                    fc = f.color if f.color is not None else Color(0xFFFFFF)
                    col.extend([(fc.r, fc.g, fc.b)] * 3)
            if has_uv and fi < len(uv_list) and len(uv_list[fi]) >= 3:
                u3 = uv_list[fi]
                uv.extend([(u3[0].x, u3[0].y), (u3[1].x, u3[1].y), (u3[2].x, u3[2].y)])
            elif has_uv:
                uv.extend([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
        self.attributes["position"] = np.asarray(pos, dtype=np.float32)
        self.attributes["normal"] = np.asarray(nrm, dtype=np.float32)
        if has_col:
            self.attributes["color"] = np.asarray(col, dtype=np.float32)
        if has_uv and uv:
            self.attributes["uv"] = np.asarray(uv, dtype=np.float32)
        self._index = None


class ParametricGeometry(BufferGeometry):
    """three.js ParametricGeometry(func, slices, stacks)."""

    def __init__(self, func=None, slices=8, stacks=8):
        super().__init__()
        sl = max(1, int(slices))
        st = max(1, int(stacks))
        if not callable(func):
            return
        positions = []
        uvs = []
        p = Vector3()
        for i in range(st + 1):
            v = i / st
            for j in range(sl + 1):
                u = j / sl
                func(u, v, p)
                positions.append((p.x, p.y, p.z))
                uvs.append((u, v))
        indices = []
        stride = sl + 1
        for i in range(st):
            for j in range(sl):
                a = i * stride + j
                b = i * stride + j + 1
                c = (i + 1) * stride + j + 1
                d = (i + 1) * stride + j
                indices.extend([a, b, d, b, c, d])
        self.attributes["position"] = np.asarray(positions, dtype=np.float32)
        self.attributes["uv"] = np.asarray(uvs, dtype=np.float32)
        self.index = np.asarray(indices, dtype=np.int32)
        self.computeVertexNormals()


class _SkeletonUtils:
    @staticmethod
    def clone(source):
        if source is None:
            return None
        if hasattr(source, "clone"):
            return source.clone(True)
        return source


SkeletonUtils = _SkeletonUtils()
cloneSkeleton = _SkeletonUtils.clone


class _RectAreaLightUniformsLib:
    @staticmethod
    def init():
        return None


RectAreaLightUniformsLib = _RectAreaLightUniformsLib()


__all__ += [
    "MeshDepthMaterial",
    "HDRLoader",
    "RGBELoader",
    "PLYLoader",
    "Face3",
    "Geometry",
    "ParametricGeometry",
    "PolyhedronGeometry",
    "SkeletonUtils",
    "RectAreaLightUniformsLib",
    "InstancedBufferGeometry",
    "WebGLCubeRenderTarget",
    "ArrayCamera",
    "FileLoader",
    "UniformsGroup",
    "RoomEnvironment",
    "ImprovedNoise",
    "CubeCamera",
    "LOD",
    "InterleavedBuffer",
    "InterleavedBufferAttribute",
    "Data3DTexture",
    "BufferGeometryLoader",
    "LightProbe",
    "Sky",
    "AlwaysStencilFunc",
    "HalfFloatType",
    "Uniform",
    "UniformsUtils",
    "InstancedBufferAttribute",
    "LineDashedMaterial",
    "VideoTexture",
    "SkeletonHelper",
    "RectAreaLight",
    "WireframeGeometry",
    "LineLoop",
    "CapsuleGeometry",
    "MeshToonMaterial",
    "MeshMatcapMaterial",
    "ColorManagement",
    "AnimationUtils",
    "ShaderChunk",
]


__all__ += [
    "ShaderMaterial",
    "RawShaderMaterial",
    "WebGLRenderTarget",
    "Pass",
    "RenderPass",
    "ShaderPass",
    "OutputPass",
    "CopyShader",
    "EffectComposer",
    "EdgesGeometry",
    "InstancedMesh",
    "FogExp2",
    "Vector4",
    "Spherical",
    "Layers",
    "DepthTexture",
    "PMREMGenerator",
    "PlaneHelper",
    "MeshPhysicalMaterial",
    "ShadowMaterial",
    "Int8BufferAttribute",
    "Uint8BufferAttribute",
    "Uint8ClampedBufferAttribute",
    "Int16BufferAttribute",
    "Uint16BufferAttribute",
    "Int32BufferAttribute",
    "Uint32BufferAttribute",
]

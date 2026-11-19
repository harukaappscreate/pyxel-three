"""three.js on Pyxel — host application.

Runs three.js-style JavaScript source on stock Pyxel:
  dropped JS text -> jsinterp (from-scratch JS interpreter)
                  -> three.py (from-scratch three.js core subset)
                  -> render.py (numpy software rasterizer)
                  -> Pyxel screen

- On startup the built-in sample (the official three.js introductory
  rotating cube) runs immediately, so the app works on a phone with no
  file drop.
- Drop a .js / .txt file (PC) to run that source instead.
- Drop a .html file (PC) to run the inline script code inside it —
  <script type="module"> blocks for the official three.js examples
  (single .html files, dropped as-is), or, when a page has no module
  block, its inline classic scripts (the CodePen export form).
  Assets they name (textures, models) resolve by file name against
  files dropped with them.
- The gamepad A button restarts the current source.  The mouse is
  input for the JS program (OrbitControls drags, click listeners), so
  a tap restarts only while an error is on screen (recovery).
- Pointer input (mouse; touch arrives as mouse on Pyxel) is bridged
  into the program every frame: OrbitControls polls it through
  three.set_pointer_state, and pointerdown/click/pointerup/
  pointermove are dispatched to JS event listeners.
- JS errors are shown on screen; the host never crashes on them.
"""

import json
import re
from pathlib import Path

import numpy as np
import pyxel

import render
import three
import jsinterp
from jsinterp import JSInterpreter, JSError, JSObject

try:  # PIL decodes dropped images for TextureLoader
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover - PIL unavailable on this runtime
    _PILImage = None

WIDTH = 256
HEIGHT = 192
TITLE = "three.js on Pyxel"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
MODEL_EXTS = (".glb", ".gltf")
OBJ_EXTS = (".obj", ".mtl")
HTML_EXTS = (".html", ".htm")

# The official three.js introductory sample (threejs.org manual,
# "Creating a scene"), used verbatim as the built-in demo.
SAMPLE_JS = """
import * as THREE from 'three';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

const geometry = new THREE.BoxGeometry( 1, 1, 1 );
const material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
const cube = new THREE.Mesh( geometry, material );
scene.add( cube );

camera.position.z = 5;

function animate() {
	requestAnimationFrame( animate );

	cube.rotation.x += 0.01;
	cube.rotation.y += 0.01;

	renderer.render( scene, camera );
}

animate();
"""

# ---------------------------------------------------------------------------
# Renderer tracking: the JS program creates its own WebGLRenderer inside
# the interpreter, so the host subclasses it (before any interpreter is
# created) to keep a registry of live instances and read their frames.
# ---------------------------------------------------------------------------
_renderers = []
_RealWebGLRenderer = three.WebGLRenderer


class _TrackingRenderer(_RealWebGLRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _renderers.append(self)


three.WebGLRenderer = _TrackingRenderer

_HEX = np.array([ord(c) for c in "0123456789abcdef"], dtype=np.uint8)


def _frame_to_rows(frame):
    """Convert a (h, w) palette-index array to Image.set hex-string rows,
    cropped/padded to the screen size."""
    out = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    if frame is not None:
        f = np.asarray(frame)
        if f.ndim == 2 and f.size:
            h = min(HEIGHT, f.shape[0])
            w = min(WIDTH, f.shape[1])
            out[:h, :w] = np.clip(f[:h, :w], 0, 15)
    chars = _HEX[out]
    return [chars[y].tobytes().decode("ascii") for y in range(HEIGHT)]


# ---------------------------------------------------------------------------
# HTML drop support: official three.js examples are single .html files
# whose program lives in inline <script type="module"> blocks.  Dropping
# the .html extracts those blocks (in document order) and runs the
# concatenation through the same path as a dropped .js file.  When a
# document has no module block at all (CodePen exports: three.js comes
# from a CDN <script src> and the pen body is a classic inline script
# written against the global THREE), the inline classic scripts are
# extracted instead.  A script body containing the literal text
# "</script>" would end its block early — the same rule a browser
# applies — and does not occur in the official examples.
# ---------------------------------------------------------------------------
_SCRIPT_RE = re.compile(
    r"<script\b([^>]*)>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL
)


def _script_attr(attrs, name):
    """Value of one attribute in a <script> tag's attribute text, or
    None when the attribute is absent."""
    m = re.search(
        name + r"""\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""",
        attrs,
        re.IGNORECASE,
    )
    if m is None:
        return None
    for group in m.groups():
        if group is not None:
            return group
    return ""


# type values of a classic (non-module) JavaScript script block; a
# missing type attribute also means a classic script.
_PLAIN_JS_TYPES = {
    "",
    "text/javascript",
    "application/javascript",
    "text/ecmascript",
    "application/ecmascript",
}


def extract_inline_scripts(html):
    """Extract the runnable inline script source of an HTML document.

    Primary form: the inline <script type="module"> blocks, concatenated
    in document order (the official three.js examples).  When the
    document has no inline module block, fall back to the inline classic
    scripts (no type, or a classic JavaScript type) in document order —
    the CodePen export form, whose body is written against the global
    THREE that this host already provides.  The two forms are never
    mixed: a document with any module block yields modules only.

    Scripts with a src attribute are external files and are always
    skipped (the host has no way to fetch them); other typed blocks
    (importmap, JSON, ...) are not code and are ignored.  Returns ""
    when the document has no usable inline script."""
    modules = []
    plains = []
    for m in _SCRIPT_RE.finditer(html):
        attrs, body = m.group(1), m.group(2)
        if _script_attr(attrs, "src") is not None:
            continue
        stype = (_script_attr(attrs, "type") or "").strip().lower()
        if stype == "module":
            modules.append(body)
        elif stype in _PLAIN_JS_TYPES:
            plains.append(body)
    if modules:
        return "\n".join(modules)
    return "\n".join(plains)


def extract_html_elements(html):
    """Extract {id: textContent} for inline elements with an id attribute
    (e.g. <script id="vertexShader" type="x-shader/x-vertex">...</script>)."""
    out = {}
    for m in _SCRIPT_RE.finditer(html):
        attrs, body = m.group(1), m.group(2)
        eid = _script_attr(attrs, "id")
        if eid:
            out[eid] = body
    return out


def extract_importmap(html):
    """Return the "imports" dict of the first <script type="importmap">
    block ({} when there is none or it is malformed).  The interpreter
    resolves the 'three' and 'three/addons/...' specifiers itself, so
    the map is read for the record and otherwise unused; code that
    imports a mapped URL directly stays out of scope, as before."""
    for m in _SCRIPT_RE.finditer(html):
        attrs, body = m.group(1), m.group(2)
        if (_script_attr(attrs, "type") or "").strip().lower() != "importmap":
            continue
        try:
            data = json.loads(body)
        except Exception:
            return {}
        imports = data.get("imports") if isinstance(data, dict) else None
        return dict(imports) if isinstance(imports, dict) else {}
    return {}


class App:
    def __init__(self, run=True, init_pyxel=True):
        self._screen_buf = None
        if init_pyxel:
            pyxel.init(WIDTH, HEIGHT, title=TITLE)
            cubes = render.cube_colours()
            pyxel.colors[16 : 16 + len(cubes)] = cubes
            self._screen_buf = np.ctypeslib.as_array(
                pyxel.screen.data_ptr()
            ).reshape(HEIGHT, WIDTH)
        # Built-in procedural checker texture, resolvable from JS as
        # new THREE.TextureLoader().load('checker').
        try:
            three.TextureLoader.register(
                "checker", three.checker_texture_image()
            )
        except Exception:
            pass
        self.interp = None
        self.source = SAMPLE_JS
        self.html_elements = {}
        self.importmap = {}
        self.error = None
        self.frame = None
        self._prev_left = False
        self._prev_xy = (0, 0)
        self.load_source(SAMPLE_JS)
        if run:
            pyxel.run(self.update, self.draw)

    # -- source management --------------------------------------------------
    def load_source(self, source, html_elements=None):
        self.source = source
        if html_elements is not None:
            self.html_elements = dict(html_elements)
        del _renderers[:]
        self.error = None
        self.frame = None
        self._prev_left = False
        try:
            interp = JSInterpreter(three, html_elements=self.html_elements)
            interp.run(source)
            self.interp = interp
        except JSError as exc:
            self.interp = None
            self.error = str(exc)
        except Exception as exc:  # the host must survive any JS failure
            self.interp = None
            self.error = f"{type(exc).__name__}: {exc}"
        self._grab_frame()

    def _current_renderer(self):
        if _renderers:
            return _renderers[-1]
        if self.interp is not None:
            try:
                return self.interp.global_env.lookup("renderer")
            except Exception:
                return None
        return None

    def _grab_frame(self):
        renderer = self._current_renderer()
        if renderer is not None:
            frame = getattr(renderer, "frame", None)
            if frame is not None:
                self.frame = frame

    # -- file drop ------------------------------------------------------------
    def _check_drop(self):
        paths = list(getattr(pyxel, "dropped_files", None) or [])
        for p in getattr(pyxel, "_dropped_files", None) or []:
            if p not in paths:
                paths.append(p)
        if not paths:
            return
        try:
            pyxel._dropped_files = []
        except Exception:
            pass
        js_source = None
        html_elements = {}
        for path in paths:
            if not isinstance(path, str):
                continue
            low = path.lower()
            if low.endswith(IMAGE_EXTS):
                self._register_image(path)
            elif low.endswith(MODEL_EXTS) or low.endswith(".bin"):
                # A .bin next to its .gltf is the model's external
                # buffer: it goes into the same GLTFLoader registry,
                # where gltf.py resolves buffer URIs by name.
                self._register_model(path)
            elif low.endswith(OBJ_EXTS):
                self._register_obj(path)
            elif low.endswith(".hdr"):
                try:
                    three.HDRLoader.register(Path(path).name, open(path, "rb").read())
                except Exception:
                    pass
            elif low.endswith(".ply"):
                try:
                    three.PLYLoader.register(Path(path).name, open(path, "rb").read())
                except Exception:
                    pass
            elif low.endswith(HTML_EXTS):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        html_text = f.read()
                except Exception as exc:
                    self.error = f"drop read error: {exc}"
                    return
                # The importmap is parsed for the record; the
                # interpreter resolves 'three' / 'three/addons/...'
                # specifiers itself, so nothing else is done with it.
                self.importmap = extract_importmap(html_text)
                html_elements = extract_html_elements(html_text)
                extracted = extract_inline_scripts(html_text)
                if not extracted.strip():
                    self.error = (
                        "dropped HTML has no usable inline <script>"
                    )
                    return
                js_source = extracted
            elif low.endswith((".js", ".txt")):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        js_source = f.read()
                except Exception as exc:
                    self.error = f"drop read error: {exc}"
                    return
        if js_source is not None:
            self.load_source(js_source, html_elements=html_elements)

    def _register_image(self, path):
        """Decode a dropped image with PIL and register it by file name
        so JS can resolve it via TextureLoader.load(name)."""
        if _PILImage is None:
            return
        try:
            with _PILImage.open(path) as im:
                arr = np.asarray(im.convert("RGB"))
            three.TextureLoader.register(Path(path).name, arr)
        except Exception:
            pass  # an undecodable image must not take the host down

    def _register_model(self, path):
        """Register a dropped .glb / .gltf file's raw bytes by file name
        so JS can load it via GLTFLoader.load(name).  Models
        dropped together with a .js file are registered before the JS
        runs, because load_source happens after this loop."""
        try:
            with open(path, "rb") as f:
                data = f.read()
            three.GLTFLoader.register(Path(path).name, data)
        except Exception:
            pass  # an unreadable model must not take the host down

    def _register_obj(self, path):
        """Register a dropped .obj / .mtl file's text by file name so
        JS can load it via OBJLoader.load(name).  The .mtl
        is found by the OBJ parser through the same registry when the
        .obj names it in an mtllib line.  Registration happens before
        any dropped JS runs, like _register_model."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            three.OBJLoader.register(Path(path).name, text)
        except Exception:
            pass  # an unreadable model must not take the host down

    # -- input bridge ----------------------------------------------
    def _forward_input(self):
        """Sample Pyxel's pointer once per frame and bridge it into the
        JS program: OrbitControls polls the state via
        three.set_pointer_state, and press/release/move edges are
        dispatched as synthetic DOM-style events to JS listeners on
        window, document and each renderer's domElement.  Touch input
        reaches Pyxel as mouse input, so it takes the same path."""
        x = int(pyxel.mouse_x)
        y = int(pyxel.mouse_y)
        left = bool(pyxel.btn(pyxel.MOUSE_BUTTON_LEFT))
        right = bool(pyxel.btn(pyxel.MOUSE_BUTTON_RIGHT))
        try:
            wheel = float(pyxel.mouse_wheel)
        except Exception:
            wheel = 0.0
        three.set_pointer_state(x, y, left=left, right=right, wheel=wheel)
        if self.interp is None:
            self._prev_left = left
            self._prev_xy = (x, y)
            return
        event_types = []
        if left and not self._prev_left:
            event_types.append("pointerdown")
            event_types.append("click")
        if not left and self._prev_left:
            event_types.append("pointerup")
        if (x, y) != self._prev_xy:
            event_types.append("pointermove")
        self._prev_left = left
        self._prev_xy = (x, y)
        for etype in event_types:
            self._dispatch_pointer_event(etype, x, y)

    def _dispatch_pointer_event(self, etype, x, y):
        interp = self.interp
        if interp is None:
            return
        event = JSObject({
            "type": etype,
            "clientX": float(x),
            "clientY": float(y),
            "offsetX": float(x),
            "offsetY": float(y),
            "button": 0.0,
            "preventDefault": lambda: None,
        })
        targets = []
        for name in ("window", "document"):
            try:
                targets.append(interp.global_env.lookup(name))
            except Exception:
                pass
        for renderer in list(_renderers):
            element = getattr(renderer, "domElement", None)
            if element is not None:
                targets.append(element)
        for target in targets:
            interp.dispatch_event(target, event)

    # -- pyxel callbacks ------------------------------------------------------
    def update(self):
        self._check_drop()
        restart = pyxel.btnp(pyxel.GAMEPAD1_BUTTON_A)
        if self.error is not None and pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT):
            # Recovery tap: with an error on screen, a tap restarts.
            # (Otherwise the mouse belongs to the JS program.)
            restart = True
        if restart:
            self.load_source(self.source)
            return
        try:
            self._forward_input()
        except JSError as exc:
            self.error = str(exc)
            return
        except Exception as exc:  # never crash the host
            self.error = f"{type(exc).__name__}: {exc}"
            return
        if self.interp is None or self.error is not None:
            return
        try:
            self.interp.call_raf()
            # Samples written with renderer.setAnimationLoop(cb) instead of
            # requestAnimationFrame are driven here, once per frame.
            for renderer in list(_renderers):
                cb = renderer.getAnimationLoop()
                if cb is not None:
                    cb()
        except JSError as exc:
            self.error = str(exc)
            return
        except Exception as exc:  # never crash the host
            self.error = f"{type(exc).__name__}: {exc}"
            return
        self._grab_frame()

    def draw(self):
        pyxel.cls(0)
        if self.frame is not None:
            if self._screen_buf is not None:
                f = np.asarray(self.frame)
                if f.ndim == 2 and f.size:
                    h = min(HEIGHT, f.shape[0])
                    w = min(WIDTH, f.shape[1])
                    self._screen_buf[:h, :w] = np.clip(f[:h, :w], 0, 225).astype(np.uint8)
            else:
                pyxel.images[0].set(0, 0, _frame_to_rows(self.frame))
                pyxel.blt(0, 0, 0, 0, 0, WIDTH, HEIGHT)
        if self.error:
            self._draw_error()

    def _draw_error(self):
        msg = "".join(c if 32 <= ord(c) < 127 else "?" for c in self.error)
        lines = []
        while msg and len(lines) < 6:
            lines.append(msg[:63])
            msg = msg[63:]
        y = HEIGHT - 8 * len(lines) - 2
        pyxel.rect(0, y - 2, WIDTH, HEIGHT - y + 2, 0)
        for i, line in enumerate(lines):
            pyxel.text(2, y + 8 * i, line, 8)


if __name__ == "__main__":
    App()

"""Vectorized GLSL ES 1.00 evaluator -- production version.

Executes a parsed GLSL program over a whole grid of lanes (all pixels of
a fragment stage, or all vertices of a vertex stage) at once with numpy.
This is the evaluator half of the port's GLSL layer; the parser half is
``glsl.py``.  No exec/eval/compile of generated code is
used anywhere -- the C&M public-code rules ban them, and walking the AST
is the whole point of this module.

The execution model follows the research prototype
(``glsl-research/glsl_eval.py``, whose numbers and correctness checks are
in ``glsl-research/report.md``) and the design study in
``glsl-research/report-design.md``:

  - Values are SoA (structure of arrays): a vecN is a tuple of N
    per-component lane arrays, never an (..., N) block.  This matches
    render.py's existing layout and makes swizzles free.
  - Uniform scalars stay Python scalars end to end ("scalar routing"):
    they are never expanded to full lane arrays, and operations on them
    cost scalar time.  An ``if`` whose condition is a scalar executes
    only the taken branch; a loop whose condition/bounds stay scalar
    runs as a plain host loop with no mask bookkeeping.
  - Lane-varying control flow is PREDICATED (GPU SIMT style): both
    branches of an ``if`` execute and stores merge under a mask;
    break/continue/return/discard invalidate lanes through masks.
    Signals that happen under "all lanes live" (mask is None) take a
    fast path through Python exceptions instead.
  - Functions are copy-in/copy-out over the full lane set; ``out`` /
    ``inout`` parameters are copied back under the caller's mask.
    Overload resolution is by argument type signature, with the GLSL
    implicit int -> float promotion.  GLSL forbids recursion; a depth
    guard turns it into an error instead of a hang.
  - texture2D is NEAREST sampling only (bilinear is out of scope; a texture's "filter" entry is ignored).  LOD arguments are
    accepted and ignored (LOD 0).  dFdx/dFdy/fwidth are neighbour
    differences on the lane grid, which is why fragment lanes must keep
    the (H, W) grid shape.

=====================================================================
AST FORMAT (the contract with glsl.py)
=====================================================================
Tuples with a string tag first, exactly the research prototype's style.
Type names stay strings at this level ("float", "vec3", "mat4",
"sampler2D", or a struct name); the evaluator maps them to internal
type descriptors.  Qualifier lists are lists of strings (possibly
empty), e.g. ["uniform"], ["const"], ["in", "out"] -> for parameters
["out"] / ["inout"] / [].

Program = list of items:
  ("structdef", name, [(field_type_name, field_name), ...])
  ("global", quals, type_name, name, init_expr | None,
             array_size_expr | None)
  ("func", return_type_name, name, params, body_stmt)
      params = [(quals, type_name, param_name | None,
                 array_size_expr | None), ...]
      (a 3-tuple (quals, type_name, name) is also accepted)

Statements:
  ("block", [stmt, ...])
  ("decl", quals, type_name,
           [(name, init_expr | None, array_size_expr | None), ...])
  ("exprstmt", expr)
  ("if", cond_expr, then_stmt, else_stmt | None)
  ("for", init_stmt | None, cond_expr | None, step_expr | None,
          body_stmt)         -- init_stmt is a "decl" or ("exprstmt", e)
  ("while", cond_expr, body_stmt)
  ("do", body_stmt, cond_expr)
  ("return", expr | None)
  ("break",)   ("continue",)   ("discard",)
  ("empty",)                   -- a bare ";"

Expressions:
  ("num", python_value, "float" | "int" | "bool")
  ("ident", name)
  ("bin", op, lhs, rhs)
      op in + - * / % == != < > <= >= && || & | ^ << >>
  ("unary", op, expr)          op in - + ! ~ ++ --
  ("postfix", op, expr)        op in ++ --
  ("assign", op, lhs, rhs)     op in = += -= *= /= %= &= |= ^= <<= >>=
  ("ternary", cond, then_expr, else_expr)
  ("call", ("ident", name), [arg_expr, ...])
      -- user functions, builtin functions, and constructors
         (callee name is a type name for constructors)
  ("field", base_expr, name)   -- struct member or swizzle
  ("index", base_expr, index_expr)
  ("comma", expr, expr)        -- evaluates left for side effects

Array sizes are constant expressions; they are evaluated in the scope
of the globals declared so far and must reduce to a scalar int.

Contract adjustments for the G1 parser's output: top-level
("precision", ...) and ("funcdecl", ...) items are accepted and
skipped; structdef fields may arrive as (type, name, arr) triples and
are normalised to pairs; the do-while statement may be spelled
("dowhile", body, cond); array constructors arrive as
("arrctor", type_name, size_expr | None, [arg, ...]) and an
("unsized",) array marker takes its length from the constructor.

=====================================================================
VALUE REPRESENTATION (what the host passes in and gets back)
=====================================================================
A value is a pair (data, type).  Internal type descriptors:
  "float" "int" "bool" "void" "sampler" "samplerCube"
  ("vec", n) ("ivec", n) ("bvec", n)   -- float / int / bool vectors
  ("mat", n)                           -- square float matrices
  ("array", elem_type, length)
  ("struct", name)

Data layout per type (S = lane shape, e.g. (192, 256) for a fragment
grid or (nverts,) for a vertex stage):
  float/int/bool : a Python scalar (uniform value) OR an ndarray of
                   shape S (float32 / int32 / bool per lane)
  vec/ivec/bvec  : tuple of n components, each a scalar or an S-array
  mat            : tuple of n columns; each column is a tuple of n
                   float components (GLSL column-major order)
  array          : list of values, one per element
  struct         : dict {field_name: value}
  sampler        : a texture dict {"data": ndarray (h, w, 3|4),
                   "wrap": "repeat" | "clamp"} -- a bare ndarray is
                   also accepted (repeat wrap).  Row 0 of "data" is the
                   TOP of the image; uv (0, 0) is bottom-left, as in GL
                   with flipY textures (the render.py convention).

As a bridge convenience, host inputs may also arrive AoS-style: a vec
as one ndarray of shape S + (n,), a mat as one ndarray of shape
S + (n, n) in [row][col] layout.  run_main() converts those to SoA.
Type arguments in input pairs may be type-name strings ("vec3") or
internal descriptors.

Host entry point:

  ev = GLSLEvaluator(program, structs=None)
  result = ev.run_main((192, 256), inputs={
      "vUv":      ((u_array, v_array), ("vec", 2)),
      "tDiffuse": ({"data": img}, "sampler"),
      "opacity":  (0.85, "float"),
      "gl_FragCoord": ...,          # optional; auto-filled on 2D grids
  })
  result["globals"]["gl_FragColor"]  -> final (data, type)
  result["discard"]                  -> bool ndarray S, or None

Vertex stages use the same entry point with shape (nverts,); the host
reads gl_Position from the returned globals.  Derivatives are only
meaningful on a 2D lane grid and evaluate to 0 elsewhere.

Limitations of this subset (by design):
  - nearest texture sampling only; no mipmaps / LOD selection
  - square matrices only (mat2/3/4); no mat2x3 etc.
  - no dynamic (per-lane) writes through an index into a struct field
    chain deeper than field/index alternation handled generically --
    in practice all GLSL lvalue forms in the chunk vocabulary work
  - short-circuit && / || evaluate both sides (predication); GLSL
    restricts side effects in such expressions, so this is safe for
    well-formed shaders, at the cost of paying for both sides
"""

import numpy as np

F32 = np.float32
I32 = np.int32

__all__ = ["GLSLEvaluator", "GLSLError", "glsl_type", "evaluate"]


class GLSLError(Exception):
    """Any GLSL-level failure: unknown name, bad types, bad lvalue..."""


# ------------------------------------------------------------------ types
_VEC_KIND = {"vec": "float", "ivec": "int", "bvec": "bool"}
_KIND_VEC = {"float": "vec", "int": "ivec", "bool": "bvec"}


def glsl_type(name, structs=None):
    """Map a GLSL type-name string to an internal type descriptor."""
    if name in ("float", "int", "bool", "void", "sampler", "samplerCube"):
        return name
    if name == "sampler2D":
        return "sampler"
    if name == "samplerCube":
        return "samplerCube"
    for prefix, kind in (("vec", "vec"), ("ivec", "ivec"), ("bvec", "bvec"),
                         ("mat", "mat")):
        if name.startswith(prefix) and name[len(prefix):] in ("2", "3", "4"):
            return (kind, int(name[len(prefix):]))
    if structs and name in structs:
        return ("struct", name)
    raise GLSLError(f"unknown type name {name!r}")


def _is_vec(t):
    return isinstance(t, tuple) and t[0] in _VEC_KIND


def _is_mat(t):
    return isinstance(t, tuple) and t[0] == "mat"


def _is_array(t):
    return isinstance(t, tuple) and t[0] == "array"


def _is_struct(t):
    return isinstance(t, tuple) and t[0] == "struct"


def _kind(t):
    """Scalar kind ('float'/'int'/'bool') of a scalar or vector type."""
    if _is_vec(t):
        return _VEC_KIND[t[0]]
    return t


def _vec_type(kind, n):
    if n == 1:
        return kind
    return (_KIND_VEC[kind], n)


def _ncomp(t):
    if _is_vec(t):
        return t[1]
    return 1


SWIZ = {"x": 0, "y": 1, "z": 2, "w": 3,
        "r": 0, "g": 1, "b": 2, "a": 3,
        "s": 0, "t": 1, "p": 2, "q": 3}

_CONSTRUCTOR_TYPES = frozenset(
    ["float", "int", "bool"]
    + [f"{p}{n}" for p in ("vec", "ivec", "bvec", "mat")
       for n in (2, 3, 4)])


# ------------------------------------------------------------- flow signals
class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Discard(Exception):
    pass


# ------------------------------------------------------------ value helpers
def _is_field(x, S):
    """True if x is a per-lane array of shape S (not a scalar)."""
    return isinstance(x, np.ndarray) and x.shape == S


def _cast_scalar(x, kind):
    """Cast one component (scalar or lane array) to a GLSL scalar kind."""
    if kind == "float":
        if isinstance(x, np.ndarray):
            return x.astype(F32, copy=False)
        return float(x)
    if kind == "int":
        if isinstance(x, np.ndarray):
            return x.astype(I32, copy=False)
        return int(x)
    if isinstance(x, np.ndarray):
        return x.astype(bool, copy=False)
    return bool(x)


def _components(v):
    """Component list of a scalar/vector value's data."""
    d, t = v
    if _is_vec(t):
        return list(d)
    return [d]


def _rebuild(comps, t):
    """Inverse of _components for scalar/vector types."""
    if _is_vec(t):
        return (tuple(comps), t)
    return (comps[0], t)


def _promote(a, b):
    """Common scalar kind for a binary op (float wins over int)."""
    ka, kb = _kind(a[1]), _kind(b[1])
    if ka == "float" or kb == "float":
        return "float"
    if ka == "int" or kb == "int":
        return "int"
    return "bool"


_MAX_CALL_DEPTH = 64
_MAX_LOOP_ITER = 1000000


class GLSLEvaluator:
    """Executes one parsed GLSL program (see module docstring)."""

    def __init__(self, program, structs=None):
        self.structs = {}
        if structs:
            for name, fields in structs.items():
                self.structs[name] = list(fields)
        self.funcs = {}        # name -> [(ret_type, params, body), ...]
        self.global_decls = []  # (quals, type, name, init, arr_expr)
        for item in program:
            tag = item[0]
            if tag == "precision" or tag == "funcdecl":
                # Contract adjustment (G3): the G1 parser emits these
                # items; precision statements carry no semantics here,
                # and prototypes have no body to execute (a call to a
                # declared-but-undefined function fails at call time).
                continue
            if tag == "structdef":
                # The parser's fields are (type, name, arr) triples;
                # the evaluator's contract is (type, name) pairs.
                # Array-sized struct fields are not modelled (the
                # chunk vocabulary's structs have none).
                self.structs[item[1]] = [
                    (f[0], f[1]) for f in item[2]]
            elif tag == "func":
                _, rt_name, name, params, body = item
                norm = []
                for p in params:
                    if len(p) == 3:
                        quals, pt_name, pname = p
                        parr = None
                    else:
                        quals, pt_name, pname, parr = p
                    norm.append((list(quals), pt_name, pname, parr))
                self.funcs.setdefault(name, []).append(
                    (glsl_type(rt_name, self.structs), norm, body))
            elif tag == "global":
                _, quals, tname, name, init, arr = item
                self.global_decls.append(
                    (list(quals), tname, name, init, arr))
            else:
                raise GLSLError(f"unknown program item {tag!r}")
        self.S = None
        self.globals = {}
        self._depth = 0

    # ------------------------------------------------------------ defaults
    def _default(self, t):
        if t == "float":
            return (0.0, "float")
        if t == "int":
            return (0, "int")
        if t == "bool":
            return (False, "bool")
        if t in ("sampler", "samplerCube"):
            return (None, t)
        if _is_vec(t):
            zero = {"float": 0.0, "int": 0, "bool": False}[_VEC_KIND[t[0]]]
            return (tuple(zero for _ in range(t[1])), t)
        if _is_mat(t):
            return (tuple(tuple(0.0 for _ in range(t[1]))
                          for _ in range(t[1])), t)
        if _is_array(t):
            return ([self._default(t[1]) for _ in range(t[2])], t)
        if _is_struct(t):
            return ({fn: self._default(glsl_type(ft, self.structs))
                     for ft, fn in self.structs[t[1]]}, t)
        raise GLSLError(f"no default value for type {t!r}")

    # --------------------------------------------------------------- merge
    def _merge(self, old, new, mask, t):
        """new where mask else old, recursively over aggregates."""
        if old is None:
            return new
        od, nd = old[0], new[0]
        if t in ("float", "int", "bool"):
            r = np.where(mask, np.asarray(nd), np.asarray(od))
            return (_cast_scalar(r, t), t)
        if _is_vec(t):
            comps = []
            for oc, nc in zip(od, nd):
                r = np.where(mask, np.asarray(nc), np.asarray(oc))
                comps.append(_cast_scalar(r, _VEC_KIND[t[0]]))
            return (tuple(comps), t)
        if _is_mat(t):
            cols = []
            for ocol, ncol in zip(od, nd):
                cols.append(tuple(
                    _cast_scalar(np.where(mask, np.asarray(nc),
                                          np.asarray(oc)), "float")
                    for oc, nc in zip(ocol, ncol)))
            return (tuple(cols), t)
        if _is_struct(t):
            out = {}
            for ft, fn in self.structs[t[1]]:
                out[fn] = self._merge(old[0][fn], new[0][fn], mask,
                                      glsl_type(ft, self.structs))
            return (out, t)
        if _is_array(t):
            return ([self._merge(o, n, mask, t[1])
                     for o, n in zip(od, nd)], t)
        return new

    def _masked_store(self, old, new, ctx):
        mask = ctx["mask"]
        if mask is None or old is None:
            return new
        return self._merge(old, new, mask, old[1])

    # -------------------------------------------------------------- scopes
    def _lookup(self, env, name):
        for scope in reversed(env):
            if name in scope:
                return scope[name]
        if name in self.globals:
            return self.globals[name]
        raise GLSLError(f"unknown identifier {name!r}")

    def _find_scope(self, env, name):
        for scope in reversed(env):
            if name in scope:
                return scope
        if name in self.globals:
            return self.globals
        return None

    # ------------------------------------------------------------- run API
    def run_main(self, shape, inputs=None, main="main"):
        """Run main() over the lane grid `shape`.

        inputs: {name: (data, type)} host bindings (uniforms, attributes,
        varyings, samplers, gl_* values).  Returns a dict with the final
        "globals" and the "discard" mask (None if nothing discarded).
        """
        self.S = tuple(shape)
        self._build_globals(inputs or {})
        ctx = {"mask": None, "dead": None, "discard": None,
               "loops": [], "ret": []}
        all_discarded = False
        with np.errstate(all="ignore"):
            try:
                self._call_user(main, [], [], [{}], ctx)
            except _Discard:
                all_discarded = True
        discard = ctx["discard"]
        if all_discarded:
            discard = np.ones(self.S, dtype=bool)
        return {"globals": dict(self.globals), "discard": discard}

    def _build_globals(self, inputs):
        self.globals = {}
        ctx = {"mask": None, "dead": None, "discard": None,
               "loops": [], "ret": []}
        for quals, tname, name, init, arr in self.global_decls:
            t = glsl_type(tname, self.structs)
            if arr is not None:
                t = ("array", t, self._const_int(arr, [], ctx))
            if init is not None:
                v = self._eval(init, [], ctx)
                self.globals[name] = self._convert(v, t)
            else:
                self.globals[name] = self._default(t)
        for name, v in inputs.items():
            self.globals[name] = self._normalize_input(v)
        # Built-in outputs/inputs the host may rely on.
        if "gl_FragColor" not in self.globals:
            self.globals["gl_FragColor"] = self._default(("vec", 4))
        if "gl_Position" not in self.globals:
            self.globals["gl_Position"] = self._default(("vec", 4))
        if "gl_PointSize" not in self.globals:
            self.globals["gl_PointSize"] = (1.0, "float")
        if "gl_FrontFacing" not in self.globals:
            self.globals["gl_FrontFacing"] = (True, "bool")
        if "gl_FragCoord" not in self.globals and len(self.S) == 2:
            H, W = self.S
            yy, xx = np.mgrid[0:H, 0:W].astype(F32)
            self.globals["gl_FragCoord"] = (
                (xx + 0.5, (H - 0.5) - yy), ("vec", 2))

    def _normalize_input(self, v):
        data, t = v
        if isinstance(t, str):
            t = glsl_type(t, self.structs)
        if _is_vec(t) and isinstance(data, np.ndarray):
            if data.shape == self.S + (t[1],):
                data = tuple(data[..., k] for k in range(t[1]))
            elif data.shape == (t[1],):
                data = tuple(data[k] for k in range(t[1]))
        elif _is_mat(t) and isinstance(data, np.ndarray):
            n = t[1]
            if data.shape == self.S + (n, n):
                data = tuple(tuple(data[..., i, j] for i in range(n))
                             for j in range(n))
            elif data.shape == (n, n):
                data = tuple(tuple(data[i, j] for i in range(n))
                             for j in range(n))
        return (data, t)

    def _const_int(self, expr, env, ctx):
        v = self._eval(expr, env, ctx)
        d = v[0]
        if _is_vec(v[1]):
            d = d[0]
        if isinstance(d, np.ndarray):
            if d.size != 1:
                raise GLSLError("array size is not a constant scalar")
            d = d.flat[0]
        return int(d)

    def _convert(self, v, t):
        """The implicit conversions GLSL allows (int -> float family)."""
        if v[1] == t:
            return v
        if t == "float" and v[1] == "int":
            return (_cast_scalar(v[0], "float"), "float")
        if _is_vec(t) and _is_vec(v[1]) and t[1] == v[1][1]:
            kind = _VEC_KIND[t[0]]
            return (tuple(_cast_scalar(c, kind) for c in v[0]), t)
        return v

    # ---------------------------------------------------- mask bookkeeping
    def _restore_mask(self, ctx, base):
        """Mask after a branch/loop: `base` minus lanes that died or were
        invalidated (return/discard/break/continue) in the meantime."""
        excl = ctx["dead"]
        for frame in ctx["loops"]:
            for key in ("broken", "continued"):
                m = frame[key]
                if m is not None:
                    excl = m if excl is None else (excl | m)
        if excl is None:
            return base
        if base is None:
            return ~excl
        return base & ~excl

    @staticmethod
    def _record(holder, key, m):
        """OR the lane mask m into holder[key] (None means none yet)."""
        cur = holder[key]
        holder[key] = m if cur is None else (cur | m)

    # ---------------------------------------------------------- statements
    def _exec_block(self, stmts, env, ctx):
        env.append({})
        try:
            for st in stmts:
                m = ctx["mask"]
                if m is not None and not m.any():
                    break  # every lane is predicated off: nothing observable
                self._exec_stmt(st, env, ctx)
        finally:
            env.pop()

    def _exec_stmt(self, st, env, ctx):
        tag = st[0]
        if tag == "block":
            self._exec_block(st[1], env, ctx)
        elif tag == "decl":
            self._exec_decl(st, env, ctx)
        elif tag == "exprstmt":
            self._eval(st[1], env, ctx)
        elif tag == "if":
            self._exec_if(st, env, ctx)
        elif tag == "for":
            _, init, cond, step, body = st
            self._exec_loop(env, ctx, init, cond, step, body, do_first=False)
        elif tag == "while":
            _, cond, body = st
            self._exec_loop(env, ctx, None, cond, None, body, do_first=False)
        elif tag == "do" or tag == "dowhile":
            # "dowhile" is the G1 parser's spelling of the same node.
            _, body, cond = st
            self._exec_loop(env, ctx, None, cond, None, body, do_first=True)
        elif tag == "return":
            self._exec_return(st, env, ctx)
        elif tag == "break":
            self._exec_jump(ctx, "broken", _Break)
        elif tag == "continue":
            self._exec_jump(ctx, "continued", _Continue)
        elif tag == "discard":
            m = ctx["mask"]
            if m is None:
                raise _Discard()
            self._record(ctx, "discard", m)
            self._record(ctx, "dead", m)
            ctx["mask"] = m & ~m
        elif tag == "empty":
            pass
        else:
            raise GLSLError(f"unknown statement {tag!r}")

    def _exec_decl(self, st, env, ctx):
        _, quals, tname, decls = st
        base_t = glsl_type(tname, self.structs)
        for name, init, arr in decls:
            t = base_t
            if arr == ("unsized",) and init is not None \
                    and init[0] == "arrctor":
                # `float w[] = float[](...)`: the parser marks the
                # brackets with ("unsized",); the length comes from
                # the constructor value itself.
                v = self._eval(init, env, ctx)
                env[-1][name] = v
                continue
            if arr is not None:
                t = ("array", base_t, self._const_int(arr, env, ctx))
            if init is not None:
                v = self._convert(self._eval(init, env, ctx), t)
                if ctx["mask"] is not None:
                    v = self._merge(self._default(t), v, ctx["mask"], t)
                env[-1][name] = v
            else:
                env[-1][name] = self._default(t)

    def _exec_if(self, st, env, ctx):
        _, cond_node, then, els = st
        c = self._eval(cond_node, env, ctx)[0]
        base = ctx["mask"]
        if not isinstance(c, np.ndarray) or c.shape != self.S:
            # Scalar (uniform) condition: only the taken branch runs.
            if self._scalar_bool(c):
                self._exec_stmt(then, env, ctx)
            elif els is not None:
                self._exec_stmt(els, env, ctx)
            ctx["mask"] = self._restore_mask(ctx, base)
            return
        c = c.astype(bool, copy=False)
        ctx["mask"] = c if base is None else (base & c)
        self._exec_stmt(then, env, ctx)
        if els is not None:
            ctx["mask"] = ~c if base is None else (base & ~c)
            # Lanes that died in the then-branch stay dead in the else one.
            if ctx["dead"] is not None:
                ctx["mask"] = ctx["mask"] & ~ctx["dead"]
            self._exec_stmt(els, env, ctx)
        ctx["mask"] = self._restore_mask(ctx, base)

    def _exec_return(self, st, env, ctx):
        v = None
        if st[1] is not None:
            v = self._eval(st[1], env, ctx)
        m = ctx["mask"]
        if m is None:
            raise _Return(v)
        frame = ctx["ret"][-1]
        if v is not None:
            if frame["val"] is None:
                frame["val"] = v
            else:
                frame["val"] = self._merge(frame["val"], v, m, v[1])
            self._record(frame, "mask", m)
        self._record(ctx, "dead", m)
        ctx["mask"] = m & ~m

    def _exec_jump(self, ctx, key, signal):
        if not ctx["loops"]:
            raise GLSLError(f"{key} outside a loop")
        m = ctx["mask"]
        if m is None:
            raise signal()
        frame = ctx["loops"][-1]
        self._record(frame, key, m)
        ctx["mask"] = m & ~m

    @staticmethod
    def _and_not(mask, excl):
        """mask minus excl, where None means 'all lanes'."""
        if excl is None:
            return mask
        if mask is None:
            return ~excl
        return mask & ~excl

    def _exec_loop(self, env, ctx, init, cond_node, step_node, body,
                   do_first):
        base = ctx["mask"]
        frame = {"broken": None, "continued": None}
        ctx["loops"].append(frame)
        env.append({})  # loop scope: holds the init declaration
        try:
            if init is not None:
                if init[0] == "decl":
                    self._exec_decl(init, env, ctx)
                else:
                    self._eval(init[1], env, ctx)
            guard = 0
            first = True
            while True:
                guard += 1
                if guard > _MAX_LOOP_ITER:
                    raise GLSLError("loop iteration limit exceeded")
                # Last iteration's `continue` lanes rejoin here.
                frame["continued"] = None
                if first and do_first:
                    active = self._restore_mask(ctx, base)
                elif cond_node is not None:
                    ctx["mask"] = self._restore_mask(ctx, base)
                    c = self._eval(cond_node, env, ctx)[0]
                    active = self._cond_mask(ctx, base, c)
                else:
                    active = self._restore_mask(ctx, base)
                if active is not None and not active.any():
                    break
                ctx["mask"] = active
                try:
                    self._exec_stmt(body, env, ctx)
                except _Break:
                    break
                except _Continue:
                    pass
                if step_node is not None:
                    # `continue` lanes DO reach the step of a for-loop;
                    # broken and dead lanes do not.
                    sm = self._and_not(active, frame["broken"])
                    sm = self._and_not(sm, ctx["dead"])
                    if sm is None or sm.any():
                        ctx["mask"] = sm
                        self._eval(step_node, env, ctx)
                first = False
        finally:
            env.pop()
            ctx["loops"].pop()
        ctx["mask"] = self._restore_mask_after_loop(ctx, base)

    @staticmethod
    def _scalar_bool(c):
        if isinstance(c, np.ndarray):
            return bool(c.flat[0])
        return bool(c)

    def _cond_mask(self, ctx, base, c):
        """Active mask for a loop iteration from condition value c."""
        live = self._restore_mask(ctx, base)
        if not isinstance(c, np.ndarray) or c.shape != self.S:
            if self._scalar_bool(c):
                return live
            return np.zeros(self.S, dtype=bool)
        active = c.astype(bool, copy=False)
        if live is not None:
            active = live & active
        return active

    def _restore_mask_after_loop(self, ctx, base):
        """After a loop: broken/continued lanes revive, dead lanes don't."""
        if ctx["dead"] is None:
            return base
        if base is None:
            return ~ctx["dead"]
        return base & ~ctx["dead"]

    # ---------------------------------------------------------- expressions
    def _eval(self, e, env, ctx):
        tag = e[0]
        if tag == "num":
            return (e[1], e[2])
        if tag == "ident":
            return self._lookup(env, e[1])
        if tag == "bin":
            _, op, a, b = e
            return self._binop(op, self._eval(a, env, ctx),
                               self._eval(b, env, ctx))
        if tag == "unary":
            return self._eval_unary(e, env, ctx)
        if tag == "postfix":
            _, op, a = e
            old = self._eval(a, env, ctx)
            new = self._binop("+" if op == "++" else "-",
                              old, (1, _kind(old[1])))
            self._assign(a, new, env, ctx)
            return old
        if tag == "ternary":
            _, c, a, b = e
            cv = self._eval(c, env, ctx)[0]
            if not isinstance(cv, np.ndarray) or cv.shape != self.S:
                return self._eval(a if self._scalar_bool(cv) else b,
                                  env, ctx)
            va = self._eval(a, env, ctx)
            vb = self._eval(b, env, ctx)
            t = va[1]
            if _is_vec(t):
                comps = [np.where(cv, np.asarray(x), np.asarray(y))
                         for x, y in zip(va[0], vb[0])]
                return (tuple(_cast_scalar(x, _VEC_KIND[t[0]])
                              for x in comps), t)
            r = np.where(cv, np.asarray(va[0]), np.asarray(vb[0]))
            return (_cast_scalar(r, t), t)
        if tag == "field":
            base = self._eval(e[1], env, ctx)
            return self._read_field(base, e[2])
        if tag == "index":
            base = self._eval(e[1], env, ctx)
            idx = self._eval(e[2], env, ctx)
            return self._read_index(base, idx)
        if tag == "call":
            return self._eval_call(e, env, ctx)
        if tag == "arrctor":
            # Array constructor (G1 parser node): float[3](a, b, c) /
            # vec3[](...) -- build the ("array", elem, n) value from
            # the evaluated element expressions.
            _, tname, size_expr, arg_exprs = e
            elem_t = glsl_type(tname, self.structs)
            vals = [self._convert(self._eval(a, env, ctx), elem_t)
                    for a in arg_exprs]
            if size_expr is not None:
                n = self._const_int(size_expr, env, ctx)
            else:
                n = len(vals)
            if len(vals) < n:
                vals = vals + [self._default(elem_t)] * (n - len(vals))
            return (vals[:n], ("array", elem_t, n))
        if tag == "assign":
            _, op, lhs, rhs = e
            rv = self._eval(rhs, env, ctx)
            if op != "=":
                cur = self._eval(lhs, env, ctx)
                rv = self._binop(op[:-1], cur, rv)
            rv = self._assign(lhs, rv, env, ctx)
            return rv
        if tag == "comma":
            self._eval(e[1], env, ctx)
            return self._eval(e[2], env, ctx)
        raise GLSLError(f"unknown expression {tag!r}")

    def _eval_unary(self, e, env, ctx):
        _, op, a = e
        if op in ("++", "--"):
            old = self._eval(a, env, ctx)
            new = self._binop("+" if op == "++" else "-",
                              old, (1, _kind(old[1])))
            self._assign(a, new, env, ctx)
            return new
        v = self._eval(a, env, ctx)
        d, t = v
        if op == "+":
            return v
        if op == "-":
            if _is_vec(t) or _is_mat(t):
                return (self._map_comps(d, t, lambda x: -np.asarray(x)), t)
            return (-np.asarray(d), t)
        if op == "!":
            if _is_vec(t):
                return (tuple(np.logical_not(np.asarray(x, bool))
                              for x in d), t)
            return (np.logical_not(np.asarray(d, bool)), "bool")
        if op == "~":
            if _is_vec(t):
                return (tuple(np.bitwise_not(np.asarray(x, I32))
                              for x in d), t)
            return (np.bitwise_not(np.asarray(d, I32)), "int")
        raise GLSLError(f"unknown unary op {op!r}")

    def _map_comps(self, d, t, fn):
        """Apply fn to every scalar component of a vec/mat data block."""
        if _is_vec(t):
            return tuple(fn(x) for x in d)
        if _is_mat(t):
            return tuple(tuple(fn(x) for x in col) for col in d)
        return fn(d)

    # ------------------------------------------------------------- binops
    def _binop(self, op, a, b):
        ta, tb = a[1], b[1]
        if op in ("&&", "||"):
            fn = np.logical_and if op == "&&" else np.logical_or
            return self._componentwise(
                a, b, lambda x, y: fn(np.asarray(x, bool),
                                      np.asarray(y, bool)), "bool")
        if op in ("&", "|", "^"):
            fn = {"&": np.bitwise_and, "|": np.bitwise_or,
                  "^": np.bitwise_xor}[op]
            kind = "int" if "int" in (_kind(ta), _kind(tb)) else "bool"
            return self._componentwise(a, b, fn, kind)
        if op in ("<<", ">>"):
            fn = np.left_shift if op == "<<" else np.right_shift
            return self._componentwise(
                a, b, lambda x, y: fn(np.asarray(x, I32),
                                      np.asarray(y, I32)), "int")
        if op in ("==", "!="):
            eq = self._equals(a, b)
            return (eq if op == "==" else np.logical_not(eq), "bool")
        if op in ("<", ">", "<=", ">="):
            fn = {"<": np.less, ">": np.greater,
                  "<=": np.less_equal, ">=": np.greater_equal}[op]
            return self._compare_reduce(a, b, fn)
        # arithmetic
        if _is_mat(ta) or _is_mat(tb):
            return self._mat_binop(op, a, b)
        fn = {"+": np.add, "-": np.subtract, "*": np.multiply}.get(op)
        if op == "/":
            return self._divide(a, b)
        if op == "%":
            return self._remainder(a, b)
        if fn is None:
            raise GLSLError(f"unknown binary op {op!r}")
        kind = _promote(a, b)
        return self._componentwise(a, b, fn, kind)

    def _componentwise(self, a, b, fn, kind):
        """fn over aligned components of two scalar/vector values."""
        ta, tb = a[1], b[1]
        na, nb = _ncomp(ta), _ncomp(tb)
        n = max(na, nb)
        ca = _components(a)
        cb = _components(b)
        if na == 1 and n > 1:
            ca = ca * n
        if nb == 1 and n > 1:
            cb = cb * n
        out = [_cast_scalar(fn(np.asarray(x), np.asarray(y)), kind)
               for x, y in zip(ca, cb)]
        return _rebuild(out, _vec_type(kind, n))

    def _divide(self, a, b):
        kind = _promote(a, b)
        if kind == "int":
            def idiv(x, y):
                x = np.asarray(x, I32)
                y = np.asarray(y, I32)
                q = np.abs(x) // np.abs(y)
                return q * np.sign(x) * np.sign(y)
            return self._componentwise(a, b, idiv, "int")
        return self._componentwise(
            a, b, lambda x, y: np.divide(np.asarray(x, F32),
                                         np.asarray(y, F32)), "float")

    def _remainder(self, a, b):
        kind = _promote(a, b)
        if kind == "int":
            def irem(x, y):
                x = np.asarray(x, I32)
                y = np.asarray(y, I32)
                q = np.abs(x) // np.abs(y)
                q = q * np.sign(x) * np.sign(y)
                return x - q * y
            return self._componentwise(a, b, irem, "int")
        return self._componentwise(
            a, b, lambda x, y: np.mod(np.asarray(x, F32),
                                      np.asarray(y, F32)), "float")

    def _equals(self, a, b):
        """GLSL == on aggregates: a single bool (all components equal)."""
        ta, tb = a[1], b[1]
        if _is_struct(ta):
            return np.logical_and.reduce(
                [self._equals(a[0][fn], b[0][fn])
                 for _, fn in self.structs[ta[1]]]) \
                if self.structs[ta[1]] else True
        if _is_array(ta):
            out = None
            for x, y in zip(a[0], b[0]):
                e = self._equals(x, y)
                out = e if out is None else np.logical_and(out, e)
            return True if out is None else out
        if _is_mat(ta):
            out = None
            for ca, cb in zip(a[0], b[0]):
                for x, y in zip(ca, cb):
                    e = np.equal(np.asarray(x), np.asarray(y))
                    out = e if out is None else np.logical_and(out, e)
            return out
        ca, cb = _components(a), _components(b)
        if len(ca) == 1 and len(cb) > 1:
            ca = ca * len(cb)
        if len(cb) == 1 and len(ca) > 1:
            cb = cb * len(ca)
        out = None
        for x, y in zip(ca, cb):
            e = np.equal(np.asarray(x), np.asarray(y))
            out = e if out is None else np.logical_and(out, e)
        return out

    def _compare_reduce(self, a, b, fn):
        """Relational ops: scalars in GLSL; vector inputs reduce with
        all() (the research prototype's forgiving behaviour)."""
        ca, cb = _components(a), _components(b)
        if len(ca) == 1 and len(cb) > 1:
            ca = ca * len(cb)
        if len(cb) == 1 and len(ca) > 1:
            cb = cb * len(ca)
        out = None
        for x, y in zip(ca, cb):
            r = fn(np.asarray(x), np.asarray(y))
            out = r if out is None else np.logical_and(out, r)
        return (out, "bool")

    # -------------------------------------------------------- matrix math
    def _stack_comps(self, comps):
        """Stack value components along a new last axis, broadcasting
        scalar components to the lane shape when others are fields."""
        arrs = [np.asarray(x, F32) for x in comps]
        if any(a.shape == self.S for a in arrs):
            arrs = [a if a.shape == self.S
                    else np.broadcast_to(a, self.S) for a in arrs]
        return np.stack(arrs, axis=-1)

    def _mat_stack(self, v):
        """SoA mat -> dense array (..., n, n) with [row][col] layout."""
        d, t = v
        cols = [self._stack_comps(col) for col in d]
        return np.stack(cols, axis=-1)

    def _mat_unstack(self, m, n):
        return (tuple(tuple(m[..., i, j] for i in range(n))
                      for j in range(n)), ("mat", n))

    def _mat_binop(self, op, a, b):
        ta, tb = a[1], b[1]
        if op in ("+", "-"):
            fn = np.add if op == "+" else np.subtract
            cols = tuple(
                tuple(_cast_scalar(fn(np.asarray(x), np.asarray(y)),
                                   "float")
                      for x, y in zip(ca, cb))
                for ca, cb in zip(a[0], b[0]))
            return (cols, ta if _is_mat(ta) else tb)
        if op == "*":
            if _is_mat(ta) and _is_mat(tb):
                return self._mat_unstack(
                    np.matmul(self._mat_stack(a), self._mat_stack(b)),
                    ta[1])
            if _is_mat(ta) and _is_vec(tb):
                m = self._mat_stack(a)
                v = self._stack_comps(b[0])
                r = np.matmul(m, v[..., None])[..., 0]
                return (tuple(r[..., k] for k in range(ta[1])), tb)
            if _is_vec(ta) and _is_mat(tb):
                m = self._mat_stack(b)
                v = self._stack_comps(a[0])
                r = np.matmul(v[..., None, :], m)[..., 0, :]
                return (tuple(r[..., k] for k in range(tb[1])), ta)
            # mat * scalar / scalar * mat
            mat, sca = (a, b) if _is_mat(ta) else (b, a)
            s = np.asarray(sca[0], F32)
            return (self._map_comps(mat[0], mat[1],
                                    lambda x: np.asarray(x, F32) * s),
                    mat[1])
        if op == "/" and _is_mat(ta) and not _is_mat(tb):
            s = np.asarray(b[0], F32)
            return (self._map_comps(a[0], ta,
                                    lambda x: np.asarray(x, F32) / s), ta)
        raise GLSLError(f"unsupported matrix op {op!r}")

    # --------------------------------------------------------- field/index
    def _read_field(self, base, name):
        d, t = base
        if _is_struct(t):
            if name not in d:
                raise GLSLError(f"struct {t[1]} has no field {name!r}")
            return d[name]
        if _is_vec(t):
            idx = [SWIZ[c] for c in name]
            kind = _VEC_KIND[t[0]]
            if len(idx) == 1:
                return (d[idx[0]], kind)
            return (tuple(d[i] for i in idx), (_KIND_VEC[kind], len(idx)))
        raise GLSLError(f"field access on {t!r}")

    def _read_index(self, base, idx):
        d, t = base
        iv = idx[0]
        if _is_vec(t):
            if _is_field(iv, self.S):
                stacked = self._stack_comps(d)
                sel = np.take_along_axis(
                    stacked, np.asarray(iv, np.int64)[..., None],
                    axis=-1)[..., 0]
                return (sel, _VEC_KIND[t[0]])
            return (d[int(np.asarray(iv).flat[0])], _VEC_KIND[t[0]])
        if _is_mat(t):
            if _is_field(iv, self.S):
                m = self._mat_stack(base)
                idxs = np.broadcast_to(
                    np.asarray(iv, np.int64)[..., None, None],
                    self.S + (t[1], 1))
                sel = np.take_along_axis(m, idxs, axis=-1)
                return (tuple(sel[..., k, 0] for k in range(t[1])),
                        ("vec", t[1]))
            return (d[int(np.asarray(iv).flat[0])], ("vec", t[1]))
        if _is_array(t):
            if _is_field(iv, self.S):
                return self._gather_elems(d, np.asarray(iv, np.int64), t[1])
            return d[int(np.asarray(iv).flat[0])]
        raise GLSLError(f"indexing a {t!r}")

    def _gather_elems(self, elems, idx, elem_t):
        """Per-lane gather from a list of same-typed values."""
        if elem_t in ("float", "int", "bool"):
            stacked = np.stack(
                [np.broadcast_to(np.asarray(e[0]), self.S)
                 for e in elems], axis=0)
            r = np.take_along_axis(stacked, idx[None, ...], axis=0)[0]
            return (_cast_scalar(r, elem_t), elem_t)
        if _is_vec(elem_t):
            comps = []
            for k in range(elem_t[1]):
                stacked = np.stack(
                    [np.broadcast_to(np.asarray(e[0][k]), self.S)
                     for e in elems], axis=0)
                r = np.take_along_axis(stacked, idx[None, ...], axis=0)[0]
                comps.append(_cast_scalar(r, _VEC_KIND[elem_t[0]]))
            return (tuple(comps), elem_t)
        if _is_struct(elem_t):
            return ({fn: self._gather_elems([e[0][fn] for e in elems],
                                            idx, glsl_type(ft, self.structs))
                     for ft, fn in self.structs[elem_t[1]]}, elem_t)
        if _is_mat(elem_t):
            cols = []
            for j in range(elem_t[1]):
                col = self._gather_elems(
                    [(e[0][j], ("vec", elem_t[1])) for e in elems],
                    idx, ("vec", elem_t[1]))
                cols.append(col[0])
            return (tuple(cols), elem_t)
        if _is_array(elem_t):
            return ([self._gather_elems([e[0][k] for e in elems], idx,
                                         elem_t[1])
                     for k in range(elem_t[2])], elem_t)
        raise GLSLError(f"gather of {elem_t!r}")

    # ---------------------------------------------------------- assignment
    def _assign(self, lhs, v, env, ctx):
        """Store value v into the lvalue node lhs, under the current
        mask.  Returns the (full-lane) value now held by the lvalue."""
        tag = lhs[0]
        if tag == "ident":
            scope = self._find_scope(env, lhs[1])
            if scope is None:
                raise GLSLError(f"assignment to undeclared {lhs[1]!r}")
            old = scope[lhs[1]]
            nv = self._masked_store(old, self._convert(v, old[1]), ctx)
            scope[lhs[1]] = nv
            return nv
        # General path: root ident + a chain of field/index steps.
        path = []
        node = lhs
        while node[0] in ("field", "index"):
            if node[0] == "field":
                path.append(("field", node[2]))
            else:
                path.append(("index", node[2]))
            node = node[1]
        if node[0] != "ident":
            raise GLSLError("unsupported lvalue")
        path.reverse()
        scope = self._find_scope(env, node[1])
        if scope is None:
            raise GLSLError(f"assignment to undeclared {node[1]!r}")
        old = scope[node[1]]
        nv = self._write_at(old, path, v, env, ctx)
        scope[node[1]] = nv
        return nv

    def _write_at(self, cur, path, v, env, ctx):
        """Rebuild `cur` with v written at `path`, masked by ctx."""
        if not path:
            return self._masked_store(cur, self._convert(v, cur[1]), ctx)
        step, rest = path[0], path[1:]
        d, t = cur
        if step[0] == "field":
            name = step[1]
            if _is_struct(t):
                ft = None
                for ftype, fname in self.structs[t[1]]:
                    if fname == name:
                        ft = glsl_type(ftype, self.structs)
                nd = dict(d)
                nd[name] = self._write_at(d[name], rest, v, env, ctx)
                return (nd, t)
            if _is_vec(t):
                if rest:
                    raise GLSLError("swizzle is not a deep lvalue")
                return (self._write_swizzle(cur, name, v, ctx), t)
            raise GLSLError(f"field write on {t!r}")
        # index step
        idx = self._eval(step[1], env, ctx)
        return self._write_index(cur, idx, rest, v, env, ctx)

    def _write_swizzle(self, cur, name, v, ctx):
        d, t = cur
        idxs = [SWIZ[c] for c in name]
        src = _components(v)
        if len(src) == 1 and len(idxs) > 1:
            src = src * len(idxs)
        comps = list(d)
        mask = ctx["mask"]
        kind = _VEC_KIND[t[0]]
        for k, ci in enumerate(idxs):
            piece = (src[k], kind)
            old = (comps[ci], kind)
            comps[ci] = self._masked_store(old, piece, ctx)[0] \
                if mask is not None else src[k]
        return tuple(comps)

    def _write_index(self, cur, idx, rest, v, env, ctx):
        d, t = cur
        iv = idx[0]
        mask = ctx["mask"]
        if _is_vec(t):
            kind = _VEC_KIND[t[0]]
            if _is_field(iv, self.S):
                comps = list(d)
                ia = np.asarray(iv, np.int64)
                for k in range(t[1]):
                    sel = ia == k
                    if mask is not None:
                        sel = sel & mask
                    if sel.any():
                        src = v if not rest else self._write_at(
                            (comps[k], kind), rest, v, env, ctx)
                        comps[k] = np.where(sel, np.asarray(src[0]),
                                            np.asarray(comps[k]))
                        comps[k] = _cast_scalar(comps[k], kind)
                return (tuple(comps), t)
            k = int(np.asarray(iv).flat[0])
            comps = list(d)
            sub = self._write_at((comps[k], kind), rest, v, env, ctx)
            comps[k] = sub[0]
            return (tuple(comps), t)
        if _is_mat(t):
            if _is_field(iv, self.S):
                cols = list(d)
                ia = np.asarray(iv, np.int64)
                for k in range(t[1]):
                    sel = ia == k
                    if mask is not None:
                        sel = sel & mask
                    if sel.any():
                        sub = self._write_at((cols[k], ("vec", t[1])),
                                             rest, v, env, ctx)
                        cols[k] = tuple(
                            _cast_scalar(
                                np.where(sel, np.asarray(nc),
                                         np.asarray(oc)), "float")
                            for oc, nc in zip(cols[k], sub[0]))
                return (tuple(cols), t)
            k = int(np.asarray(iv).flat[0])
            cols = list(d)
            cols[k] = self._write_at((cols[k], ("vec", t[1])),
                                     rest, v, env, ctx)[0]
            return (tuple(cols), t)
        if _is_array(t):
            if _is_field(iv, self.S):
                elems = list(d)
                ia = np.asarray(iv, np.int64)
                for k in range(t[2]):
                    sel = ia == k
                    if mask is not None:
                        sel = sel & mask
                    if not sel.any():
                        continue
                    sub = self._write_at(elems[k], rest, v, env, ctx)
                    sub_ctx = dict(ctx)
                    sub_ctx["mask"] = sel
                    elems[k] = self._masked_store(elems[k], sub, sub_ctx)
                return (elems, t)
            k = int(np.asarray(iv).flat[0])
            elems = list(d)
            elems[k] = self._write_at(elems[k], rest, v, env, ctx)
            return (elems, t)
        raise GLSLError(f"index write on {t!r}")

    # --------------------------------------------------------------- calls
    def _eval_call(self, e, env, ctx):
        _, callee, argnodes = e
        if callee[0] != "ident":
            raise GLSLError("only named calls are supported")
        name = callee[1]
        # Constructors: type names and struct names.
        if name in _CONSTRUCTOR_TYPES or name in self.structs:
            args = [self._eval(a, env, ctx) for a in argnodes]
            return self._construct(name, args)
        args = [self._eval(a, env, ctx) for a in argnodes]
        if name in _BUILTINS:
            return _BUILTINS[name](self, args)
        if name in self.funcs:
            return self._call_user(name, args, argnodes, env, ctx)
        raise GLSLError(f"no such function {name!r}")

    def _call_user(self, name, args, argnodes, env, ctx):
        if self._depth >= _MAX_CALL_DEPTH:
            raise GLSLError("call depth limit (recursion is not allowed)")
        chosen = self._resolve(name, args)
        if chosen is None:
            raise GLSLError(f"no overload of {name!r} matches the "
                            f"argument types {[a[1] for a in args]}")
        ret_t, params, body = chosen
        local = [{}]
        out_params = []
        for (quals, pt_name, pname, parr), av, anode in \
                zip(params, args, argnodes):
            if pname is None:
                continue
            pt = glsl_type(pt_name, self.structs)
            if parr is not None:
                pt = ("array", pt,
                      self._const_int(parr, env, ctx))
            local[0][pname] = self._convert(av, pt)
            if "out" in quals or "inout" in quals:
                out_params.append((pname, anode))
        call_mask = ctx["mask"]
        saved_loops = ctx["loops"]
        frame = {"val": None, "mask": None}
        ctx["loops"] = []
        ctx["ret"].append(frame)
        self._depth += 1
        try:
            try:
                self._exec_stmt(body, local, ctx)
            except _Return as r:
                frame["val"] = r.value
                frame["mask"] = None  # returned on every live lane
        finally:
            self._depth -= 1
            ctx["ret"].pop()
            ctx["loops"] = saved_loops
            ctx["mask"] = self._and_not(call_mask, ctx["dead"])
        # Copy out/inout parameters back under the caller's mask.
        for pname, anode in out_params:
            if pname in local[0]:
                self._assign(anode, local[0][pname], env, ctx)
        rv = frame["val"]
        if rv is None:
            return (None, "void")
        if frame["mask"] is not None:
            # Some lanes fell off the end without returning: their value
            # is undefined in GLSL; fill with the type default.
            rv = self._merge(self._default(rv[1]), rv, frame["mask"],
                             rv[1])
        return rv

    def _resolve(self, name, args):
        best = None
        for ret_t, params, body in self.funcs.get(name, []):
            if len(params) != len(args):
                continue
            score = 0
            ok = True
            for (quals, pt_name, pname, parr), av in zip(params, args):
                pt = glsl_type(pt_name, self.structs)
                if parr is not None:
                    pt = ("array", pt, None)
                    if not _is_array(av[1]) or av[1][1] != pt[1]:
                        ok = False
                        break
                    continue
                if pt == av[1]:
                    continue
                if pt == "float" and av[1] == "int":
                    score += 1
                    continue
                if _is_vec(pt) and _is_vec(av[1]) and pt[1] == av[1][1] \
                        and _VEC_KIND[pt[0]] == "float" \
                        and _VEC_KIND[av[1][0]] == "int":
                    score += 1
                    continue
                ok = False
                break
            if ok and (best is None or score < best[0]):
                best = (score, (ret_t, params, body))
        return best[1] if best else None

    # -------------------------------------------------------- constructors
    def _construct(self, name, args):
        if name in self.structs:
            fields = self.structs[name]
            if len(args) != len(fields):
                raise GLSLError(f"struct {name} constructor arity")
            data = {}
            for (ft, fn), av in zip(fields, args):
                data[fn] = self._convert(av, glsl_type(ft, self.structs))
            return (data, ("struct", name))
        if name in ("float", "int", "bool"):
            a = args[0]
            d = _components(a)[0]
            return (_cast_scalar(d, name), name)
        t = glsl_type(name, self.structs)
        if _is_vec(t):
            kind = _VEC_KIND[t[0]]
            flat = []
            for av in args:
                flat.extend(_components(av))
            if len(flat) == 1:
                flat = flat * t[1]
            if len(flat) < t[1]:
                raise GLSLError(f"{name} constructor: too few components")
            comps = tuple(_cast_scalar(x, kind) for x in flat[:t[1]])
            return (comps, t)
        if _is_mat(t):
            return self._construct_mat(t[1], args)
        raise GLSLError(f"cannot construct {name!r}")

    def _construct_mat(self, n, args):
        zero, one = 0.0, 1.0
        if not args:
            cols = tuple(tuple(one if i == j else zero for i in range(n))
                         for j in range(n))
            return (cols, ("mat", n))
        if len(args) == 1:
            a = args[0]
            if _is_mat(a[1]):
                m = a[1][1]
                cols = []
                for j in range(n):
                    col = []
                    for i in range(n):
                        if j < m and i < m:
                            col.append(a[0][j][i])
                        else:
                            col.append(one if i == j else zero)
                    cols.append(tuple(col))
                return (tuple(cols), ("mat", n))
            if not _is_vec(a[1]):
                s = _cast_scalar(a[0], "float")
                cols = tuple(tuple(s if i == j else zero for i in range(n))
                             for j in range(n))
                return (cols, ("mat", n))
        flat = []
        for av in args:
            if _is_vec(av[1]):
                flat.extend(av[0])
            elif _is_mat(av[1]):
                for col in av[0]:
                    flat.extend(col)
            else:
                flat.append(av[0])
        if len(flat) != n * n:
            raise GLSLError(f"mat{n} constructor: {len(flat)} components")
        cols = tuple(
            tuple(_cast_scalar(flat[j * n + i], "float") for i in range(n))
            for j in range(n))
        return (cols, ("mat", n))

    # ------------------------------------------------------- texture / deriv
    def _texture_data(self, samp):
        tex = samp[0]
        if tex is None:
            raise GLSLError("sampling an unbound sampler")
        if isinstance(tex, dict):
            data, wrap = tex["data"], tex.get("wrap", "repeat")
        else:
            data, wrap = tex, "repeat"
        data = np.asarray(data)
        if data.dtype == np.uint8:
            data = data.astype(F32) / 255.0
        return data, wrap

    def _sample2d(self, samp, uv, proj=False):
        data, wrap = self._texture_data(samp)
        h, w = data.shape[0], data.shape[1]
        comps = _components(uv)
        u = np.asarray(comps[0], F32)
        v = np.asarray(comps[1], F32)
        if proj:
            q = np.asarray(comps[3] if len(comps) > 3 else comps[2], F32)
            u = u / q
            v = v / q
        if wrap == "clamp":
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
        else:
            u = u - np.floor(u)
            v = v - np.floor(v)
        xi = np.clip((u * w).astype(np.int64), 0, w - 1)
        yi = np.clip(((1.0 - v) * h).astype(np.int64), 0, h - 1)
        px = data[yi, xi]
        out = [px[..., c] for c in range(data.shape[2])]
        if len(out) == 3:
            out.append(np.ones_like(np.asarray(out[0], F32)))
        return (tuple(out), ("vec", 4))

    def _texel_fetch(self, samp, coord):
        data, _ = self._texture_data(samp)
        h, w = data.shape[0], data.shape[1]
        comps = _components(coord)
        x = np.asarray(comps[0], np.int64)
        y = np.asarray(comps[1], np.int64)
        row = (h - 1) - y
        valid = (x >= 0) & (x < w) & (row >= 0) & (row < h)
        px = data[np.clip(row, 0, h - 1), np.clip(x, 0, w - 1)]
        out = []
        for c in range(data.shape[2]):
            out.append(np.where(valid, px[..., c], 0.0))
        if len(out) == 3:
            out.append(np.where(valid, 1.0, 0.0))
        return (tuple(out), ("vec", 4))

    def _deriv(self, v, axis):
        """Neighbour difference along a grid axis (-1: x, -2: y)."""
        d, t = v
        if len(self.S) != 2:
            return self._default(t) if t != "void" else v

        def one(x):
            if not _is_field(x, self.S):
                return 0.0
            out = np.zeros_like(x)
            hi = [slice(None)] * 2
            lo = [slice(None)] * 2
            hi[axis] = slice(1, None)
            lo[axis] = slice(None, -1)
            out[tuple(hi)] = x[tuple(hi)] - x[tuple(lo)]
            edge = [slice(None)] * 2
            edge[axis] = 0
            second = [slice(None)] * 2
            second[axis] = 1
            out[tuple(edge)] = out[tuple(second)]
            return out

        if _is_vec(t):
            return (tuple(one(x) for x in d), t)
        return (one(d), t)


# ------------------------------------------------------------ builtin impls
def _align(args):
    n = max(_ncomp(a[1]) for a in args)
    lists = []
    for a in args:
        comps = _components(a)
        if len(comps) == 1 and n > 1:
            comps = comps * n
        lists.append(comps)
    return lists, n


def _cw(fn, kind):
    """Componentwise builtin over aligned scalar/vector arguments."""
    def impl(ev, args):
        lists, n = _align(args)
        out = [_cast_scalar(fn(*[np.asarray(x) for x in xs]), kind)
               for xs in zip(*lists)]
        return _rebuild(out, _vec_type(kind, n))
    return impl


def _cw_generic(fn):
    """Componentwise builtin that keeps the promoted argument kind
    (the int overloads of min/max/clamp/abs/sign)."""
    def impl(ev, args):
        kind = _kind(args[0][1])
        for a in args[1:]:
            kind = _promote((None, kind), a)
        lists, n = _align(args)
        out = [_cast_scalar(fn(*[np.asarray(x) for x in xs]), kind)
               for xs in zip(*lists)]
        return _rebuild(out, _vec_type(kind, n))
    return impl


def _is_builtin(fn):
    def impl(ev, args):
        comps = _components(args[0])
        out = [fn(np.asarray(x, F32)) for x in comps]
        return _rebuild(out, _vec_type("bool", _ncomp(args[0][1])))
    return impl


def _dot(ev, args):
    a, b = args
    ca, cb = _components(a), _components(b)
    acc = None
    for x, y in zip(ca, cb):
        p = np.asarray(x, F32) * np.asarray(y, F32)
        acc = p if acc is None else acc + p
    return (acc, "float")


def _cross(ev, args):
    a, b = args
    x, y, z = [np.asarray(c, F32) for c in a[0]]
    u, v, w = [np.asarray(c, F32) for c in b[0]]
    return ((y * w - z * v, z * u - x * w, x * v - y * u), ("vec", 3))


def _length(ev, args):
    a = args[0]
    if not _is_vec(a[1]):
        return (np.abs(np.asarray(a[0], F32)), "float")
    acc = None
    for x in a[0]:
        p = np.asarray(x, F32) ** 2
        acc = p if acc is None else acc + p
    return (np.sqrt(acc), "float")


def _normalize(ev, args):
    a = args[0]
    ln = _length(ev, args)[0]
    return (tuple(np.asarray(x, F32) / ln for x in a[0]), a[1])


def _reflect(ev, args):
    i, n = args
    d = _dot(ev, [n, i])[0]
    return (tuple(np.asarray(x, F32) - 2.0 * d * np.asarray(y, F32)
                  for x, y in zip(i[0], n[0])), i[1])


def _refract(ev, args):
    i, n, eta = args
    d = _dot(ev, [n, i])[0]
    e = np.asarray(eta[0], F32)
    k = 1.0 - e * e * (1.0 - d * d)
    coef = e * d + np.sqrt(np.maximum(k, 0.0))
    out = []
    for x, y in zip(i[0], n[0]):
        val = e * np.asarray(x, F32) - coef * np.asarray(y, F32)
        out.append(np.where(k >= 0.0, val, 0.0))
    return (tuple(out), i[1])


def _faceforward(ev, args):
    n, i, nref = args
    d = _dot(ev, [nref, i])[0]
    return (tuple(np.where(d < 0.0, np.asarray(x, F32),
                           -np.asarray(x, F32)) for x in n[0]), n[1])


def _distance(ev, args):
    diff = ev._binop("-", args[0], args[1])
    return _length(ev, [diff])


def _transpose(ev, args):
    d, t = args[0]
    n = t[1]
    return (tuple(tuple(d[i][j] for i in range(n)) for j in range(n)), t)


def _matrix_comp_mult(ev, args):
    a, b = args
    cols = tuple(
        tuple(_cast_scalar(np.asarray(x, F32) * np.asarray(y, F32),
                           "float")
              for x, y in zip(ca, cb))
        for ca, cb in zip(a[0], b[0]))
    return (cols, a[1])


def _determinant(ev, args):
    return (np.linalg.det(ev._mat_stack(args[0])), "float")


def _inverse(ev, args):
    m = ev._mat_stack(args[0])
    try:
        inv = np.linalg.inv(m)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(m)
    return ev._mat_unstack(inv, args[0][1][1])


def _outer_product(ev, args):
    c, r = args
    cols = tuple(
        tuple(np.asarray(x, F32) * np.asarray(r[0][j])
              for x in c[0])
        for j in range(len(r[0])))
    return (cols, ("mat", len(r[0])))


def _compare_builtin(fn):
    def impl(ev, args):
        lists, n = _align(args)
        out = [fn(np.asarray(x), np.asarray(y))
               for x, y in zip(*lists)]
        return _rebuild(out, _vec_type("bool", n))
    return impl


def _any_all(fn):
    def impl(ev, args):
        comps = _components(args[0])
        acc = np.asarray(comps[0], bool)
        for x in comps[1:]:
            acc = fn(acc, np.asarray(x, bool))
        return (acc, "bool")
    return impl


def _deriv_builtin(axis):
    def impl(ev, args):
        return ev._deriv(args[0], axis)
    return impl


def _fwidth(ev, args):
    dx = ev._deriv(args[0], -1)
    dy = ev._deriv(args[0], -2)
    if _is_vec(args[0][1]):
        return (tuple(np.abs(x) + np.abs(y)
                      for x, y in zip(dx[0], dy[0])), args[0][1])
    return (np.abs(dx[0]) + np.abs(dy[0]), args[0][1])


def _texture_size(ev, args):
    data, _ = ev._texture_data(args[0])
    return ((data.shape[1], data.shape[0]), ("ivec", 2))


_BUILTINS = {
    # Frequency order follows glsl-research/report-shaders.md (chunk
    # vocabulary counts): dot 62, normalize 52, max 43, mix 36,
    # texture2D 31, length 17, clamp 15, cross 15, texelFetch 13,
    # pow 12, sqrt 12, abs 11, fract 9, log2 8, smoothstep 8,
    # dFdx/dFdy 7, min 6, textureSize 6, reflect 5, sin 5, exp 5,
    # exp2 5, floor 5, step 5, cos 4, fwidth 4, refract 3,
    # inversesqrt 2, ceil 2, then the rest of the ES 1.00 vocabulary.
    "dot": _dot,
    "normalize": _normalize,
    "max": _cw_generic(np.maximum),
    "mix": _cw(lambda x, y, a: x * (1.0 - a) + y * a, "float"),
    "texture2D": lambda ev, a: ev._sample2d(a[0], a[1]),
    "texture": lambda ev, a: ev._sample2d(a[0], a[1]),
    "texture2DLod": lambda ev, a: ev._sample2d(a[0], a[1]),
    "texture2DLodEXT": lambda ev, a: ev._sample2d(a[0], a[1]),
    "texture2DProj": lambda ev, a: ev._sample2d(a[0], a[1], proj=True),
    "length": _length,
    "clamp": _cw_generic(lambda x, lo, hi: np.minimum(np.maximum(x, lo),
                                                      hi)),
    "cross": _cross,
    "texelFetch": lambda ev, a: ev._texel_fetch(a[0], a[1]),
    "pow": _cw(np.power, "float"),
    "sqrt": _cw(np.sqrt, "float"),
    "abs": _cw_generic(np.abs),
    "fract": _cw(lambda x: x - np.floor(x), "float"),
    "log2": _cw(np.log2, "float"),
    "smoothstep": _cw(lambda e0, e1, x: (
        lambda t: t * t * (3.0 - 2.0 * t))(
        np.clip((x - e0) / (e1 - e0), 0.0, 1.0)), "float"),
    "dFdx": _deriv_builtin(-1),
    "dFdy": _deriv_builtin(-2),
    "fwidth": _fwidth,
    "min": _cw_generic(np.minimum),
    "textureSize": _texture_size,
    "reflect": _reflect,
    "sin": _cw(np.sin, "float"),
    "cos": _cw(np.cos, "float"),
    "tan": _cw(np.tan, "float"),
    "asin": _cw(np.arcsin, "float"),
    "acos": _cw(np.arccos, "float"),
    "atan": lambda ev, a: _cw(np.arctan2, "float")(ev, a)
    if len(a) == 2 else _cw(np.arctan, "float")(ev, a),
    "exp": _cw(np.exp, "float"),
    "exp2": _cw(np.exp2, "float"),
    "log": _cw(np.log, "float"),
    "floor": _cw(np.floor, "float"),
    "step": _cw(lambda edge, x: np.where(x < edge, 0.0, 1.0), "float"),
    "mod": _cw(lambda x, y: x - y * np.floor(x / y), "float"),
    "sign": _cw_generic(np.sign),
    "ceil": _cw(np.ceil, "float"),
    "trunc": _cw(np.trunc, "float"),
    "round": _cw(np.round, "float"),
    "inversesqrt": _cw(lambda x: 1.0 / np.sqrt(x), "float"),
    "distance": _distance,
    "faceforward": _faceforward,
    "refract": _refract,
    "radians": _cw(lambda x: x * (np.pi / 180.0), "float"),
    "degrees": _cw(lambda x: x * (180.0 / np.pi), "float"),
    "matrixCompMult": _matrix_comp_mult,
    "transpose": _transpose,
    "determinant": _determinant,
    "inverse": _inverse,
    "outerProduct": _outer_product,
    "lessThan": _compare_builtin(np.less),
    "lessThanEqual": _compare_builtin(np.less_equal),
    "greaterThan": _compare_builtin(np.greater),
    "greaterThanEqual": _compare_builtin(np.greater_equal),
    "equal": _compare_builtin(np.equal),
    "notEqual": _compare_builtin(np.not_equal),
    "any": _any_all(np.logical_or),
    "all": _any_all(np.logical_and),
    "not": lambda ev, a: _rebuild(
        [np.logical_not(np.asarray(x, bool)) for x in _components(a[0])],
        _vec_type("bool", _ncomp(a[0][1]))),
    "isnan": _is_builtin(np.isnan),
    "isinf": _is_builtin(np.isinf),
}


def evaluate(program, shape, inputs=None, structs=None):
    """One-shot convenience: build an evaluator and run main()."""
    return GLSLEvaluator(program, structs).run_main(shape, inputs)

"""GLSL ES 1.00 front end for the three.js port.

Pipeline: source text -> preprocessor -> lexer -> recursive-descent
parser -> AST.  The vector evaluator that consumes this AST is G2's
module (glsl_eval.py); this file contains NO evaluation.  No exec /
eval / compile anywhere (Code & Magic public-code rule); imports are
stdlib whitelist only (re, pathlib).

Public API
----------
- GLSLError            -- raised for preprocessor / parse failures.
- load_chunks(dirpath=None) -> {name: source}
    The ShaderChunk texts: the embedded r187 set (see
    _EMBEDDED_CHUNKS) overlaid with the three.js clone's
    ``*.glsl.js`` extractions when the clone is on disk (default
    location: the clone next to this port).  The embedded base keeps
    #include resolution complete in standalone deployments (C&M).
- load_shader_lib(dirpath=None) -> {name: {"vertex": s, "fragment": s}}
- preprocess(source, defines=None, chunks=None) -> str
    Full C-preprocessor pass: #define (object- and function-like),
    #undef, #if / #ifdef / #ifndef / #elif / #else / #endif (with
    ``defined(X)`` / ``defined X``), #include <name> resolved through
    the chunks dict (recursively), #error (raises), #version /
    #extension / #pragma / #line dropped.  Macro expansion is plain
    token substitution (the ``#`` stringify and ``##`` paste operators
    are NOT supported -- the real ShaderChunk sources do not use
    them).  Predefined macros: GL_ES=1, __VERSION__=100.
    ``defines`` maps names to values; a value of False means "not
    defined" (mirrors WebGLProgram.generateDefines skipping false).
    Note: WebGLProgram's replaceLightNums() string replacement is
    realised here by passing NUM_* entries in ``defines``.
- vertex_prefix(defines=None, parameters=None) -> str
- fragment_prefix(defines=None, parameters=None, chunks=None) -> str
- raw_prefix(defines=None, parameters=None) -> str
    Reproductions of the prefixes WebGLProgram prepends to a
    ShaderMaterial (ES 1.00 form, i.e. BEFORE the WebGL2
    "#version 300 es" conversion block, which this port does not
    use).  ``parameters`` mirrors the WebGLProgram parameter names in
    snake_case; defaults describe a plain ShaderMaterial on a default
    renderer with a position/normal/uv geometry (see _VERTEX_DEFAULTS
    / _FRAGMENT_DEFAULTS).  The fragment prefix embeds the
    colorspace_pars_fragment chunk text (and the tonemapping chunk
    when tone_mapping is set); when the chunk is unavailable the
    literal ``#include <...>`` line is emitted instead and resolved
    later by preprocess().
- parse(source, stage=None) -> Program
    Parse ALREADY-preprocessed source.  Leftover lines starting with
    '#' are ignored by the lexer, so parse() is safe on raw text too,
    but directives then have no effect -- use parse_shader() for the
    real pipeline.
- parse_shader(source, stage="fragment", defines=None, chunks=None,
               raw=False, parameters=None) -> Program
    prefix (unless raw=False handling: raw=True uses raw_prefix) +
    preprocess + parse, the ShaderMaterial entry point.

AST CONTRACT (for G2 -- read this before writing the evaluator)
---------------------------------------------------------------
All nodes are tuples whose first element is a string tag.  Type names
are the raw GLSL spellings: "float", "int", "uint", "bool",
"vec2"/"vec3"/"vec4", "ivec2".., "bvec2".., "mat2"/"mat3"/"mat4",
"sampler2D", "samplerCube", "sampler3D", "sampler2DShadow",
"samplerCubeShadow", "sampler2DArray", "sampler2DArrayShadow",
"void", or a struct name.  Precision qualifiers (highp/mediump/lowp)
are parsed and kept in qualifier lists; G2 may ignore them (the
design report sanctions float32 throughout).

Program object:
    .items   -- top-level items, in source order (tuples below)
    .structs -- {struct name: [(field_type, field_name, arr), ...]}
    .stage   -- "vertex" / "fragment" / None
    .functions() / .function_names() / .globals()
    .uniforms() / .varyings() / .attributes()
        -> [(type, name, arr), ...] filtered by storage qualifier

Top-level items:
    ("structdef", name, fields)
        fields = [(field_type, field_name, arr), ...]
    ("func", rettype, name, params, body)
        body is a ("block", ...) statement node.  Overloaded
        functions appear as separate "func" items with the same name.
    ("funcdecl", rettype, name, params)     -- prototype, no body
    ("global", quals, type, name, init, arr)
        ONE item per declarator (``uniform float a, b;`` yields two).
        init is an expression node or None.
    ("precision", precision_qual, type)     -- precision statement

params = [(quals, type, name, arr), ...]; name is None for unnamed
parameters.  quals is a list of qualifier strings ("uniform",
"varying", "attribute", "in", "out", "inout", "const", "highp",
"mediump", "lowp", "invariant"); a layout(...) qualifier appears as
the tuple ("layout", inner_text).

``arr`` (array declarator size) is one of:
    None            -- not an array
    ("unsized",)    -- empty brackets ``[]``
    <expr node>     -- the size expression (usually ("num", n, "int")
                       after preprocessing)

Statements:
    ("block", [stmt, ...])
    ("decl", quals, type, [(name, init, arr), ...])
    ("exprstmt", expr)
    ("if", cond, then_stmt, else_stmt_or_None)
    ("for", init_or_None, cond_or_None, step_expr_or_None, body)
        init is a ("decl", ...) or ("exprstmt", ...) node
    ("while", cond, body)
    ("dowhile", body, cond)
    ("return", expr_or_None)
    ("break",) / ("continue",) / ("discard",)

Expressions:
    ("num", value, kind)     kind in "float"/"int"/"bool"; value is a
                             Python float / int / bool
    ("ident", name)
    ("bin", op, lhs, rhs)    op: + - * / % << >> < > <= >= == !=
                             & ^ | && ||
    ("unary", op, expr)      op: + - ! ~ ++ --   (prefix)
    ("postfix", op, expr)    op: ++ --
    ("assign", op, lhs, rhs) op: = += -= *= /= %= <<= >>= &= ^= |=
    ("ternary", cond, then_expr, else_expr)
    ("call", callee, [arg, ...])
        callee is ("ident", name): a user function, a builtin, or a
        constructor (type name / struct name).  G2 distinguishes via
        its static typing, as in the research prototype.
    ("arrctor", type_name, size_expr_or_None, [arg, ...])
        array constructor ``float[3](...)`` / ``vec3[](...)``
    ("index", base, index_expr)
    ("field", base, name)    struct member access OR swizzle; G2
                             distinguishes via static typing

The shapes above extend the research prototype (glsl_eval.py) only
where GLSL ES 1.00 / the real chunks require it: mat types, bitwise
and shift operators, while/do/discard, function prototypes, array
constructors, parameter/struct-field array sizes, and the
("unsized",) marker.  Everything the prototype had keeps its shape.
"""

import re
from pathlib import Path

__all__ = [
    "GLSLError", "Program", "UNSIZED",
    "load_chunks", "load_shader_lib",
    "preprocess", "vertex_prefix", "fragment_prefix", "raw_prefix",
    "lex", "parse", "parse_shader",
]


class GLSLError(Exception):
    """Preprocessor or parser failure in GLSL source."""


UNSIZED = ("unsized",)

# ---------------------------------------------------------------------------
# ShaderChunk / ShaderLib extraction from the three.js clone
# ---------------------------------------------------------------------------

def _default_chunk_dir():
    return (Path(__file__).resolve().parent.parent / "three.js" / "src"
            / "renderers" / "shaders" / "ShaderChunk")


def _default_lib_dir():
    return (Path(__file__).resolve().parent.parent / "three.js" / "src"
            / "renderers" / "shaders" / "ShaderLib")


def _extract_template(text, start=0):
    """Return (template_text, end_index) for the first backtick string
    at or after ``start``.  Chunk sources contain no backticks or
    ${...} interpolations (verified on the r187 clone)."""
    i = text.index("`", start)
    j = text.index("`", i + 1)
    return text[i + 1:j], j + 1



# ---------------------------------------------------------------------------
# Embedded ShaderChunk texts (standalone completeness)
#
# The 109 ShaderChunk GLSL sources of three.js r187 (package version
# 0.187.0), extracted from the clone's
# src/renderers/shaders/ShaderChunk/*.glsl.js files and embedded
# verbatim, so that #include <...> resolves in standalone deployments
# (e.g. Code & Magic) where the clone is not on disk.  load_chunks()
# uses these as the base and overlays the on-disk extraction when the
# clone is present (the on-disk texts win, so local development keeps
# tracking the clone).  This block is generated data -- do not edit by
# hand; regenerate it from the clone instead.
#
# License: the ShaderChunk sources below are part of three.js and are
# distributed under the MIT License:
#
#   The MIT License
#   Copyright (c) 2010-2026 three.js authors
#
#   Permission is hereby granted, free of charge, to any person
#   obtaining a copy of this software and associated documentation
#   files (the "Software"), to deal in the Software without
#   restriction, including without limitation the rights to use,
#   copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following
#   conditions:
#
#   The above copyright notice and this permission notice shall be
#   included in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#   OTHER DEALINGS IN THE SOFTWARE.
# ---------------------------------------------------------------------------

_EMBEDDED_CHUNKS = {
    'alphahash_fragment': '\n#ifdef USE_ALPHAHASH\n\n\tif ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;\n\n#endif\n',
    'alphahash_pars_fragment': '\n#ifdef USE_ALPHAHASH\n\n\t/**\n\t * See: https://casual-effects.com/research/Wyman2017Hashed/index.html\n\t */\n\n\tconst float ALPHA_HASH_SCALE = 0.05; // Derived from trials only, and may be changed.\n\n\tfloat hash2D( vec2 value ) {\n\n\t\treturn fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );\n\n\t}\n\n\tfloat hash3D( vec3 value ) {\n\n\t\treturn hash2D( vec2( hash2D( value.xy ), value.z ) );\n\n\t}\n\n\tfloat getAlphaHashThreshold( vec3 position ) {\n\n\t\t// Find the discretized derivatives of our coordinates\n\t\tfloat maxDeriv = max(\n\t\t\tlength( dFdx( position.xyz ) ),\n\t\t\tlength( dFdy( position.xyz ) )\n\t\t);\n\t\tfloat pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );\n\n\t\t// Find two nearest log-discretized noise scales\n\t\tvec2 pixScales = vec2(\n\t\t\texp2( floor( log2( pixScale ) ) ),\n\t\t\texp2( ceil( log2( pixScale ) ) )\n\t\t);\n\n\t\t// Compute alpha thresholds at our two noise scales\n\t\tvec2 alpha = vec2(\n\t\t\thash3D( floor( pixScales.x * position.xyz ) ),\n\t\t\thash3D( floor( pixScales.y * position.xyz ) )\n\t\t);\n\n\t\t// Factor to interpolate lerp with\n\t\tfloat lerpFactor = fract( log2( pixScale ) );\n\n\t\t// Interpolate alpha threshold from noise at two scales\n\t\tfloat x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;\n\n\t\t// Pass into CDF to compute uniformly distrib threshold\n\t\tfloat a = min( lerpFactor, 1.0 - lerpFactor );\n\t\tvec3 cases = vec3(\n\t\t\tx * x / ( 2.0 * a * ( 1.0 - a ) ),\n\t\t\t( x - 0.5 * a ) / ( 1.0 - a ),\n\t\t\t1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )\n\t\t);\n\n\t\t// Find our final, uniformly distributed alpha threshold (ατ)\n\t\tfloat threshold = ( x < ( 1.0 - a ) )\n\t\t\t? ( ( x < a ) ? cases.x : cases.y )\n\t\t\t: cases.z;\n\n\t\t// Avoids ατ == 0. Could also do ατ =1-ατ\n\t\treturn clamp( threshold , 1.0e-6, 1.0 );\n\n\t}\n\n#endif\n',
    'alphamap_fragment': '\n#ifdef USE_ALPHAMAP\n\n\tdiffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;\n\n#endif\n',
    'alphamap_pars_fragment': '\n#ifdef USE_ALPHAMAP\n\n\tuniform sampler2D alphaMap;\n\n#endif\n',
    'alphatest_fragment': '\n#ifdef USE_ALPHATEST\n\n\t#ifdef ALPHA_TO_COVERAGE\n\n\tdiffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );\n\tif ( diffuseColor.a == 0.0 ) discard;\n\n\t#else\n\n\tif ( diffuseColor.a < alphaTest ) discard;\n\n\t#endif\n\n#endif\n',
    'alphatest_pars_fragment': '\n#ifdef USE_ALPHATEST\n\tuniform float alphaTest;\n#endif\n',
    'aomap_fragment': '\n#ifdef USE_AOMAP\n\n\t// reads channel R, compatible with a combined OcclusionRoughnessMetallic (RGB) texture\n\tfloat ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;\n\n\t#if defined( STANDARD )\n\t\treflectedLight.indirectDiffuse *= computeMultiBounceOcclusion( ambientOcclusion, material.diffuseContribution );\n\t#else\n\t\treflectedLight.indirectDiffuse *= ambientOcclusion;\n\t#endif\n\n\t#if defined( USE_CLEARCOAT ) \n\t\tclearcoatSpecularIndirect *= computeSpecularOcclusion( saturate( dot( geometryClearcoatNormal, geometryViewDir ) ), ambientOcclusion, material.clearcoatRoughness );\n\t#endif\n\n\t#if defined( USE_SHEEN ) \n\t\tsheenSpecularIndirect *= ambientOcclusion;\n\t#endif\n\n\t#if defined( STANDARD )\n\n\t\tfloat dotNV = saturate( dot( geometryNormal, geometryViewDir ) );\n\n\t\tfloat specularOcclusion = computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );\n\n\t\treflectedLight.indirectSpecular *= computeMultiBounceOcclusion( specularOcclusion, material.specularColorBlended );\n\n\t#endif\n\n#endif\n',
    'aomap_pars_fragment': '\n#ifdef USE_AOMAP\n\n\tuniform sampler2D aoMap;\n\tuniform float aoMapIntensity;\n\n#endif\n',
    'batching_pars_vertex': '\n#ifdef USE_BATCHING\n\t#if ! defined( GL_ANGLE_multi_draw )\n\t#define gl_DrawID _gl_DrawID\n\tuniform int _gl_DrawID;\n\t#endif\n\n\tuniform highp sampler2D batchingTexture;\n\tuniform highp usampler2D batchingIdTexture;\n\tmat4 getBatchingMatrix( const in float i ) {\n\n\t\tint size = textureSize( batchingTexture, 0 ).x;\n\t\tint j = int( i ) * 4;\n\t\tint x = j % size;\n\t\tint y = j / size;\n\t\tvec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );\n\t\tvec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );\n\t\tvec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );\n\t\tvec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );\n\t\treturn mat4( v1, v2, v3, v4 );\n\n\t}\n\n\tfloat getIndirectIndex( const in int i ) {\n\n\t\tint size = textureSize( batchingIdTexture, 0 ).x;\n\t\tint x = i % size;\n\t\tint y = i / size;\n\t\treturn float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );\n\n\t}\n\n#endif\n\n#ifdef USE_BATCHING_COLOR\n\n\tuniform sampler2D batchingColorTexture;\n\tvec4 getBatchingColor( const in float i ) {\n\n\t\tint size = textureSize( batchingColorTexture, 0 ).x;\n\t\tint j = int( i );\n\t\tint x = j % size;\n\t\tint y = j / size;\n\t\treturn texelFetch( batchingColorTexture, ivec2( x, y ), 0 );\n\n\t}\n\n#endif\n',
    'batching_vertex': '\n#ifdef USE_BATCHING\n\tmat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );\n#endif\n',
    'begin_vertex': '\nvec3 transformed = vec3( position );\n\n#ifdef USE_ALPHAHASH\n\n\tvPosition = vec3( position );\n\n#endif\n',
    'beginnormal_vertex': '\nvec3 objectNormal = vec3( normal );\n\n#ifdef USE_TANGENT\n\n\tvec3 objectTangent = vec3( tangent.xyz );\n\n#endif\n',
    'bsdfs': '\n\nfloat G_BlinnPhong_Implicit( /* const in float dotNL, const in float dotNV */ ) {\n\n\t// geometry term is (n dot l)(n dot v) / 4(n dot l)(n dot v)\n\treturn 0.25;\n\n}\n\nfloat D_BlinnPhong( const in float shininess, const in float dotNH ) {\n\n\treturn RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );\n\n}\n\nvec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {\n\n\tvec3 halfDir = normalize( lightDir + viewDir );\n\n\tfloat dotNH = saturate( dot( normal, halfDir ) );\n\tfloat dotVH = saturate( dot( viewDir, halfDir ) );\n\n\tvec3 F = F_Schlick( specularColor, 1.0, dotVH );\n\n\tfloat G = G_BlinnPhong_Implicit( /* dotNL, dotNV */ );\n\n\tfloat D = D_BlinnPhong( shininess, dotNH );\n\n\treturn F * ( G * D );\n\n} // validated\n\n',
    'bumpmap_pars_fragment': "\n#ifdef USE_BUMPMAP\n\n\tuniform sampler2D bumpMap;\n\tuniform float bumpScale;\n\n\t// Bump Mapping Unparametrized Surfaces on the GPU by Morten S. Mikkelsen\n\t// https://mmikk.github.io/papers3d/mm_sfgrad_bump.pdf\n\n\t// Evaluate the derivative of the height w.r.t. screen-space using forward differencing (listing 2)\n\n\tvec2 dHdxy_fwd() {\n\n\t\tvec2 dSTdx = dFdx( vBumpMapUv );\n\t\tvec2 dSTdy = dFdy( vBumpMapUv );\n\n\t\tfloat Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;\n\t\tfloat dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;\n\t\tfloat dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;\n\n\t\treturn vec2( dBx, dBy );\n\n\t}\n\n\tvec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {\n\n\t\t// normalize is done to ensure that the bump map looks the same regardless of the texture's scale\n\t\tvec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );\n\t\tvec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );\n\t\tvec3 vN = surf_norm; // normalized\n\n\t\tvec3 R1 = cross( vSigmaY, vN );\n\t\tvec3 R2 = cross( vN, vSigmaX );\n\n\t\tfloat fDet = dot( vSigmaX, R1 ) * faceDirection;\n\n\t\tvec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );\n\t\treturn normalize( abs( fDet ) * surf_norm - vGrad );\n\n\t}\n\n#endif\n",
    'clearcoat_normal_fragment_begin': '\n#ifdef USE_CLEARCOAT\n\n\tvec3 clearcoatNormal = nonPerturbedNormal;\n\n#endif\n',
    'clearcoat_normal_fragment_maps': '\n#ifdef USE_CLEARCOAT_NORMALMAP\n\n\tvec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;\n\tclearcoatMapN.xy *= clearcoatNormalScale;\n\n\tclearcoatNormal = normalize( tbn2 * clearcoatMapN );\n\n#endif\n',
    'clearcoat_pars_fragment': '\n\n#ifdef USE_CLEARCOATMAP\n\n\tuniform sampler2D clearcoatMap;\n\n#endif\n\n#ifdef USE_CLEARCOAT_NORMALMAP\n\n\tuniform sampler2D clearcoatNormalMap;\n\tuniform vec2 clearcoatNormalScale;\n\n#endif\n\n#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n\n\tuniform sampler2D clearcoatRoughnessMap;\n\n#endif\n',
    'clipping_planes_fragment': '\n#if NUM_CLIPPING_PLANES > 0\n\n\tvec4 plane;\n\n\t#ifdef ALPHA_TO_COVERAGE\n\n\t\tfloat distanceToPlane, distanceGradient;\n\t\tfloat clipOpacity = 1.0;\n\n\t\t#pragma unroll_loop_start\n\t\tfor ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {\n\n\t\t\tplane = clippingPlanes[ i ];\n\t\t\tdistanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;\n\t\t\tdistanceGradient = fwidth( distanceToPlane ) / 2.0;\n\t\t\tclipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );\n\n\t\t\tif ( clipOpacity == 0.0 ) discard;\n\n\t\t}\n\t\t#pragma unroll_loop_end\n\n\t\t#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES\n\n\t\t\tfloat unionClipOpacity = 1.0;\n\n\t\t\t#pragma unroll_loop_start\n\t\t\tfor ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {\n\n\t\t\t\tplane = clippingPlanes[ i ];\n\t\t\t\tdistanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;\n\t\t\t\tdistanceGradient = fwidth( distanceToPlane ) / 2.0;\n\t\t\t\tunionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );\n\n\t\t\t}\n\t\t\t#pragma unroll_loop_end\n\n\t\t\tclipOpacity *= 1.0 - unionClipOpacity;\n\n\t\t#endif\n\n\t\tdiffuseColor.a *= clipOpacity;\n\n\t\tif ( diffuseColor.a == 0.0 ) discard;\n\n\t#else\n\n\t\t#pragma unroll_loop_start\n\t\tfor ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {\n\n\t\t\tplane = clippingPlanes[ i ];\n\t\t\tif ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;\n\n\t\t}\n\t\t#pragma unroll_loop_end\n\n\t\t#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES\n\n\t\t\tbool clipped = true;\n\n\t\t\t#pragma unroll_loop_start\n\t\t\tfor ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {\n\n\t\t\t\tplane = clippingPlanes[ i ];\n\t\t\t\tclipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;\n\n\t\t\t}\n\t\t\t#pragma unroll_loop_end\n\n\t\t\tif ( clipped ) discard;\n\n\t\t#endif\n\n\t#endif\n\n#endif\n',
    'clipping_planes_pars_fragment': '\n#if NUM_CLIPPING_PLANES > 0\n\n\tvarying vec3 vClipPosition;\n\n\tuniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];\n\n#endif\n',
    'clipping_planes_pars_vertex': '\n#if NUM_CLIPPING_PLANES > 0\n\n\tvarying vec3 vClipPosition;\n\n#endif\n',
    'clipping_planes_vertex': '\n#if NUM_CLIPPING_PLANES > 0\n\n\tvClipPosition = - mvPosition.xyz;\n\n#endif\n',
    'color_fragment': '\n#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )\n\n\tdiffuseColor *= vColor;\n\n#endif\n',
    'color_pars_fragment': '\n#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )\n\n\tvarying vec4 vColor;\n\n#endif\n',
    'color_pars_vertex': '\n#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )\n\n\tvarying vec4 vColor;\n\n#endif\n',
    'color_vertex': '\n#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )\n\n\tvColor = vec4( 1.0 );\n\n#endif\n\n#ifdef USE_COLOR_ALPHA\n\n\tvColor *= color;\n\n#elif defined( USE_COLOR )\n\n\tvColor.rgb *= color;\n\n#endif\n\n#ifdef USE_INSTANCING_COLOR\n\n\tvColor.rgb *= instanceColor.rgb;\n\n#endif\n\n#ifdef USE_BATCHING_COLOR\n\n\tvColor *= getBatchingColor( getIndirectIndex( gl_DrawID ) );\n\n#endif\n',
    'colorspace_fragment': '\ngl_FragColor = linearToOutputTexel( gl_FragColor );\n',
    'colorspace_pars_fragment': '\n\nvec4 LinearTransferOETF( in vec4 value ) {\n\treturn value;\n}\n\nvec4 sRGBTransferEOTF( in vec4 value ) {\n\treturn vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );\n}\n\nvec4 sRGBTransferOETF( in vec4 value ) {\n\treturn vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );\n}\n\n',
    'common': "\n#define PI 3.141592653589793\n#define PI2 6.283185307179586\n#define PI_HALF 1.5707963267948966\n#define RECIPROCAL_PI 0.3183098861837907\n#define RECIPROCAL_PI2 0.15915494309189535\n#define EPSILON 1e-6\n\n#ifndef saturate\n// <tonemapping_pars_fragment> may have defined saturate() already\n#define saturate( a ) clamp( a, 0.0, 1.0 )\n#endif\n#define whiteComplement( a ) ( 1.0 - saturate( a ) )\n\nfloat pow2( const in float x ) { return x*x; }\nvec3 pow2( const in vec3 x ) { return x*x; }\nfloat pow3( const in float x ) { return x*x*x; }\nfloat pow4( const in float x ) { float x2 = x*x; return x2*x2; }\nfloat max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }\nfloat average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }\n\n// expects values in the range of [0,1]x[0,1], returns values in the [0,1] range.\n// do not collapse into a single function per: http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/\nhighp float rand( const in vec2 uv ) {\n\n\tconst highp float a = 12.9898, b = 78.233, c = 43758.5453;\n\thighp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );\n\n\treturn fract( sin( sn ) * c );\n\n}\n\n#ifdef HIGH_PRECISION\n\tfloat precisionSafeLength( vec3 v ) { return length( v ); }\n#else\n\tfloat precisionSafeLength( vec3 v ) {\n\t\tfloat maxComponent = max3( abs( v ) );\n\t\treturn length( v / maxComponent ) * maxComponent;\n\t}\n#endif\n\nstruct IncidentLight {\n\tvec3 color;\n\tvec3 direction;\n\tbool visible;\n};\n\nstruct ReflectedLight {\n\tvec3 directDiffuse;\n\tvec3 directSpecular;\n\tvec3 indirectDiffuse;\n\tvec3 indirectSpecular;\n};\n\n#ifdef USE_ALPHAHASH\n\n\tvarying vec3 vPosition;\n\n#endif\n\nvec3 transformDirection( in vec3 dir, in mat4 matrix ) {\n\n\treturn normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );\n\n}\n\n#define inverseTransformDirection transformDirectionByInverseViewMatrix // @deprecated r185\n\nvec3 transformNormalByInverseViewMatrix( in vec3 normal, in mat4 viewMatrix ) {\n\n\t// upper-left 3x3 of view matrix is assumed to be orthogonal\n\n\treturn normalize( ( vec4( normal, 0.0 ) * viewMatrix ).xyz );\n\n}\n\nvec3 transformDirectionByInverseViewMatrix( in vec3 dir, in mat4 viewMatrix ) {\n\n\t// upper-left 3x3 of view matrix is assumed to be orthogonal\n\n\treturn normalize( ( vec4( dir, 0.0 ) * viewMatrix ).xyz );\n\n}\n\nbool isPerspectiveMatrix( mat4 m ) {\n\n\treturn m[ 2 ][ 3 ] == - 1.0;\n\n}\n\nvec2 equirectUv( in vec3 dir ) {\n\n\t// dir is assumed to be unit length\n\n\tfloat u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;\n\n\tfloat v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;\n\n\treturn vec2( u, v );\n\n}\n\nvec3 BRDF_Lambert( const in vec3 diffuseColor ) {\n\n\treturn RECIPROCAL_PI * diffuseColor;\n\n} // validated\n\nvec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {\n\n\t// Original approximation by Christophe Schlick '94\n\t// float fresnel = pow( 1.0 - dotVH, 5.0 );\n\n\t// Optimized variant (presented by Epic at SIGGRAPH '13)\n\t// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf\n\tfloat fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );\n\n\treturn f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );\n\n} // validated\n\nfloat F_Schlick( const in float f0, const in float f90, const in float dotVH ) {\n\n\t// Original approximation by Christophe Schlick '94\n\t// float fresnel = pow( 1.0 - dotVH, 5.0 );\n\n\t// Optimized variant (presented by Epic at SIGGRAPH '13)\n\t// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf\n\tfloat fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );\n\n\treturn f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );\n\n} // validated\n",
    'default_fragment': '\nvoid main() {\n\tgl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );\n}\n',
    'default_vertex': '\nvoid main() {\n\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n}\n',
    'defaultnormal_vertex': '\n\nvec3 transformedNormal = objectNormal;\n#ifdef USE_TANGENT\n\n\tvec3 transformedTangent = objectTangent;\n\n#endif\n\n#ifdef USE_BATCHING\n\n\t// this is in lieu of a per-instance normal-matrix\n\t// non-uniform scaling in the instance matrix is supported\n\t// shear transforms are not supported\n\n\tmat3 bm = mat3( batchingMatrix );\n\ttransformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );\n\ttransformedNormal = bm * transformedNormal;\n\n\t#ifdef USE_TANGENT\n\n\t\ttransformedTangent = bm * transformedTangent;\n\n\t#endif\n\n#endif\n\n#ifdef USE_INSTANCING\n\n\t// this is in lieu of a per-instance normal-matrix\n\t// non-uniform scaling in the instance matrix is supported\n\t// shear transforms are not supported\n\n\tmat3 im = mat3( instanceMatrix );\n\ttransformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );\n\ttransformedNormal = im * transformedNormal;\n\n\t#ifdef USE_TANGENT\n\n\t\ttransformedTangent = im * transformedTangent;\n\n\t#endif\n\n#endif\n\ntransformedNormal = normalMatrix * transformedNormal;\n\n#ifdef FLIP_SIDED\n\n\ttransformedNormal = - transformedNormal;\n\n#endif\n\n#ifdef USE_TANGENT\n\n\ttransformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;\n\n#endif\n',
    'displacementmap_pars_vertex': '\n#ifdef USE_DISPLACEMENTMAP\n\n\tuniform sampler2D displacementMap;\n\tuniform float displacementScale;\n\tuniform float displacementBias;\n\n#endif\n',
    'displacementmap_vertex': '\n#ifdef USE_DISPLACEMENTMAP\n\n\ttransformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );\n\n#endif\n',
    'dithering_fragment': '\n#ifdef DITHERING\n\n\tgl_FragColor.rgb = dithering( gl_FragColor.rgb );\n\n#endif\n',
    'dithering_pars_fragment': '\n#ifdef DITHERING\n\n\t// based on https://www.shadertoy.com/view/MslGR8\n\tvec3 dithering( vec3 color ) {\n\t\t//Calculate grid position\n\t\tfloat grid_position = rand( gl_FragCoord.xy );\n\n\t\t//Shift the individual colors differently, thus making it even harder to see the dithering pattern\n\t\tvec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );\n\n\t\t//modify shift according to grid position.\n\t\tdither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );\n\n\t\t//shift the color by dither_shift\n\t\treturn color + dither_shift_RGB;\n\t}\n\n#endif\n',
    'emissivemap_fragment': '\n#ifdef USE_EMISSIVEMAP\n\n\tvec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );\n\n\t#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE\n\n\t\t// use inline sRGB decode until browsers properly support SRGB8_ALPHA8 with video textures (#26516)\n\n\t\temissiveColor = sRGBTransferEOTF( emissiveColor );\n\n\t#endif\n\n\ttotalEmissiveRadiance *= emissiveColor.rgb;\n\n#endif\n',
    'emissivemap_pars_fragment': '\n#ifdef USE_EMISSIVEMAP\n\n\tuniform sampler2D emissiveMap;\n\n#endif\n',
    'envmap_common_pars_fragment': "\n#ifdef USE_ENVMAP\n\n\tuniform float envMapIntensity;\n\tuniform mat3 envMapRotation;\n\tuniform samplerCube envMap;\n\n\t#ifdef ENVMAP_TYPE_PMREM\n\n\t\t// Invert PMREMGenerator.lodToRoughness(). Matches Filament's perceptualRoughnessToLod().\n\t\t// https://github.com/google/filament/blob/main/shaders/src/surface_light_indirect.fs\n\t\tfloat roughnessToMip( const in float roughness ) {\n\n\t\t\tfloat r = clamp( roughness, 0.0, 1.0 );\n\n\t\t\treturn ENVMAP_MAX_LOD * r * ( 2.0 - r );\n\n\t\t}\n\n\t#endif\n\n#endif\n",
    'envmap_fragment': '\n#ifdef USE_ENVMAP\n\n\t#ifdef ENV_WORLDPOS\n\n\t\tvec3 cameraToFrag;\n\n\t\tif ( isOrthographic ) {\n\n\t\t\tcameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );\n\n\t\t} else {\n\n\t\t\tcameraToFrag = normalize( vWorldPosition - cameraPosition );\n\n\t\t}\n\n\t\t// Transforming Normal Vectors with the Inverse Transformation\n\t\tvec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );\n\n\t\t#ifdef ENVMAP_MODE_REFLECTION\n\n\t\t\tvec3 reflectVec = reflect( cameraToFrag, worldNormal );\n\n\t\t#else\n\n\t\t\tvec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );\n\n\t\t#endif\n\n\t#else\n\n\t\tvec3 reflectVec = vReflect;\n\n\t#endif\n\n\t#ifdef ENVMAP_TYPE_CUBE\n\n\t\tvec4 envColor = textureCube( envMap, envMapRotation * reflectVec );\n\n\t\t#ifdef ENVMAP_BLENDING_MULTIPLY\n\n\t\t\toutgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );\n\n\t\t#elif defined( ENVMAP_BLENDING_MIX )\n\n\t\t\toutgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );\n\n\t\t#elif defined( ENVMAP_BLENDING_ADD )\n\n\t\t\toutgoingLight += envColor.xyz * specularStrength * reflectivity;\n\n\t\t#endif\n\n\t#endif\n\n#endif\n',
    'envmap_pars_fragment': '\n#ifdef USE_ENVMAP\n\n\tuniform float reflectivity;\n\n\t#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )\n\n\t\t#define ENV_WORLDPOS\n\n\t#endif\n\n\t#ifdef ENV_WORLDPOS\n\n\t\tvarying vec3 vWorldPosition;\n\t\tuniform float refractionRatio;\n\t#else\n\t\tvarying vec3 vReflect;\n\t#endif\n\n#endif\n',
    'envmap_pars_vertex': '\n#ifdef USE_ENVMAP\n\n\t#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )\n\n\t\t#define ENV_WORLDPOS\n\n\t#endif\n\n\t#ifdef ENV_WORLDPOS\n\t\t\n\t\tvarying vec3 vWorldPosition;\n\n\t#else\n\n\t\tvarying vec3 vReflect;\n\t\tuniform float refractionRatio;\n\n\t#endif\n\n#endif\n',
    'envmap_physical_pars_fragment': '\n#ifdef USE_ENVMAP\n\n\tvec3 getIBLIrradiance( const in vec3 normal ) {\n\n\t\t#ifdef ENVMAP_TYPE_PMREM\n\n\t\t\tvec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );\n\n\t\t\tvec4 envMapColor = textureLod( envMap, envMapRotation * worldNormal, ENVMAP_MAX_LOD );\n\n\t\t\treturn PI * envMapColor.rgb * envMapIntensity;\n\n\t\t#else\n\n\t\t\treturn vec3( 0.0 );\n\n\t\t#endif\n\n\t}\n\n\tvec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {\n\n\t\t#ifdef ENVMAP_TYPE_PMREM\n\n\t\t\tvec3 reflectVec = reflect( - viewDir, normal );\n\n\t\t\t// Mixing the reflection with the normal is more accurate and keeps rough objects from gathering light from behind their tangent plane.\n\t\t\treflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );\n\n\t\t\treflectVec = transformDirectionByInverseViewMatrix( reflectVec, viewMatrix );\n\n\t\t\tvec4 envMapColor = textureLod( envMap, envMapRotation * reflectVec, roughnessToMip( roughness ) );\n\n\t\t\treturn envMapColor.rgb * envMapIntensity;\n\n\t\t#else\n\n\t\t\treturn vec3( 0.0 );\n\n\t\t#endif\n\n\t}\n\n\t#ifdef USE_RETROREFLECTION\n\n\t\tvec3 getIBLRetroRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {\n\n\t\t\t#ifdef ENVMAP_TYPE_PMREM\n\n\t\t\t\t// The retroreflective lobe returns light toward its source, so the environment is sampled along the view direction\n\t\t\t\tvec3 retroVec = normalize( mix( viewDir, normal, pow4( roughness ) ) );\n\n\t\t\t\tretroVec = transformDirectionByInverseViewMatrix( retroVec, viewMatrix );\n\n\t\t\t\tvec4 envMapColor = textureLod( envMap, envMapRotation * retroVec, roughnessToMip( roughness ) );\n\n\t\t\t\treturn envMapColor.rgb * envMapIntensity;\n\n\t\t\t#else\n\n\t\t\t\treturn vec3( 0.0 );\n\n\t\t\t#endif\n\n\t\t}\n\n\t#endif\n\n\t#ifdef USE_ANISOTROPY\n\n\t\tvec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {\n\n\t\t\t#ifdef ENVMAP_TYPE_PMREM\n\n\t\t\t  // https://google.github.io/filament/Filament.md.html#lighting/imagebasedlights/anisotropy\n\t\t\t\tvec3 bentNormal = cross( bitangent, viewDir );\n\t\t\t\tbentNormal = normalize( cross( bentNormal, bitangent ) );\n\t\t\t\tbentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );\n\n\t\t\t\treturn getIBLRadiance( viewDir, bentNormal, roughness );\n\n\t\t\t#else\n\n\t\t\t\treturn vec3( 0.0 );\n\n\t\t\t#endif\n\n\t\t}\n\n\t\t#ifdef USE_RETROREFLECTION\n\n\t\t\tvec3 getIBLAnisotropyRetroRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {\n\n\t\t\t\t#ifdef ENVMAP_TYPE_PMREM\n\n\t\t\t\t  // https://google.github.io/filament/Filament.md.html#lighting/imagebasedlights/anisotropy\n\t\t\t\t\tvec3 bentNormal = cross( bitangent, viewDir );\n\t\t\t\t\tbentNormal = normalize( cross( bentNormal, bitangent ) );\n\t\t\t\t\tbentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );\n\n\t\t\t\t\treturn getIBLRetroRadiance( viewDir, bentNormal, roughness );\n\n\t\t\t\t#else\n\n\t\t\t\t\treturn vec3( 0.0 );\n\n\t\t\t\t#endif\n\n\t\t\t}\n\n\t\t#endif\n\n\t#endif\n\n#endif\n',
    'envmap_vertex': '\n#ifdef USE_ENVMAP\n\n\t#ifdef ENV_WORLDPOS\n\n\t\tvWorldPosition = worldPosition.xyz;\n\n\t#else\n\n\t\tvec3 cameraToVertex;\n\n\t\tif ( isOrthographic ) {\n\n\t\t\tcameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );\n\n\t\t} else {\n\n\t\t\tcameraToVertex = normalize( worldPosition.xyz - cameraPosition );\n\n\t\t}\n\n\t\tvec3 worldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );\n\n\t\t#ifdef ENVMAP_MODE_REFLECTION\n\n\t\t\tvReflect = reflect( cameraToVertex, worldNormal );\n\n\t\t#else\n\n\t\t\tvReflect = refract( cameraToVertex, worldNormal, refractionRatio );\n\n\t\t#endif\n\n\t#endif\n\n#endif\n',
    'fog_fragment': '\n#ifdef USE_FOG\n\n\t#ifdef FOG_EXP2\n\n\t\tfloat fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );\n\n\t#else\n\n\t\tfloat fogFactor = smoothstep( fogNear, fogFar, vFogDepth );\n\n\t#endif\n\n\tgl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );\n\n#endif\n',
    'fog_pars_fragment': '\n#ifdef USE_FOG\n\n\tuniform vec3 fogColor;\n\tvarying float vFogDepth;\n\n\t#ifdef FOG_EXP2\n\n\t\tuniform float fogDensity;\n\n\t#else\n\n\t\tuniform float fogNear;\n\t\tuniform float fogFar;\n\n\t#endif\n\n#endif\n',
    'fog_pars_vertex': '\n#ifdef USE_FOG\n\n\tvarying float vFogDepth;\n\n#endif\n',
    'fog_vertex': '\n#ifdef USE_FOG\n\n\tvFogDepth = - mvPosition.z;\n\n#endif\n',
    'gradientmap_pars_fragment': '\n\n#ifdef USE_GRADIENTMAP\n\n\tuniform sampler2D gradientMap;\n\n#endif\n\nvec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {\n\n\t// dotNL will be from -1.0 to 1.0\n\tfloat dotNL = dot( normal, lightDirection );\n\tvec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );\n\n\t#ifdef USE_GRADIENTMAP\n\n\t\treturn vec3( textureLod( gradientMap, coord, 0.0 ).r );\n\n\t#else\n\n\t\tvec2 fw = fwidth( coord ) * 0.5;\n\t\treturn mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );\n\n\t#endif\n\n}\n',
    'iridescence_fragment': "\n\n#ifdef USE_IRIDESCENCE\n\n\t// XYZ to linear-sRGB color space\n\tconst mat3 XYZ_TO_REC709 = mat3(\n\t\t 3.2404542, -0.9692660,  0.0556434,\n\t\t-1.5371385,  1.8760108, -0.2040259,\n\t\t-0.4985314,  0.0415560,  1.0572252\n\t);\n\n\t// Assume air interface for top\n\t// Note: We don't handle the case fresnel0 == 1\n\tvec3 Fresnel0ToIor( vec3 fresnel0 ) {\n\n\t\tvec3 sqrtF0 = sqrt( fresnel0 );\n\t\treturn ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );\n\n\t}\n\n\t// Conversion FO/IOR\n\tvec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {\n\n\t\treturn pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );\n\n\t}\n\n\t// ior is a value between 1.0 and 3.0. 1.0 is air interface\n\tfloat IorToFresnel0( float transmittedIor, float incidentIor ) {\n\n\t\treturn pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));\n\n\t}\n\n\t// Fresnel equations for dielectric/dielectric interfaces.\n\t// Ref: https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html\n\t// Evaluation XYZ sensitivity curves in Fourier space\n\tvec3 evalSensitivity( float OPD, vec3 shift ) {\n\n\t\tfloat phase = 2.0 * PI * OPD * 1.0e-9;\n\t\tvec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );\n\t\tvec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );\n\t\tvec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );\n\n\t\tvec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );\n\t\txyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );\n\t\txyz /= 1.0685e-7;\n\n\t\tvec3 rgb = XYZ_TO_REC709 * xyz;\n\t\treturn rgb;\n\n\t}\n\n\tvec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {\n\n\t\tvec3 I;\n\n\t\t// Force iridescenceIOR -> outsideIOR when thinFilmThickness -> 0.0\n\t\tfloat iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );\n\t\t// Evaluate the cosTheta on the base layer (Snell law)\n\t\tfloat sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );\n\n\t\t// Handle TIR:\n\t\tfloat cosTheta2Sq = 1.0 - sinTheta2Sq;\n\t\tif ( cosTheta2Sq < 0.0 ) {\n\n\t\t\treturn vec3( 1.0 );\n\n\t\t}\n\n\t\tfloat cosTheta2 = sqrt( cosTheta2Sq );\n\n\t\t// First interface\n\t\tfloat R0 = IorToFresnel0( iridescenceIOR, outsideIOR );\n\t\tfloat R12 = F_Schlick( R0, 1.0, cosTheta1 );\n\t\tfloat T121 = 1.0 - R12;\n\t\tfloat phi12 = 0.0;\n\t\tif ( iridescenceIOR < outsideIOR ) phi12 = PI;\n\t\tfloat phi21 = PI - phi12;\n\n\t\t// Second interface\n\t\tvec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) ); // guard against 1.0\n\t\tvec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );\n\t\tvec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );\n\t\tvec3 phi23 = vec3( 0.0 );\n\t\tif ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;\n\t\tif ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;\n\t\tif ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;\n\n\t\t// Phase shift\n\t\tfloat OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;\n\t\tvec3 phi = vec3( phi21 ) + phi23;\n\n\t\t// Compound terms\n\t\tvec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );\n\t\tvec3 r123 = sqrt( R123 );\n\t\tvec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );\n\n\t\t// Reflectance term for m = 0 (DC term amplitude)\n\t\tvec3 C0 = R12 + Rs;\n\t\tI = C0;\n\n\t\t// Reflectance term for m > 0 (pairs of diracs)\n\t\tvec3 Cm = Rs - T121;\n\t\tfor ( int m = 1; m <= 2; ++ m ) {\n\n\t\t\tCm *= r123;\n\t\t\tvec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );\n\t\t\tI += Cm * Sm;\n\n\t\t}\n\n\t\t// Since out of gamut colors might be produced, negative color values are clamped to 0.\n\t\treturn max( I, vec3( 0.0 ) );\n\n\t}\n\n#endif\n\n",
    'iridescence_pars_fragment': '\n\n#ifdef USE_IRIDESCENCEMAP\n\n\tuniform sampler2D iridescenceMap;\n\n#endif\n\n#ifdef USE_IRIDESCENCE_THICKNESSMAP\n\n\tuniform sampler2D iridescenceThicknessMap;\n\n#endif\n',
    'lightmap_pars_fragment': '\n#ifdef USE_LIGHTMAP\n\n\tuniform sampler2D lightMap;\n\tuniform float lightMapIntensity;\n\n#endif\n',
    'lightprobes_pars_fragment': "\n#if NUM_LIGHT_PROBE_GRIDS > 0\n\n// Single atlas 3D texture that stores all 7 SH sub-volumes stacked along Z.\n// Atlas depth = 7 * ( nz + 2 ) where nz = probesResolution.z.\n// Each sub-volume occupies ( nz + 2 ) slices: 1 padding + nz data + 1 padding.\n// Padding is a copy of the first / last data slice and prevents color bleeding\n// when the hardware linear filter reads across a sub-volume boundary.\nuniform highp sampler3D probesSH[ NUM_LIGHT_PROBE_GRIDS ];\n\nuniform vec3 probesMin[ NUM_LIGHT_PROBE_GRIDS ];\nuniform vec3 probesMax[ NUM_LIGHT_PROBE_GRIDS ];\nuniform vec3 probesResolution[ NUM_LIGHT_PROBE_GRIDS ];\n\nbool isInsideLightProbeGrid( vec3 worldPos, vec3 worldNormal, vec3 probesMin, vec3 probesMax, vec3 probesResolution ) {\n\n\tvec3 spacing = ( probesMax - probesMin ) / probesResolution;\n\tvec3 samplePos = worldPos + worldNormal * spacing * 0.5;\n\n\t// Grow the bounds by 1% of a cell, so fragments on the bounds aren't left out by\n\t// precision errors or samples outside of the triangle.\n\n\tvec3 margin = spacing * 0.01;\n\treturn all( greaterThanEqual( samplePos, probesMin - margin ) ) && all( lessThanEqual( samplePos, probesMax + margin ) );\n\n}\n\nvec3 getLightProbeGridIrradiance( highp sampler3D probesSH, vec3 probesMin, vec3 probesMax, vec3 probesResolution, vec3 worldPos, vec3 worldNormal ) {\n\n\tvec3 res = probesResolution;\n\tvec3 gridRange = probesMax - probesMin;\n\tvec3 probeSpacing = gridRange / res;\n\n\t// Offset sample position along normal by half a probe spacing\n\tvec3 samplePos = worldPos + worldNormal * probeSpacing * 0.5;\n\n\t// The bounds hold one cell per probe, so they map directly to the probe grid.\n\t// Clamp to the texel centers of the outermost probes (XY and Z).\n\tvec3 uvw = clamp( ( samplePos - probesMin ) / gridRange, 0.5 / res, 1.0 - 0.5 / res );\n\n\t// Atlas UV mapping along Z:\n\t//   paddedSlices = nz + 2  (1 padding texel at each end of every sub-volume)\n\t//   atlasDepth   = 7 * paddedSlices\n\t//   For sub-volume t the first DATA texel sits at atlas slice t*paddedSlices + 1.\n\t//   Given probe-grid texel-centre UVZ = ( iz + 0.5 ) / nz the atlas UV is:\n\t//     atlasUvZ = ( uvw.z * nz + t * paddedSlices + 1 ) / atlasDepth\n\t//\n\t// uvZBase encodes the nz-scaled Z plus the intra-volume offset (+ 1 for padding),\n\t// so adding t*paddedSlices steps to each successive sub-volume.\n\tfloat nz          = res.z;\n\tfloat paddedSlices = nz + 2.0;\n\tfloat atlasDepth  = 7.0 * paddedSlices;\n\tfloat uvZBase     = uvw.z * nz + 1.0;\n\n\tvec4 s0 = textureLod( probesSH, vec3( uvw.xy, ( uvZBase                       ) / atlasDepth ), 0.0 );\n\tvec4 s1 = textureLod( probesSH, vec3( uvw.xy, ( uvZBase +       paddedSlices   ) / atlasDepth ), 0.0 );\n\tvec4 s2 = textureLod( probesSH, vec3( uvw.xy, ( uvZBase + 2.0 * paddedSlices   ) / atlasDepth ), 0.0 );\n\tvec4 s3 = textureLod( probesSH, vec3( uvw.xy, ( uvZBase + 3.0 * paddedSlices   ) / atlasDepth ), 0.0 );\n\tvec4 s4 = textureLod( probesSH, vec3( uvw.xy, ( uvZBase + 4.0 * paddedSlices   ) / atlasDepth ), 0.0 );\n\tvec4 s5 = textureLod( probesSH, vec3( uvw.xy, ( uvZBase + 5.0 * paddedSlices   ) / atlasDepth ), 0.0 );\n\tvec4 s6 = textureLod( probesSH, vec3( uvw.xy, ( uvZBase + 6.0 * paddedSlices   ) / atlasDepth ), 0.0 );\n\n\t// Unpack 9 vec3 SH L2 coefficients\n\tvec3 c0 = s0.xyz;\n\tvec3 c1 = vec3( s0.w, s1.xy );\n\tvec3 c2 = vec3( s1.zw, s2.x );\n\tvec3 c3 = s2.yzw;\n\tvec3 c4 = s3.xyz;\n\tvec3 c5 = vec3( s3.w, s4.xy );\n\tvec3 c6 = vec3( s4.zw, s5.x );\n\tvec3 c7 = s5.yzw;\n\tvec3 c8 = s6.xyz;\n\n\t// Evaluate L2 irradiance\n\tfloat x = worldNormal.x, y = worldNormal.y, z = worldNormal.z;\n\n\tvec3 result = c0 * 0.886227;\n\tresult += c1 * 2.0 * 0.511664 * y;\n\tresult += c2 * 2.0 * 0.511664 * z;\n\tresult += c3 * 2.0 * 0.511664 * x;\n\tresult += c4 * 2.0 * 0.429043 * x * y;\n\tresult += c5 * 2.0 * 0.429043 * y * z;\n\tresult += c6 * ( 0.743125 * z * z - 0.247708 );\n\tresult += c7 * 2.0 * 0.429043 * x * z;\n\tresult += c8 * 0.429043 * ( x * x - y * y );\n\n\treturn max( result, vec3( 0.0 ) );\n\n}\n\n#endif\n",
    'lights_fragment_begin': '\n/**\n * This is a template that can be used to light a material, it uses pluggable\n * RenderEquations (RE)for specific lighting scenarios.\n *\n * Instructions for use:\n * - Ensure that both RE_Direct, RE_IndirectDiffuse and RE_IndirectSpecular are defined\n * - Create a material parameter that is to be passed as the third parameter to your lighting functions.\n *\n * TODO:\n * - Add area light support.\n * - Add sphere light support.\n * - Add diffuse light probe (irradiance cubemap) support.\n */\n\nvec3 geometryPosition = - vViewPosition;\nvec3 geometryNormal = normal;\nvec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );\n\nvec3 geometryClearcoatNormal = vec3( 0.0 );\n\n#ifdef USE_CLEARCOAT\n\n\tgeometryClearcoatNormal = clearcoatNormal;\n\n#endif\n\n#ifdef USE_IRIDESCENCE\n\n\tfloat dotNVi = saturate( dot( normal, geometryViewDir ) );\n\n\tif ( material.iridescenceThickness == 0.0 ) {\n\n\t\tmaterial.iridescence = 0.0;\n\n\t} else {\n\n\t\tmaterial.iridescence = saturate( material.iridescence );\n\n\t}\n\n\tif ( material.iridescence > 0.0 ) {\n\n\t\tvec3 iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );\n\t\tvec3 iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );\n\n\t\tmaterial.iridescenceFresnel = mix( iridescenceFresnelDielectric, iridescenceFresnelMetallic, material.metalness );\n\n\t\t// Iridescence F0 approximation\n\t\tmaterial.iridescenceF0Dielectric = Schlick_to_F0( iridescenceFresnelDielectric, 1.0, dotNVi );\n\t\tmaterial.iridescenceF0Metallic = Schlick_to_F0( iridescenceFresnelMetallic, 1.0, dotNVi );\n\n\t}\n\n#endif\n\n#ifdef STANDARD\n\n\tfloat dotNVms = saturate( dot( geometryNormal, geometryViewDir ) );\n\n\tmaterial.dfg = textureLod( dfgLUT, vec2( material.roughness, dotNVms ), 0.0 ).rg;\n\n\t#if ( NUM_SUN_LIGHTS > 0 || NUM_DIR_LIGHTS > 0 || NUM_POINT_LIGHTS > 0 || NUM_SPOT_LIGHTS > 0 )\n\n\t\t// Multi-scattering energy compensation for direct lighting\n\t\t// Based on "Practical Multiple Scattering Compensation for Microfacet Models"\n\t\t// https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf\n\n\t\t// Energy of the single-scattering lobe in a white furnace ( F0 = F90 = 1 )\n\t\tfloat EssMs = material.dfg.x + material.dfg.y;\n\n\t\t// Compensate for the energy lost to multiple scattering, tinting the added term by F0 ( equation 16 )\n\t\tmaterial.multiScatteringCompensation = 1.0 + material.specularColorBlended * ( 1.0 / EssMs - 1.0 );\n\n\t#endif\n\n#endif\n\nIncidentLight directLight;\n\n#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )\n\n\tPointLight pointLight;\n\t#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0\n\tPointLightShadow pointLightShadow;\n\t#endif\n\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {\n\n\t\tpointLight = pointLights[ i ];\n\n\t\tgetPointLightInfo( pointLight, geometryPosition, directLight );\n\n\t\t#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )\n\t\tpointLightShadow = pointLightShadows[ i ];\n\t\tdirectLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;\n\t\t#endif\n\n\t\tRE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n\n\t}\n\t#pragma unroll_loop_end\n\n#endif\n\n#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )\n\n\tSpotLight spotLight;\n\tvec4 spotColor;\n\tvec3 spotLightCoord;\n\tbool inSpotLightMap;\n\n\t#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0\n\tSpotLightShadow spotLightShadow;\n\t#endif\n\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {\n\n\t\tspotLight = spotLights[ i ];\n\n\t\tgetSpotLightInfo( spotLight, geometryPosition, directLight );\n\n\t\t// spot lights are ordered [shadows with maps, shadows without maps, maps without shadows, none]\n\t\t#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )\n\t\t#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX\n\t\t#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )\n\t\t#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS\n\t\t#else\n\t\t#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )\n\t\t#endif\n\n\t\t#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )\n\t\t\tspotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;\n\t\t\tinSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );\n\t\t\tspotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );\n\t\t\tdirectLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;\n\t\t#endif\n\n\t\t#undef SPOT_LIGHT_MAP_INDEX\n\n\t\t#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )\n\t\tspotLightShadow = spotLightShadows[ i ];\n\t\tdirectLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;\n\t\t#endif\n\n\t\tRE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n\n\t}\n\t#pragma unroll_loop_end\n\n#endif\n\n#if ( NUM_SUN_LIGHTS > 0 ) && defined( RE_Direct )\n\n\tSunLight sunLight;\n\t#if defined( USE_SHADOWMAP ) && NUM_SUN_LIGHT_SHADOWS > 0\n\tSunLightShadow sunLightShadow;\n\t#endif\n\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_SUN_LIGHTS; i ++ ) {\n\n\t\tsunLight = sunLights[ i ];\n\n\t\tgetSunLightInfo( sunLight, directLight );\n\n\t\t#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SUN_LIGHT_SHADOWS )\n\t\tsunLightShadow = sunLightShadows[ i ];\n\t\tdirectLight.color *= ( directLight.visible && receiveShadow ) ? getSunShadow( sunShadowMap[ i ], sunLightShadow, UNROLLED_LOOP_INDEX ) : 1.0;\n\t\t#endif\n\n\t\tRE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n\n\t}\n\t#pragma unroll_loop_end\n\n#endif\n\n#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )\n\n\tDirectionalLight directionalLight;\n\t#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0\n\tDirectionalLightShadow directionalLightShadow;\n\t#endif\n\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {\n\n\t\tdirectionalLight = directionalLights[ i ];\n\n\t\tgetDirectionalLightInfo( directionalLight, directLight );\n\n\t\t#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )\n\t\tdirectionalLightShadow = directionalLightShadows[ i ];\n\t\tdirectLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;\n\t\t#endif\n\n\t\tRE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n\n\t}\n\t#pragma unroll_loop_end\n\n#endif\n\n#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )\n\n\tRectAreaLight rectAreaLight;\n\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {\n\n\t\trectAreaLight = rectAreaLights[ i ];\n\t\tRE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n\n\t}\n\t#pragma unroll_loop_end\n\n#endif\n\n#if defined( RE_IndirectDiffuse )\n\n\tvec3 irradiance = getAmbientLightIrradiance( ambientLightColor );\n\n\t#if defined( USE_LIGHT_PROBES )\n\n\t\tirradiance += getLightProbeIrradiance( lightProbe, geometryNormal );\n\n\t#endif\n\n\t#if ( NUM_HEMI_LIGHTS > 0 )\n\n\t\t#pragma unroll_loop_start\n\t\tfor ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {\n\n\t\t\tirradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );\n\n\t\t}\n\t\t#pragma unroll_loop_end\n\n\t#endif\n\n\t#if NUM_LIGHT_PROBE_GRIDS > 0\n\n\t\tvec3 probeWorldPos = ( ( vec4( geometryPosition, 1.0 ) - viewMatrix[ 3 ] ) * viewMatrix ).xyz;\n\t\tvec3 probeWorldNormal = transformNormalByInverseViewMatrix( geometryNormal, viewMatrix );\n\n\t\t// Sample the last grid whose cells contain the fragment.\n\n\t\tint probeGrid = - 1;\n\n\t\t#pragma unroll_loop_start\n\t\tfor ( int i = 0; i < NUM_LIGHT_PROBE_GRIDS; i ++ ) {\n\n\t\t\tif ( isInsideLightProbeGrid( probeWorldPos, probeWorldNormal, probesMin[ i ], probesMax[ i ], probesResolution[ i ] ) ) probeGrid = UNROLLED_LOOP_INDEX;\n\n\t\t}\n\t\t#pragma unroll_loop_end\n\n\t\t#pragma unroll_loop_start\n\t\tfor ( int i = 0; i < NUM_LIGHT_PROBE_GRIDS; i ++ ) {\n\n\t\t\tif ( probeGrid == UNROLLED_LOOP_INDEX ) irradiance += getLightProbeGridIrradiance( probesSH[ i ], probesMin[ i ], probesMax[ i ], probesResolution[ i ], probeWorldPos, probeWorldNormal );\n\n\t\t}\n\t\t#pragma unroll_loop_end\n\n\t#endif\n\n#endif\n\n#if defined( RE_IndirectSpecular )\n\n\tvec3 radiance = getIndirectLightRadiance( geometryViewDir, geometryNormal, material.roughness );\n\tvec3 clearcoatRadiance = vec3( 0.0 );\n\n\t#ifdef USE_CLEARCOAT\n\n\t\tclearcoatRadiance += getIndirectLightRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );\n\n\t#endif\n\n#endif\n',
    'lights_fragment_end': '\n#if defined( RE_IndirectDiffuse )\n\n\tRE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n\n#endif\n\n#if defined( RE_IndirectSpecular )\n\n\tRE_IndirectSpecular( radiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );\n\n#endif\n',
    'lights_fragment_maps': '\n#if defined( RE_IndirectDiffuse )\n\n\t#ifdef USE_LIGHTMAP\n\n\t\tvec4 lightMapTexel = texture2D( lightMap, vLightMapUv );\n\t\tvec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;\n\n\t\tirradiance += lightMapIrradiance;\n\n\t#endif\n\n\t#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_PMREM )\n\n\t\t#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )\n\n\t\t\tirradiance += getIBLIrradiance( geometryNormal );\n\n\t\t#endif\n\n\t#endif\n\n#endif\n\n#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )\n\n\t#ifdef USE_ANISOTROPY\n\n\t\tvec3 iblRadiance = getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );\n\n\t#else\n\n\t\tvec3 iblRadiance = getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );\n\n\t#endif\n\n\t#ifdef USE_RETROREFLECTION\n\n\t\t#ifdef USE_ANISOTROPY\n\n\t\t\tvec3 retroIBLRadiance = getIBLAnisotropyRetroRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );\n\n\t\t#else\n\n\t\t\tvec3 retroIBLRadiance = getIBLRetroRadiance( geometryViewDir, geometryNormal, material.roughness );\n\n\t\t#endif\n\n\t\tiblRadiance = mix( iblRadiance, retroIBLRadiance, saturate( material.retroreflectivity ) );\n\n\t#endif\n\n\tradiance += iblRadiance;\n\n\t#ifdef USE_CLEARCOAT\n\n\t\tclearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );\n\n\t#endif\n\n#endif\n',
    'lights_lambert_fragment': '\nLambertMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;\nmaterial.specularStrength = specularStrength;\n',
    'lights_lambert_pars_fragment': '\nvarying vec3 vViewPosition;\n\nstruct LambertMaterial {\n\n\tvec3 diffuseColor;\n\tfloat specularStrength;\n\n};\n\nvoid RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {\n\n\tfloat dotNL = saturate( dot( geometryNormal, directLight.direction ) );\n\tvec3 irradiance = dotNL * directLight.color;\n\n\treflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n\n}\n\nvoid RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {\n\n\treflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n\n}\n\n#define RE_Direct\t\t\t\tRE_Direct_Lambert\n#define RE_IndirectDiffuse\t\tRE_IndirectDiffuse_Lambert\n',
    'lights_pars_begin': "\nuniform bool receiveShadow;\nuniform vec3 ambientLightColor;\n\n#if defined( USE_LIGHT_PROBES )\n\n\tuniform vec3 lightProbe[ 9 ];\n\n#endif\n\n// get the irradiance (radiance convolved with cosine lobe) at the point 'normal' on the unit sphere\n// source: https://graphics.stanford.edu/papers/envmap/envmap.pdf\nvec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {\n\n\t// normal is assumed to have unit length\n\n\tfloat x = normal.x, y = normal.y, z = normal.z;\n\n\t// band 0\n\tvec3 result = shCoefficients[ 0 ] * 0.886227;\n\n\t// band 1\n\tresult += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;\n\tresult += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;\n\tresult += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;\n\n\t// band 2\n\tresult += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;\n\tresult += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;\n\tresult += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );\n\tresult += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;\n\tresult += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );\n\n\treturn result;\n\n}\n\nvec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {\n\n\tvec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );\n\n\tvec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );\n\n\treturn irradiance;\n\n}\n\nvec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {\n\n\tvec3 irradiance = ambientLightColor;\n\n\treturn irradiance;\n\n}\n\nfloat getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {\n\n\t// based upon Frostbite 3 Moving to Physically-based Rendering\n\t// page 32, equation 26: E[window1]\n\t// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf\n\tfloat distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );\n\n\tif ( cutoffDistance > 0.0 ) {\n\n\t\tdistanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );\n\n\t}\n\n\treturn distanceFalloff;\n\n}\n\nfloat getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {\n\n\treturn smoothstep( coneCosine, penumbraCosine, angleCosine );\n\n}\n\n#if NUM_SUN_LIGHTS > 0\n\n\tstruct SunLight {\n\t\tvec3 direction;\n\t\tvec3 color;\n\t};\n\n\tuniform SunLight sunLights[ NUM_SUN_LIGHTS ];\n\n\tvoid getSunLightInfo( const in SunLight sunLight, out IncidentLight light ) {\n\n\t\tlight.color = sunLight.color;\n\t\tlight.direction = sunLight.direction;\n\t\tlight.visible = true;\n\n\t}\n\n#endif\n\n\n#if NUM_DIR_LIGHTS > 0\n\n\tstruct DirectionalLight {\n\t\tvec3 direction;\n\t\tvec3 color;\n\t};\n\n\tuniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];\n\n\tvoid getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {\n\n\t\tlight.color = directionalLight.color;\n\t\tlight.direction = directionalLight.direction;\n\t\tlight.visible = true;\n\n\t}\n\n#endif\n\n\n#if NUM_POINT_LIGHTS > 0\n\n\tstruct PointLight {\n\t\tvec3 position;\n\t\tvec3 color;\n\t\tfloat distance;\n\t\tfloat decay;\n\t};\n\n\tuniform PointLight pointLights[ NUM_POINT_LIGHTS ];\n\n\t// light is an out parameter as having it as a return value caused compiler errors on some devices\n\tvoid getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {\n\n\t\tvec3 lVector = pointLight.position - geometryPosition;\n\n\t\tlight.direction = normalize( lVector );\n\n\t\tfloat lightDistance = length( lVector );\n\n\t\tlight.color = pointLight.color;\n\t\tlight.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );\n\t\tlight.visible = ( light.color != vec3( 0.0 ) );\n\n\t}\n\n#endif\n\n\n#if NUM_SPOT_LIGHTS > 0\n\n\tstruct SpotLight {\n\t\tvec3 position;\n\t\tvec3 direction;\n\t\tvec3 color;\n\t\tfloat distance;\n\t\tfloat decay;\n\t\tfloat coneCos;\n\t\tfloat penumbraCos;\n\t};\n\n\tuniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];\n\n\t// light is an out parameter as having it as a return value caused compiler errors on some devices\n\tvoid getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {\n\n\t\tvec3 lVector = spotLight.position - geometryPosition;\n\n\t\tlight.direction = normalize( lVector );\n\n\t\tfloat angleCos = dot( light.direction, spotLight.direction );\n\n\t\tfloat spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );\n\n\t\tif ( spotAttenuation > 0.0 ) {\n\n\t\t\tfloat lightDistance = length( lVector );\n\n\t\t\tlight.color = spotLight.color * spotAttenuation;\n\t\t\tlight.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );\n\t\t\tlight.visible = ( light.color != vec3( 0.0 ) );\n\n\t\t} else {\n\n\t\t\tlight.color = vec3( 0.0 );\n\t\t\tlight.visible = false;\n\n\t\t}\n\n\t}\n\n#endif\n\n\n#if NUM_RECT_AREA_LIGHTS > 0\n\n\tstruct RectAreaLight {\n\t\tvec3 color;\n\t\tvec3 position;\n\t\tvec3 halfWidth;\n\t\tvec3 halfHeight;\n\t};\n\n\t// Pre-computed values of LinearTransformedCosine approximation of BRDF\n\t// BRDF approximation Texture is 64x64\n\tuniform sampler2D ltc_1; // RGBA Float\n\tuniform sampler2D ltc_2; // RGBA Float\n\n\tuniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];\n\n#endif\n\n\n#if NUM_HEMI_LIGHTS > 0\n\n\tstruct HemisphereLight {\n\t\tvec3 direction;\n\t\tvec3 skyColor;\n\t\tvec3 groundColor;\n\t};\n\n\tuniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];\n\n\tvec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {\n\n\t\tfloat dotNL = dot( normal, hemiLight.direction );\n\t\tfloat hemiDiffuseWeight = 0.5 * dotNL + 0.5;\n\n\t\tvec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );\n\n\t\treturn irradiance;\n\n\t}\n\n#endif\n\n// Ambient lights and light probes stand in for a distant environment, so they reflect their irradiance / PI\nvec3 getIndirectLightRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {\n\n\t// Mixing the reflection with the normal is more accurate for rough objects, see getIBLRadiance()\n\tvec3 reflectVec = normalize( mix( reflect( - viewDir, normal ), normal, pow4( roughness ) ) );\n\n\tvec3 irradiance = getAmbientLightIrradiance( ambientLightColor );\n\n\t#if defined( USE_LIGHT_PROBES )\n\n\t\tirradiance += getLightProbeIrradiance( lightProbe, reflectVec );\n\n\t#endif\n\n\t#if ( NUM_HEMI_LIGHTS > 0 )\n\n\t\t#pragma unroll_loop_start\n\t\tfor ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {\n\n\t\t\tirradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], reflectVec );\n\n\t\t}\n\t\t#pragma unroll_loop_end\n\n\t#endif\n\n\treturn irradiance * RECIPROCAL_PI;\n\n}\n\n#include <lightprobes_pars_fragment>\n",
    'lights_phong_fragment': '\nBlinnPhongMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;\nmaterial.specularColor = specular;\nmaterial.specularShininess = shininess;\nmaterial.specularStrength = specularStrength;\n',
    'lights_phong_pars_fragment': '\nvarying vec3 vViewPosition;\n\nstruct BlinnPhongMaterial {\n\n\tvec3 diffuseColor;\n\tvec3 specularColor;\n\tfloat specularShininess;\n\tfloat specularStrength;\n\n};\n\nvoid RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {\n\n\tfloat dotNL = saturate( dot( geometryNormal, directLight.direction ) );\n\tvec3 irradiance = dotNL * directLight.color;\n\n\treflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n\n\treflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;\n\n}\n\nvoid RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {\n\n\treflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n\n}\n\n#define RE_Direct\t\t\t\tRE_Direct_BlinnPhong\n#define RE_IndirectDiffuse\t\tRE_IndirectDiffuse_BlinnPhong\n',
    'lights_physical_fragment': "\nPhysicalMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;\nmaterial.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );\nmaterial.metalness = metalnessFactor;\n\nvec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );\nfloat geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );\n\n// Minimum roughness, so even a perfect mirror samples a prefiltered level of the environment map.\n// Matches Filament's desktop MIN_PERCEPTUAL_ROUGHNESS: https://github.com/google/filament/blob/main/shaders/src/surface_material.fs\nmaterial.roughness = max( roughnessFactor, 0.045 );\nmaterial.roughness += geometryRoughness;\nmaterial.roughness = min( material.roughness, 1.0 );\n\n#ifdef USE_DIFFUSE_ROUGHNESS\n\n\tfloat diffuseRoughnessFactor = diffuseRoughness;\n\n\t#ifdef USE_DIFFUSE_ROUGHNESSMAP\n\n\t\tdiffuseRoughnessFactor *= texture2D( diffuseRoughnessMap, vDiffuseRoughnessMapUv ).r;\n\n\t#endif\n\n\tmaterial.diffuseRoughness = saturate( diffuseRoughnessFactor );\n\n#endif\n\n#ifdef IOR\n\n\tmaterial.ior = ior;\n\n\t#ifdef USE_SPECULAR\n\n\t\tfloat specularIntensityFactor = specularIntensity;\n\t\tvec3 specularColorFactor = specularColor;\n\n\t\t#ifdef USE_SPECULAR_COLORMAP\n\n\t\t\tspecularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;\n\n\t\t#endif\n\n\t\t#ifdef USE_SPECULAR_INTENSITYMAP\n\n\t\t\tspecularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;\n\n\t\t#endif\n\n\t\tmaterial.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );\n\n\t#else\n\n\t\tfloat specularIntensityFactor = 1.0;\n\t\tvec3 specularColorFactor = vec3( 1.0 );\n\t\tmaterial.specularF90 = 1.0;\n\n\t#endif\n\n\tmaterial.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;\n\tmaterial.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );\n\n#else\n\n\tmaterial.specularColor = vec3( 0.04 );\n\tmaterial.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );\n\tmaterial.specularF90 = 1.0;\n\n#endif\n\n#ifdef USE_CLEARCOAT\n\n\tmaterial.clearcoat = clearcoat;\n\tmaterial.clearcoatRoughness = clearcoatRoughness;\n\tmaterial.clearcoatF0 = vec3( 0.04 );\n\tmaterial.clearcoatF90 = 1.0;\n\n\t#ifdef USE_CLEARCOATMAP\n\n\t\tmaterial.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;\n\n\t#endif\n\n\t#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n\n\t\tmaterial.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;\n\n\t#endif\n\n\tmaterial.clearcoat = saturate( material.clearcoat ); // Burley clearcoat model\n\tmaterial.clearcoatRoughness = max( material.clearcoatRoughness, 0.045 );\n\tmaterial.clearcoatRoughness += geometryRoughness;\n\tmaterial.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );\n\n#endif\n\n#ifdef USE_DISPERSION\n\n\tmaterial.dispersion = dispersion;\n\n#endif\n\n#ifdef USE_RETROREFLECTION\n\n\tmaterial.retroreflectivity = retroreflectivity;\n\n#endif\n\n#ifdef USE_IRIDESCENCE\n\n\tmaterial.iridescence = iridescence;\n\tmaterial.iridescenceIOR = iridescenceIOR;\n\n\t#ifdef USE_IRIDESCENCEMAP\n\n\t\tmaterial.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;\n\n\t#endif\n\n\t#ifdef USE_IRIDESCENCE_THICKNESSMAP\n\n\t\tmaterial.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;\n\n\t#else\n\n\t\tmaterial.iridescenceThickness = iridescenceThicknessMaximum;\n\n\t#endif\n\n#endif\n\n#ifdef USE_SHEEN\n\n\tmaterial.sheenColor = sheenColor;\n\n\t#ifdef USE_SHEEN_COLORMAP\n\n\t\tmaterial.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;\n\n\t#endif\n\n\tmaterial.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );\n\n\t#ifdef USE_SHEEN_ROUGHNESSMAP\n\n\t\tmaterial.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;\n\n\t#endif\n\n#endif\n\n#ifdef USE_ANISOTROPY\n\n\t#ifdef USE_ANISOTROPYMAP\n\n\t\tmat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );\n\t\tvec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;\n\t\tvec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;\n\n\t#else\n\n\t\tvec2 anisotropyV = anisotropyVector;\n\n\t#endif\n\n\tmaterial.anisotropy = length( anisotropyV );\n\n\tif( material.anisotropy == 0.0 ) {\n\t\tanisotropyV = vec2( 1.0, 0.0 );\n\t} else {\n\t\tanisotropyV /= material.anisotropy;\n\t\tmaterial.anisotropy = saturate( material.anisotropy );\n\t}\n\n\t// Roughness along the anisotropy bitangent is the material roughness, while the tangent roughness increases with anisotropy.\n\tmaterial.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );\n\n\tmaterial.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;\n\tmaterial.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;\n\n#endif\n",
    'lights_physical_pars_fragment': '\n\nuniform sampler2D dfgLUT;\n\nstruct PhysicalMaterial {\n\n\tvec3 diffuseColor;\n\tvec3 diffuseContribution;\n\tvec3 specularColor;\n\tvec3 specularColorBlended;\n\n\tfloat roughness;\n\tfloat metalness;\n\tfloat specularF90;\n\tfloat dispersion;\n\tvec2 dfg;\n\tvec3 multiScatteringCompensation;\n\n\t#ifdef USE_DIFFUSE_ROUGHNESS\n\t\tfloat diffuseRoughness;\n\t#endif\n\n\t#ifdef USE_RETROREFLECTION\n\t\tfloat retroreflectivity;\n\t#endif\n\n\t#ifdef USE_CLEARCOAT\n\t\tfloat clearcoat;\n\t\tfloat clearcoatRoughness;\n\t\tvec3 clearcoatF0;\n\t\tfloat clearcoatF90;\n\t#endif\n\n\t#ifdef USE_IRIDESCENCE\n\t\tfloat iridescence;\n\t\tfloat iridescenceIOR;\n\t\tfloat iridescenceThickness;\n\t\tvec3 iridescenceFresnel;\n\t\tvec3 iridescenceF0Dielectric;\n\t\tvec3 iridescenceF0Metallic;\n\t#endif\n\n\t#ifdef USE_SHEEN\n\t\tvec3 sheenColor;\n\t\tfloat sheenRoughness;\n\t#endif\n\n\t#ifdef IOR\n\t\tfloat ior;\n\t#endif\n\n\t#ifdef USE_TRANSMISSION\n\t\tfloat transmission;\n\t\tfloat transmissionAlpha;\n\t\tfloat thickness;\n\t\tfloat attenuationDistance;\n\t\tvec3 attenuationColor;\n\t#endif\n\n\t#ifdef USE_ANISOTROPY\n\t\tfloat anisotropy;\n\t\tfloat alphaT;\n\t\tvec3 anisotropyT;\n\t\tvec3 anisotropyB;\n\t#endif\n\n};\n\n// temporary\nvec3 clearcoatSpecularDirect = vec3( 0.0 );\nvec3 clearcoatSpecularIndirect = vec3( 0.0 );\nvec3 sheenSpecularDirect = vec3( 0.0 );\nvec3 sheenSpecularIndirect = vec3(0.0 );\n\nvec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {\n    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );\n    float x2 = x * x;\n    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );\n\n    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );\n}\n\n#ifdef USE_DIFFUSE_ROUGHNESS\n\n\t// Portsmouth et al. 2025, "EON: A Practical Energy-Preserving Rough Diffuse BRDF"\n\t// https://jcgt.org/published/0014/01/06/\n\tconst float EON_EPSILON = 1e-7;\n\n\tfloat FON_DirectionalAlbedo( const in float mu, const in float roughness, const in float A ) {\n\n\t\tfloat muComp = 1.0 - mu;\n\t\tfloat gOverPi = muComp * ( 0.0571085289 + muComp * ( 0.491881867 + muComp * ( - 0.332181442 + muComp * 0.0714429953 ) ) );\n\n\t\treturn A * ( 1.0 + roughness * gOverPi );\n\n\t}\n\n\tvec3 BRDF_EON( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 diffuseColor, const in float roughness ) {\n\n\t\tvec3 rho = saturate( diffuseColor );\n\n\t\tif ( roughness <= EON_EPSILON ) return BRDF_Lambert( rho );\n\n\t\tfloat dotNL = saturate( dot( normal, lightDir ) );\n\t\tfloat dotNV = saturate( dot( normal, viewDir ) );\n\t\tfloat s = dot( lightDir, viewDir ) - dotNL * dotNV;\n\t\tfloat sOverT = ( s > 0.0 ) ? s / max( max( dotNL, dotNV ), EON_EPSILON ) : s;\n\n\t\tfloat A = 1.0 / ( 1.0 + ( 0.5 - 2.0 / ( 3.0 * PI ) ) * roughness );\n\t\tvec3 singleScatter = rho * RECIPROCAL_PI * A * ( 1.0 + roughness * sOverT );\n\n\t\tfloat averageAlbedo = A * ( 1.0 + ( 2.0 / 3.0 - 28.0 / ( 15.0 * PI ) ) * roughness );\n\t\tfloat albedoV = FON_DirectionalAlbedo( dotNV, roughness, A );\n\t\tfloat albedoL = FON_DirectionalAlbedo( dotNL, roughness, A );\n\n\t\tvec3 rhoMultiScatter = rho * rho * averageAlbedo / max( vec3( EON_EPSILON ), vec3( 1.0 ) - rho * ( 1.0 - averageAlbedo ) );\n\t\tvec3 multiScatter = rhoMultiScatter * RECIPROCAL_PI\n\t\t\t* max( EON_EPSILON, 1.0 - albedoV )\n\t\t\t* max( EON_EPSILON, 1.0 - albedoL )\n\t\t\t/ max( EON_EPSILON, 1.0 - averageAlbedo );\n\n\t\treturn singleScatter + multiScatter;\n\n\t}\n\n\tvec3 EON_DirectionalAlbedo( const in vec3 diffuseColor, const in float roughness, const in float dotNV ) {\n\n\t\tvec3 rho = saturate( diffuseColor );\n\n\t\tif ( roughness <= EON_EPSILON ) return rho;\n\n\t\tfloat A = 1.0 / ( 1.0 + ( 0.5 - 2.0 / ( 3.0 * PI ) ) * roughness );\n\t\tfloat directionalAlbedo = FON_DirectionalAlbedo( dotNV, roughness, A );\n\t\tfloat averageAlbedo = A * ( 1.0 + ( 2.0 / 3.0 - 28.0 / ( 15.0 * PI ) ) * roughness );\n\t\tvec3 rhoMultiScatter = rho * rho * averageAlbedo / max( vec3( EON_EPSILON ), vec3( 1.0 ) - rho * ( 1.0 - averageAlbedo ) );\n\n\t\treturn rho * directionalAlbedo + rhoMultiScatter * ( 1.0 - directionalAlbedo );\n\n\t}\n\n#endif\n\n// Moving Frostbite to Physically Based Rendering 3.0 - page 12, listing 2\n// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf\nfloat V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {\n\n\tfloat a2 = pow2( alpha );\n\tfloat gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );\n\tfloat gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );\n\n\treturn 0.5 / max( gv + gl, EPSILON );\n\n}\n\n// Microfacet Models for Refraction through Rough Surfaces - equation (33)\n// http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html\n// alpha is "roughness squared" in Disney’s reparameterization\nfloat D_GGX( const in float alpha, const in float dotNH ) {\n\n\tfloat a2 = pow2( alpha );\n\n\tfloat denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0; // avoid alpha = 0 with dotNH = 1\n\n\treturn RECIPROCAL_PI * a2 / pow2( denom );\n\n}\n\n// https://google.github.io/filament/Filament.md.html#materialsystem/anisotropicmodel/anisotropicspecularbrdf\n#ifdef USE_ANISOTROPY\n\n\tfloat V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {\n\n\t\tfloat gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );\n\t\tfloat gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );\n\t\treturn 0.5 / max( gv + gl, EPSILON );\n\n\t}\n\n\tfloat D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {\n\n\t\tfloat a2 = alphaT * alphaB;\n\t\thighp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );\n\t\thighp float v2 = dot( v, v );\n\t\tfloat w2 = a2 / v2;\n\n\t\treturn RECIPROCAL_PI * a2 * pow2 ( w2 );\n\n\t}\n\n#endif\n\n#ifdef USE_CLEARCOAT\n\n\t// GGX Distribution, Schlick Fresnel, GGX_SmithCorrelated Visibility\n\tvec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {\n\n\t\tvec3 f0 = material.clearcoatF0;\n\t\tfloat f90 = material.clearcoatF90;\n\t\tfloat roughness = max( material.clearcoatRoughness, 0.045 ); // punctual lights need a minimum roughness to show a highlight\n\n\t\tfloat alpha = pow2( roughness ); // UE4\'s roughness\n\n\t\tvec3 halfDir = normalize( lightDir + viewDir );\n\n\t\tfloat dotNL = saturate( dot( normal, lightDir ) );\n\t\tfloat dotNV = saturate( dot( normal, viewDir ) );\n\t\tfloat dotNH = saturate( dot( normal, halfDir ) );\n\t\tfloat dotVH = saturate( dot( viewDir, halfDir ) );\n\n\t\tvec3 F = F_Schlick( f0, f90, dotVH );\n\n\t\tfloat V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );\n\n\t\tfloat D = D_GGX( alpha, dotNH );\n\n\t\treturn F * ( V * D );\n\n\t}\n\n#endif\n\nvec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {\n\n\tvec3 f0 = material.specularColorBlended;\n\tfloat f90 = material.specularF90;\n\tfloat roughness = max( material.roughness, 0.045 ); // punctual lights need a minimum roughness to show a highlight\n\n\tfloat alpha = pow2( roughness ); // UE4\'s roughness\n\n\tvec3 halfDir = normalize( lightDir + viewDir );\n\n\tfloat dotNL = saturate( dot( normal, lightDir ) );\n\tfloat dotNV = saturate( dot( normal, viewDir ) );\n\tfloat dotNH = saturate( dot( normal, halfDir ) );\n\tfloat dotVH = saturate( dot( viewDir, halfDir ) );\n\n\tvec3 F = F_Schlick( f0, f90, dotVH );\n\n\t#ifdef USE_IRIDESCENCE\n\n\t\tF = mix( F, material.iridescenceFresnel, material.iridescence );\n\n\t#endif\n\n\t#ifdef USE_ANISOTROPY\n\n\t\tfloat dotTL = dot( material.anisotropyT, lightDir );\n\t\tfloat dotTV = dot( material.anisotropyT, viewDir );\n\t\tfloat dotTH = dot( material.anisotropyT, halfDir );\n\t\tfloat dotBL = dot( material.anisotropyB, lightDir );\n\t\tfloat dotBV = dot( material.anisotropyB, viewDir );\n\t\tfloat dotBH = dot( material.anisotropyB, halfDir );\n\n\t\tfloat alphaT = max( material.alphaT, alpha );\n\n\t\tfloat V = V_GGX_SmithCorrelated_Anisotropic( alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );\n\n\t\tfloat D = D_GGX_Anisotropic( alphaT, alpha, dotNH, dotTH, dotBH );\n\n\t#else\n\n\t\tfloat V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );\n\n\t\tfloat D = D_GGX( alpha, dotNH );\n\n\t#endif\n\n\treturn F * ( V * D );\n\n}\n\n// Rect Area Light\n\n// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines\n// by Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt\n// code: https://github.com/selfshadow/ltc_code/\n\nvec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {\n\n\tconst float LUT_SIZE = 64.0;\n\tconst float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;\n\tconst float LUT_BIAS = 0.5 / LUT_SIZE;\n\n\tfloat dotNV = saturate( dot( N, V ) );\n\n\t// texture parameterized by sqrt( GGX alpha ) and sqrt( 1 - cos( theta ) )\n\tvec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );\n\n\tuv = uv * LUT_SCALE + LUT_BIAS;\n\n\treturn uv;\n\n}\n\nfloat LTC_ClippedSphereFormFactor( const in vec3 f ) {\n\n\t// Real-Time Area Lighting: a Journey from Research to Production (p.102)\n\t// An approximation of the form factor of a horizon-clipped rectangle.\n\n\tfloat l = length( f );\n\n\treturn max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );\n\n}\n\nvec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {\n\n\tfloat x = dot( v1, v2 );\n\n\tfloat y = abs( x );\n\n\t// rational polynomial approximation to theta / sin( theta ) / 2PI\n\tfloat a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;\n\tfloat b = 3.4175940 + ( 4.1616724 + y ) * y;\n\tfloat v = a / b;\n\n\tfloat theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;\n\n\treturn cross( v1, v2 ) * theta_sintheta;\n\n}\n\nvec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {\n\n\t// bail if point is on back side of plane of light\n\t// assumes ccw winding order of light vertices\n\tvec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];\n\tvec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];\n\tvec3 lightNormal = cross( v1, v2 );\n\n\tif( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );\n\n\t// construct orthonormal basis around N\n\tvec3 T1, T2;\n\tT1 = normalize( V - N * dot( V, N ) );\n\tT2 = - cross( N, T1 ); // negated from paper; possibly due to a different handedness of world coordinate system\n\n\t// compute transform\n\tmat3 mat = mInv * transpose( mat3( T1, T2, N ) );\n\n\t// transform rect\n\tvec3 coords[ 4 ];\n\tcoords[ 0 ] = mat * ( rectCoords[ 0 ] - P );\n\tcoords[ 1 ] = mat * ( rectCoords[ 1 ] - P );\n\tcoords[ 2 ] = mat * ( rectCoords[ 2 ] - P );\n\tcoords[ 3 ] = mat * ( rectCoords[ 3 ] - P );\n\n\t// project rect onto sphere\n\tcoords[ 0 ] = normalize( coords[ 0 ] );\n\tcoords[ 1 ] = normalize( coords[ 1 ] );\n\tcoords[ 2 ] = normalize( coords[ 2 ] );\n\tcoords[ 3 ] = normalize( coords[ 3 ] );\n\n\t// calculate vector form factor\n\tvec3 vectorFormFactor = vec3( 0.0 );\n\tvectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );\n\tvectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );\n\tvectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );\n\tvectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );\n\n\t// adjust for horizon clipping\n\tfloat result = LTC_ClippedSphereFormFactor( vectorFormFactor );\n\n/*\n\t// alternate method of adjusting for horizon clipping (see reference)\n\t// refactoring required\n\tfloat len = length( vectorFormFactor );\n\tfloat z = vectorFormFactor.z / len;\n\n\tconst float LUT_SIZE = 64.0;\n\tconst float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;\n\tconst float LUT_BIAS = 0.5 / LUT_SIZE;\n\n\t// tabulated horizon-clipped sphere, apparently...\n\tvec2 uv = vec2( z * 0.5 + 0.5, len );\n\tuv = uv * LUT_SCALE + LUT_BIAS;\n\n\tfloat scale = texture2D( ltc_2, uv ).w;\n\n\tfloat result = len * scale;\n*/\n\n\treturn vec3( result );\n\n}\n\n// End Rect Area Light\n\n#if defined( USE_SHEEN )\n\n// https://github.com/google/filament/blob/main/shaders/src/surface_brdf.fs\nfloat D_Charlie( float roughness, float dotNH ) {\n\n\tfloat alpha = pow2( roughness );\n\n\t// Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF"\n\tfloat invAlpha = 1.0 / alpha;\n\tfloat cos2h = dotNH * dotNH;\n\tfloat sin2h = max( 1.0 - cos2h, 0.0078125 ); // 2^(-14/2), so sin2h^2 > 0 in fp16\n\n\treturn ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );\n\n}\n\n// https://github.com/google/filament/blob/main/shaders/src/surface_brdf.fs\nfloat V_Neubelt( float dotNV, float dotNL ) {\n\n\t// Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"\n\treturn saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );\n\n}\n\nvec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {\n\n\tvec3 halfDir = normalize( lightDir + viewDir );\n\n\tfloat dotNL = saturate( dot( normal, lightDir ) );\n\tfloat dotNV = saturate( dot( normal, viewDir ) );\n\tfloat dotNH = saturate( dot( normal, halfDir ) );\n\n\tfloat D = D_Charlie( sheenRoughness, dotNH );\n\tfloat V = V_Neubelt( dotNV, dotNL );\n\n\treturn sheenColor * ( D * V );\n\n}\n\n#endif\n\n// This is a curve-fit approximation to the "Charlie sheen" BRDF integrated over the hemisphere from\n// Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF".\nfloat IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {\n\n\tfloat dotNV = saturate( dot( normal, viewDir ) );\n\n\tfloat r2 = roughness * roughness;\n\tfloat rInv = 1.0 / ( roughness + 0.1 );\n\n\tfloat a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;\n\tfloat b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;\n\n\tfloat DG = exp( a * dotNV + b );\n\n\treturn saturate( DG );\n\n}\n\nvec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {\n\n\tfloat dotNV = saturate( dot( normal, viewDir ) );\n\tvec2 fab = textureLod( dfgLUT, vec2( roughness, dotNV ), 0.0 ).rg;\n\n\treturn specularColor * fab.x + specularF90 * fab.y;\n\n}\n\n// Fdez-Agüera\'s "Multiple-Scattering Microfacet Model for Real-Time Image Based Lighting"\n// Approximates multiscattering in order to preserve energy.\n// http://www.jcgt.org/published/0008/01/03/\n#ifdef USE_IRIDESCENCE\nvoid computeMultiscatteringIridescence( const in vec2 fab, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, inout vec3 singleScatter, inout vec3 multiScatter ) {\n#else\nvoid computeMultiscattering( const in vec2 fab, const in vec3 specularColor, const in float specularF90, inout vec3 singleScatter, inout vec3 multiScatter ) {\n#endif\n\n\t#ifdef USE_IRIDESCENCE\n\n\t\tvec3 Fr = mix( specularColor, iridescenceF0, iridescence );\n\n\t#else\n\n\t\tvec3 Fr = specularColor;\n\n\t#endif\n\n\tvec3 FssEss = Fr * fab.x + specularF90 * fab.y;\n\n\tfloat Ess = fab.x + fab.y;\n\tfloat Ems = 1.0 - Ess;\n\n\tvec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619; // 1/21\n\tvec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );\n\n\tsingleScatter += FssEss;\n\tmultiScatter += Fms * Ems;\n\n}\n\n#if NUM_RECT_AREA_LIGHTS > 0\n\n\tvoid RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {\n\n\t\tvec3 normal = geometryNormal;\n\t\tvec3 viewDir = geometryViewDir;\n\t\tvec3 position = geometryPosition;\n\t\tvec3 lightPos = rectAreaLight.position;\n\t\tvec3 halfWidth = rectAreaLight.halfWidth;\n\t\tvec3 halfHeight = rectAreaLight.halfHeight;\n\t\tvec3 lightColor = rectAreaLight.color;\n\t\tfloat roughness = material.roughness;\n\n\t\tvec3 rectCoords[ 4 ];\n\t\trectCoords[ 0 ] = lightPos + halfWidth - halfHeight; // counterclockwise; light shines in local neg z direction\n\t\trectCoords[ 1 ] = lightPos - halfWidth - halfHeight;\n\t\trectCoords[ 2 ] = lightPos - halfWidth + halfHeight;\n\t\trectCoords[ 3 ] = lightPos + halfWidth + halfHeight;\n\n\t\tvec2 uv = LTC_Uv( normal, viewDir, roughness );\n\n\t\tvec4 t1 = textureLod( ltc_1, uv, 0.0 );\n\t\tvec4 t2 = textureLod( ltc_2, uv, 0.0 );\n\n\t\tmat3 mInv = mat3(\n\t\t\tvec3( t1.x, 0, t1.y ),\n\t\t\tvec3(    0, 1,    0 ),\n\t\t\tvec3( t1.z, 0, t1.w )\n\t\t);\n\n\t\t// LTC Fresnel Approximation by Stephen Hill\n\t\t// http://blog.selfshadow.com/publications/s2016-advances/s2016_ltc_fresnel.pdf\n\t\tvec3 fresnel = ( material.specularColorBlended * t2.x + ( material.specularF90 - material.specularColorBlended ) * t2.y );\n\n\t\treflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );\n\n\t\treflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );\n\n\t\t#ifdef USE_CLEARCOAT\n\n\t\t\tvec3 Ncc = geometryClearcoatNormal;\n\n\t\t\tvec2 uvClearcoat = LTC_Uv( Ncc, viewDir, material.clearcoatRoughness );\n\n\t\t\tvec4 t1Clearcoat = textureLod( ltc_1, uvClearcoat, 0.0 );\n\t\t\tvec4 t2Clearcoat = textureLod( ltc_2, uvClearcoat, 0.0 );\n\n\t\t\tmat3 mInvClearcoat = mat3(\n\t\t\t\tvec3( t1Clearcoat.x, 0, t1Clearcoat.y ),\n\t\t\t\tvec3(             0, 1,             0 ),\n\t\t\t\tvec3( t1Clearcoat.z, 0, t1Clearcoat.w )\n\t\t\t);\n\n\t\t\t// LTC Fresnel Approximation for clearcoat\n\t\t\tvec3 fresnelClearcoat = material.clearcoatF0 * t2Clearcoat.x + ( material.clearcoatF90 - material.clearcoatF0 ) * t2Clearcoat.y;\n\n\t\t\tclearcoatSpecularDirect += lightColor * fresnelClearcoat * LTC_Evaluate( Ncc, viewDir, position, mInvClearcoat, rectCoords );\n\n\t\t#endif\n\n\t}\n\n#endif\n\nvoid RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {\n\n\tfloat dotNL = saturate( dot( geometryNormal, directLight.direction ) );\n\n\tvec3 irradiance = dotNL * directLight.color;\n\n\t#ifdef USE_CLEARCOAT\n\n\t\tfloat dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );\n\n\t\tvec3 ccIrradiance = dotNLcc * directLight.color;\n\n\t\tclearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );\n\n\t#endif\n\n\t#ifdef USE_SHEEN\n \n \t\tsheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );\n \n \t\tfloat sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );\n \t\tfloat sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );\n \n \t\tfloat sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );\n \n \t\tirradiance *= sheenEnergyComp;\n \n \t#endif\n\n\tvec3 specularBRDF = BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );\n\n\t#ifdef USE_RETROREFLECTION\n\n\t\t// Minimal Retroreflective Microfacet Model:\n\t\t// https://jcgt.org/published/0015/01/04/\n\t\tvec3 retroViewDir = reflect( - geometryViewDir, geometryNormal );\n\t\tvec3 retroSpecularBRDF = BRDF_GGX( directLight.direction, retroViewDir, geometryNormal, material );\n\n\t\tspecularBRDF = mix( specularBRDF, retroSpecularBRDF, saturate( material.retroreflectivity ) );\n\n\t#endif\n\n\treflectedLight.directSpecular += irradiance * specularBRDF * material.multiScatteringCompensation;\n\n\t// Light reflected by the specular interface is not available to the diffuse layer ( glTF fresnel_mix )\n\tvec3 halfDir = normalize( directLight.direction + geometryViewDir );\n\tfloat dotVH = saturate( dot( geometryViewDir, halfDir ) );\n\tvec3 F = F_Schlick( material.specularColor, material.specularF90, dotVH );\n\n\t#ifdef USE_RETROREFLECTION\n\n\t\tvec3 retroHalfDir = normalize( directLight.direction + retroViewDir );\n\t\tfloat dotRetroVH = saturate( dot( retroViewDir, retroHalfDir ) );\n\t\tvec3 retroF = F_Schlick( material.specularColor, material.specularF90, dotRetroVH );\n\n\t\tF = mix( F, retroF, saturate( material.retroreflectivity ) );\n\n\t#endif\n\n\t#ifdef USE_DIFFUSE_ROUGHNESS\n\n\t\tvec3 diffuseBRDF = BRDF_EON( directLight.direction, geometryViewDir, geometryNormal, material.diffuseColor, material.diffuseRoughness ) * ( 1.0 - material.metalness );\n\n\t#else\n\n\t\tvec3 diffuseBRDF = BRDF_Lambert( material.diffuseContribution );\n\n\t#endif\n\n\treflectedLight.directDiffuse += irradiance * diffuseBRDF * ( 1.0 - F );\n}\n\nvoid RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {\n\n\t// Energy reflected by the specular lobe is not available to the diffuse layer\n\tvec3 singleScattering = vec3( 0.0 );\n\tvec3 multiScattering = vec3( 0.0 );\n\n\t#ifdef USE_IRIDESCENCE\n\n\t\tcomputeMultiscatteringIridescence( material.dfg, material.specularColor, material.specularF90, material.iridescence, material.iridescenceF0Dielectric, singleScattering, multiScattering );\n\n\t#else\n\n\t\tcomputeMultiscattering( material.dfg, material.specularColor, material.specularF90, singleScattering, multiScattering );\n\n\t#endif\n\n\t#ifdef USE_DIFFUSE_ROUGHNESS\n\n\t\tfloat dotNV = saturate( dot( geometryNormal, geometryViewDir ) );\n\t\t// Irradiance has already integrated the cosine term, so use the directional\n\t\t// albedo to approximate EON under locally uniform incident radiance.\n\t\tvec3 diffuseAlbedo = EON_DirectionalAlbedo( material.diffuseColor, material.diffuseRoughness, dotNV ) * ( 1.0 - material.metalness );\n\t\tvec3 diffuse = irradiance * RECIPROCAL_PI * diffuseAlbedo * ( 1.0 - singleScattering - multiScattering );\n\n\t#else\n\n\t\tvec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - singleScattering - multiScattering );\n\n\t#endif\n\n\t#ifdef USE_SHEEN\n\n\t\tfloat sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );\n\n\t\tsheenSpecularIndirect += irradiance * material.sheenColor * sheenAlbedo * RECIPROCAL_PI;\n\n\t\tfloat sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;\n\n\t\tdiffuse *= sheenEnergyComp;\n\n\t#endif\n\n\treflectedLight.indirectDiffuse += diffuse;\n\n}\n\nvoid RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {\n\n\t#ifdef USE_CLEARCOAT\n\n\t\tclearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );\n\n\t#endif\n\n\t// Compute multiscattering separately for dielectric and metallic, then mix\n\n\tvec3 singleScatteringDielectric = vec3( 0.0 );\n\tvec3 multiScatteringDielectric = vec3( 0.0 );\n\n\tvec3 singleScatteringMetallic = vec3( 0.0 );\n\tvec3 multiScatteringMetallic = vec3( 0.0 );\n\n\t#ifdef USE_IRIDESCENCE\n\n\t\tcomputeMultiscatteringIridescence( material.dfg, material.specularColor, material.specularF90, material.iridescence, material.iridescenceF0Dielectric, singleScatteringDielectric, multiScatteringDielectric );\n\t\tcomputeMultiscatteringIridescence( material.dfg, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceF0Metallic, singleScatteringMetallic, multiScatteringMetallic );\n\n\t#else\n\n\t\tcomputeMultiscattering( material.dfg, material.specularColor, material.specularF90, singleScatteringDielectric, multiScatteringDielectric );\n\t\tcomputeMultiscattering( material.dfg, material.diffuseColor, material.specularF90, singleScatteringMetallic, multiScatteringMetallic );\n\n\t#endif\n\n\t// Mix based on metalness\n\tvec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );\n\tvec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );\n\n\t// The multiple scattering lobe has the shape of the single scattering lobe, so it scales the radiance, see\n\t// Turquin, "Practical multiple scattering compensation for microfacet models"\n\tvec3 indirectSpecular = radiance * ( singleScattering + multiScattering );\n\n\t#ifdef USE_SHEEN\n\n\t\tfloat sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );\n\n\t\tfloat sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;\n\n\t\tindirectSpecular *= sheenEnergyComp;\n\n\t#endif\n\n\treflectedLight.indirectSpecular += indirectSpecular;\n\n}\n\n#define RE_Direct\t\t\t\tRE_Direct_Physical\n#define RE_Direct_RectArea\t\tRE_Direct_RectArea_Physical\n#define RE_IndirectDiffuse\t\tRE_IndirectDiffuse_Physical\n#define RE_IndirectSpecular\t\tRE_IndirectSpecular_Physical\n\n// ref: https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf\nfloat computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {\n\n\treturn saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );\n\n}\n\n// Jimenez et al. 2016, "Practical Realtime Strategies for Accurate Indirect Occlusion"\n// Light surfaces bounce light back into their crevices, so they darken less\nvec3 computeMultiBounceOcclusion( const in float visibility, const in vec3 albedo ) {\n\n\tvec3 a = 2.0404 * albedo - 0.3324;\n\tvec3 b = - 4.7951 * albedo + 0.6417;\n\tvec3 c = 2.7552 * albedo + 0.6903;\n\n\treturn max( vec3( visibility ), ( ( visibility * a + b ) * visibility + c ) * visibility );\n\n}\n',
    'lights_toon_fragment': '\nToonMaterial material;\nmaterial.diffuseColor = diffuseColor.rgb;\n',
    'lights_toon_pars_fragment': '\nvarying vec3 vViewPosition;\n\nstruct ToonMaterial {\n\n\tvec3 diffuseColor;\n\n};\n\nvoid RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {\n\n\tvec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;\n\n\treflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n\n}\n\nvoid RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {\n\n\treflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );\n\n}\n\n#define RE_Direct\t\t\t\tRE_Direct_Toon\n#define RE_IndirectDiffuse\t\tRE_IndirectDiffuse_Toon\n',
    'logdepthbuf_fragment': '\n#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )\n\n\t// Doing a strict comparison with == 1.0 can cause noise artifacts\n\t// on some platforms. See issue #17623.\n\tgl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;\n\n#endif\n',
    'logdepthbuf_pars_fragment': '\n#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )\n\n\tuniform float logDepthBufFC;\n\tvarying float vFragDepth;\n\tvarying float vIsPerspective;\n\n#endif\n',
    'logdepthbuf_pars_vertex': '\n#ifdef USE_LOGARITHMIC_DEPTH_BUFFER\n\n\tvarying float vFragDepth;\n\tvarying float vIsPerspective;\n\n#endif\n',
    'logdepthbuf_vertex': '\n#ifdef USE_LOGARITHMIC_DEPTH_BUFFER\n\n\tvFragDepth = 1.0 + gl_Position.w;\n\tvIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );\n\n#endif\n',
    'map_fragment': '\n#ifdef USE_MAP\n\n\tvec4 sampledDiffuseColor = texture2D( map, vMapUv );\n\n\t#ifdef DECODE_VIDEO_TEXTURE\n\n\t\t// use inline sRGB decode until browsers properly support SRGB8_ALPHA8 with video textures (#26516)\n\n\t\tsampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );\n\n\t#endif\n\n\tdiffuseColor *= sampledDiffuseColor;\n\n#endif\n',
    'map_pars_fragment': '\n#ifdef USE_MAP\n\n\tuniform sampler2D map;\n\n#endif\n',
    'map_particle_fragment': '\n#if defined( USE_MAP ) || defined( USE_ALPHAMAP )\n\n\t#if defined( USE_POINTS_UV )\n\n\t\tvec2 uv = vUv;\n\n\t#else\n\n\t\tvec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;\n\n\t#endif\n\n#endif\n\n#ifdef USE_MAP\n\n\tdiffuseColor *= texture2D( map, uv );\n\n#endif\n\n#ifdef USE_ALPHAMAP\n\n\tdiffuseColor.a *= texture2D( alphaMap, uv ).g;\n\n#endif\n',
    'map_particle_pars_fragment': '\n#if defined( USE_POINTS_UV )\n\n\tvarying vec2 vUv;\n\n#else\n\n\t#if defined( USE_MAP ) || defined( USE_ALPHAMAP )\n\n\t\tuniform mat3 uvTransform;\n\n\t#endif\n\n#endif\n\n#ifdef USE_MAP\n\n\tuniform sampler2D map;\n\n#endif\n\n#ifdef USE_ALPHAMAP\n\n\tuniform sampler2D alphaMap;\n\n#endif\n',
    'metalnessmap_fragment': '\nfloat metalnessFactor = metalness;\n\n#ifdef USE_METALNESSMAP\n\n\tvec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );\n\n\t// reads channel B, compatible with a combined OcclusionRoughnessMetallic (RGB) texture\n\tmetalnessFactor *= texelMetalness.b;\n\n#endif\n',
    'metalnessmap_pars_fragment': '\n#ifdef USE_METALNESSMAP\n\n\tuniform sampler2D metalnessMap;\n\n#endif\n',
    'morphcolor_vertex': '\n#if defined( USE_MORPHCOLORS )\n\n\t// morphTargetBaseInfluence is set based on BufferGeometry.morphTargetsRelative value:\n\t// When morphTargetsRelative is false, this is set to 1 - sum(influences); this results in normal = sum((target - base) * influence)\n\t// When morphTargetsRelative is true, this is set to 1; as a result, all morph targets are simply added to the base after weighting\n\n\t#if defined( USE_COLOR_ALPHA )\n\n\t\tvColor *= morphTargetBaseInfluence;\n\n\t#elif defined( USE_COLOR )\n\n\t\tvColor.rgb *= morphTargetBaseInfluence;\n\n\t#endif\n\n\tfor ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {\n\n\t\t#if defined( USE_COLOR_ALPHA )\n\n\t\t\tif ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];\n\n\t\t#elif defined( USE_COLOR )\n\n\t\t\tif ( morphTargetInfluences[ i ] != 0.0 ) vColor.rgb += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];\n\n\t\t#endif\n\n\t}\n\n#endif\n',
    'morphinstance_vertex': '\n#ifdef USE_INSTANCING_MORPH\n\n\tfloat morphTargetInfluences[ MORPHTARGETS_COUNT ];\n\n\tfloat morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;\n\n\tfor ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {\n\n\t\tmorphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;\n\n\t}\n#endif\n',
    'morphnormal_vertex': '\n#ifdef USE_MORPHNORMALS\n\n\t// morphTargetBaseInfluence is set based on BufferGeometry.morphTargetsRelative value:\n\t// When morphTargetsRelative is false, this is set to 1 - sum(influences); this results in normal = sum((target - base) * influence)\n\t// When morphTargetsRelative is true, this is set to 1; as a result, all morph targets are simply added to the base after weighting\n\tobjectNormal *= morphTargetBaseInfluence;\n\n\tfor ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {\n\n\t\tif ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];\n\n\t}\n\n#endif\n',
    'morphtarget_pars_vertex': '\n#ifdef USE_MORPHTARGETS\n\n\t#ifndef USE_INSTANCING_MORPH\n\n\t\tuniform float morphTargetBaseInfluence;\n\t\tuniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];\n\n\t#endif\n\n\tuniform sampler2DArray morphTargetsTexture;\n\tuniform ivec2 morphTargetsTextureSize;\n\n\tvec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {\n\n\t\tint texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;\n\t\tint y = texelIndex / morphTargetsTextureSize.x;\n\t\tint x = texelIndex - y * morphTargetsTextureSize.x;\n\n\t\tivec3 morphUV = ivec3( x, y, morphTargetIndex );\n\t\treturn texelFetch( morphTargetsTexture, morphUV, 0 );\n\n\t}\n\n#endif\n',
    'morphtarget_vertex': '\n#ifdef USE_MORPHTARGETS\n\n\t// morphTargetBaseInfluence is set based on BufferGeometry.morphTargetsRelative value:\n\t// When morphTargetsRelative is false, this is set to 1 - sum(influences); this results in position = sum((target - base) * influence)\n\t// When morphTargetsRelative is true, this is set to 1; as a result, all morph targets are simply added to the base after weighting\n\ttransformed *= morphTargetBaseInfluence;\n\n\tfor ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {\n\n\t\tif ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];\n\n\t}\n\n#endif\n',
    'normal_fragment_begin': '\nfloat faceDirection = gl_FrontFacing ? 1.0 : - 1.0;\n\n#ifdef FLAT_SHADED\n\n\tvec3 fdx = dFdx( vViewPosition );\n\tvec3 fdy = dFdy( vViewPosition );\n\tvec3 normal = normalize( cross( fdx, fdy ) );\n\n#else\n\n\tvec3 normal = normalize( vNormal );\n\n\t#ifdef DOUBLE_SIDED\n\n\t\tnormal *= faceDirection;\n\n\t#endif\n\n#endif\n\n#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )\n\n\t#ifdef USE_TANGENT\n\n\t\tmat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );\n\n\t#else\n\n\t\tmat3 tbn = getTangentFrame( - vViewPosition, normal,\n\t\t#if defined( USE_NORMALMAP )\n\t\t\tvNormalMapUv\n\t\t#elif defined( USE_CLEARCOAT_NORMALMAP )\n\t\t\tvClearcoatNormalMapUv\n\t\t#else\n\t\t\tvUv\n\t\t#endif\n\t\t);\n\n\t#endif\n\n\t#ifdef DOUBLE_SIDED\n\n\t\ttbn[0] *= faceDirection;\n\t\ttbn[1] *= faceDirection;\n\n\t#endif\n\n#endif\n\n#ifdef USE_CLEARCOAT_NORMALMAP\n\n\t#ifdef USE_TANGENT\n\n\t\tmat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );\n\n\t#else\n\n\t\tmat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );\n\n\t#endif\n\n\t#ifdef DOUBLE_SIDED\n\n\t\ttbn2[0] *= faceDirection;\n\t\ttbn2[1] *= faceDirection;\n\n\t#endif\n\n#endif\n\n// non perturbed normal for clearcoat among others\n\nvec3 nonPerturbedNormal = normal;\n\n',
    'normal_fragment_maps': '\n\n#ifdef USE_NORMALMAP_OBJECTSPACE\n\n\tnormal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0; // overrides both flatShading and attribute normals\n\n\t#ifdef FLIP_SIDED\n\n\t\tnormal = - normal;\n\n\t#endif\n\n\t#ifdef DOUBLE_SIDED\n\n\t\tnormal = normal * faceDirection;\n\n\t#endif\n\n\tnormal = normalize( normalMatrix * normal );\n\n#elif defined( USE_NORMALMAP_TANGENTSPACE )\n\n\tvec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;\n\n\t#if defined( USE_PACKED_NORMALMAP )\n\n\t\tmapN = vec3( mapN.xy, sqrt( saturate( 1.0 - dot( mapN.xy, mapN.xy ) ) ) );\n\n\t#endif\n\n\tmapN.xy *= normalScale;\n\n\tnormal = normalize( tbn * mapN );\n\n#elif defined( USE_BUMPMAP )\n\n\tnormal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );\n\n#endif\n',
    'normal_pars_fragment': '\n#ifndef FLAT_SHADED\n\n\tvarying vec3 vNormal;\n\n\t#ifdef USE_TANGENT\n\n\t\tvarying vec3 vTangent;\n\t\tvarying vec3 vBitangent;\n\n\t#endif\n\n#endif\n',
    'normal_pars_vertex': '\n#ifndef FLAT_SHADED\n\n\tvarying vec3 vNormal;\n\n\t#ifdef USE_TANGENT\n\n\t\tvarying vec3 vTangent;\n\t\tvarying vec3 vBitangent;\n\n\t#endif\n\n#endif\n',
    'normal_vertex': '\n#ifndef FLAT_SHADED // normal is computed with derivatives when FLAT_SHADED\n\n\tvNormal = normalize( transformedNormal );\n\n\t#ifdef USE_TANGENT\n\n\t\tvTangent = normalize( transformedTangent );\n\t\tvBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );\n\n\t\t#ifdef FLIP_SIDED\n\n\t\t\tvBitangent = - vBitangent;\n\n\t\t#endif\n\n\t#endif\n\n#endif\n',
    'normalmap_pars_fragment': '\n#ifdef USE_NORMALMAP\n\n\tuniform sampler2D normalMap;\n\tuniform vec2 normalScale;\n\n#endif\n\n#ifdef USE_NORMALMAP_OBJECTSPACE\n\n\tuniform mat3 normalMatrix;\n\n#endif\n\n#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )\n\n\t// Normal Mapping Without Precomputed Tangents\n\t// http://www.thetenthplanet.de/archives/1180\n\n\tmat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {\n\n\t\tvec3 q0 = dFdx( eye_pos.xyz );\n\t\tvec3 q1 = dFdy( eye_pos.xyz );\n\t\tvec2 st0 = dFdx( uv.st );\n\t\tvec2 st1 = dFdy( uv.st );\n\n\t\tvec3 N = surf_norm; // normalized\n\n\t\tvec3 q1perp = cross( q1, N );\n\t\tvec3 q0perp = cross( N, q0 );\n\n\t\tvec3 T = q1perp * st0.x + q0perp * st1.x;\n\t\tvec3 B = q1perp * st0.y + q0perp * st1.y;\n\n\t\tfloat det = max( dot( T, T ), dot( B, B ) );\n\t\tfloat scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );\n\n\t\treturn mat3( T * scale, B * scale, N );\n\n\t}\n\n#endif\n',
    'opaque_fragment': '\n#ifdef OPAQUE\ndiffuseColor.a = 1.0;\n#endif\n\n#ifdef USE_TRANSMISSION\ndiffuseColor.a *= material.transmissionAlpha;\n#endif\n\ngl_FragColor = vec4( outgoingLight, diffuseColor.a );\n',
    'packing': '\nvec3 packNormalToRGB( const in vec3 normal ) {\n\treturn normalize( normal ) * 0.5 + 0.5;\n}\n\nvec3 unpackRGBToNormal( const in vec3 rgb ) {\n\treturn 2.0 * rgb.xyz - 1.0;\n}\n\nconst float PackUpscale = 256. / 255.; // fraction -> 0..1 (including 1)\nconst float UnpackDownscale = 255. / 256.; // 0..1 -> fraction (excluding 1)\nconst float ShiftRight8 = 1. / 256.;\nconst float Inv255 = 1. / 255.;\n\nconst vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );\n\nconst vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );\nconst vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );\nconst vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );\n\nvec4 packDepthToRGBA( const in float v ) {\n\tif( v <= 0.0 )\n\t\treturn vec4( 0., 0., 0., 0. );\n\tif( v >= 1.0 )\n\t\treturn vec4( 1., 1., 1., 1. );\n\tfloat vuf;\n\tfloat af = modf( v * PackFactors.a, vuf );\n\tfloat bf = modf( vuf * ShiftRight8, vuf );\n\tfloat gf = modf( vuf * ShiftRight8, vuf );\n\treturn vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );\n}\n\nvec3 packDepthToRGB( const in float v ) {\n\tif( v <= 0.0 )\n\t\treturn vec3( 0., 0., 0. );\n\tif( v >= 1.0 )\n\t\treturn vec3( 1., 1., 1. );\n\tfloat vuf;\n\tfloat bf = modf( v * PackFactors.b, vuf );\n\tfloat gf = modf( vuf * ShiftRight8, vuf );\n\t// the 0.9999 tweak is unimportant, very tiny empirical improvement\n\t// return vec3( vuf * Inv255, gf * PackUpscale, bf * 0.9999 );\n\treturn vec3( vuf * Inv255, gf * PackUpscale, bf );\n}\n\nvec2 packDepthToRG( const in float v ) {\n\tif( v <= 0.0 )\n\t\treturn vec2( 0., 0. );\n\tif( v >= 1.0 )\n\t\treturn vec2( 1., 1. );\n\tfloat vuf;\n\tfloat gf = modf( v * 256., vuf );\n\treturn vec2( vuf * Inv255, gf );\n}\n\nfloat unpackRGBAToDepth( const in vec4 v ) {\n\treturn dot( v, UnpackFactors4 );\n}\n\nfloat unpackRGBToDepth( const in vec3 v ) {\n\treturn dot( v, UnpackFactors3 );\n}\n\nfloat unpackRGToDepth( const in vec2 v ) {\n\treturn v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;\n}\n\nvec4 pack2HalfToRGBA( const in vec2 v ) {\n\tvec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );\n\treturn vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );\n}\n\nvec2 unpackRGBATo2Half( const in vec4 v ) {\n\treturn vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );\n}\n\n// NOTE: viewZ, the z-coordinate in camera space, is negative for points in front of the camera\n\nfloat viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {\n\t// -near maps to 0; -far maps to 1\n\treturn ( viewZ + near ) / ( near - far );\n}\n\nfloat orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {\n\n\t#ifdef USE_REVERSED_DEPTH_BUFFER\n\t\n\t\treturn depth * ( far - near ) - far;\n\n\t#else\n\n\t\treturn depth * ( near - far ) - near;\n\n\t#endif\n}\n\n// NOTE: https://twitter.com/gonnavis/status/1377183786949959682\n\nfloat viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {\n\t// -near maps to 0; -far maps to 1\n\treturn ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );\n}\n\nfloat perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {\n\t\n\t#ifdef USE_REVERSED_DEPTH_BUFFER\n\n\t\treturn ( near * far ) / ( ( near - far ) * depth - near );\n\n\t#else\n\n\t\treturn ( near * far ) / ( ( far - near ) * depth - far );\n\n\t#endif\n}\n',
    'premultiplied_alpha_fragment': '\n#ifdef PREMULTIPLIED_ALPHA\n\n\tgl_FragColor.rgb *= gl_FragColor.a;\n\n#endif\n',
    'project_vertex': '\nvec4 mvPosition = vec4( transformed, 1.0 );\n\n#ifdef USE_BATCHING\n\n\tmvPosition = batchingMatrix * mvPosition;\n\n#endif\n\n#ifdef USE_INSTANCING\n\n\tmvPosition = instanceMatrix * mvPosition;\n\n#endif\n\nmvPosition = modelViewMatrix * mvPosition;\n\ngl_Position = projectionMatrix * mvPosition;\n',
    'roughnessmap_fragment': '\nfloat roughnessFactor = roughness;\n\n#ifdef USE_ROUGHNESSMAP\n\n\tvec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );\n\n\t// reads channel G, compatible with a combined OcclusionRoughnessMetallic (RGB) texture\n\troughnessFactor *= texelRoughness.g;\n\n#endif\n',
    'roughnessmap_pars_fragment': '\n#ifdef USE_ROUGHNESSMAP\n\n\tuniform sampler2D roughnessMap;\n\n#endif\n',
    'shadowmap_pars_fragment': "\n#if NUM_SPOT_LIGHT_COORDS > 0\n\n\tvarying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];\n\n#endif\n\n#if NUM_SPOT_LIGHT_MAPS > 0\n\n\tuniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];\n\n#endif\n\n#ifdef USE_SHADOWMAP\n\n\t#if NUM_SUN_LIGHT_SHADOWS > 0\n\n\t\t// must match the cascade count in SunLightShadow\n\n\t\t#define SUN_LIGHT_CASCADES 2\n\n\t\t#if defined( SHADOWMAP_TYPE_PCF )\n\n\t\t\tuniform sampler2DShadow sunShadowMap[ NUM_SUN_LIGHT_SHADOWS ];\n\n\t\t#else\n\n\t\t\tuniform sampler2D sunShadowMap[ NUM_SUN_LIGHT_SHADOWS ];\n\n\t\t#endif\n\n\t\tuniform mat4 sunShadowMatrix[ NUM_SUN_LIGHT_SHADOWS * SUN_LIGHT_CASCADES ];\n\t\tuniform vec4 sunShadowCascade[ NUM_SUN_LIGHT_SHADOWS * SUN_LIGHT_CASCADES ];\n\t\tvarying vec4 vSunShadowWorldPosition;\n\t\tvarying vec3 vSunShadowWorldNormal;\n\n\t\tstruct SunLightShadow {\n\t\t\tfloat shadowIntensity;\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t};\n\n\t\tuniform SunLightShadow sunLightShadows[ NUM_SUN_LIGHT_SHADOWS ];\n\n\t#endif\n\n\t#if NUM_DIR_LIGHT_SHADOWS > 0\n\n\t\t#if defined( SHADOWMAP_TYPE_PCF )\n\n\t\t\tuniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];\n\n\t\t#else\n\n\t\t\tuniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];\n\n\t\t#endif\n\n\t\tvarying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];\n\n\t\tstruct DirectionalLightShadow {\n\t\t\tfloat shadowIntensity;\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t};\n\n\t\tuniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];\n\n\t#endif\n\n\t#if NUM_SPOT_LIGHT_SHADOWS > 0\n\n\t\t#if defined( SHADOWMAP_TYPE_PCF )\n\n\t\t\tuniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];\n\n\t\t#else\n\n\t\t\tuniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];\n\n\t\t#endif\n\n\t\tstruct SpotLightShadow {\n\t\t\tfloat shadowIntensity;\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t};\n\n\t\tuniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];\n\n\t#endif\n\n\t#if NUM_POINT_LIGHT_SHADOWS > 0\n\n\t\t#if defined( SHADOWMAP_TYPE_PCF )\n\n\t\t\tuniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];\n\n\t\t#elif defined( SHADOWMAP_TYPE_BASIC )\n\n\t\t\tuniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];\n\n\t\t#endif\n\n\t\tvarying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];\n\n\t\tstruct PointLightShadow {\n\t\t\tfloat shadowIntensity;\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t\tfloat shadowCameraNear;\n\t\t\tfloat shadowCameraFar;\n\t\t};\n\n\t\tuniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];\n\n\t#endif\n\n\t#if defined( SHADOWMAP_TYPE_PCF )\n\n\t\t// Interleaved Gradient Noise for randomizing sampling patterns\n\t\tfloat interleavedGradientNoise( vec2 position ) {\n\n\t\t\treturn fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );\n\n\t\t}\n\n\t\t// Vogel disk sampling for uniform circular distribution\n\t\tvec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {\n\n\t\t\tconst float goldenAngle = 2.399963229728653;\n\t\t\tfloat r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );\n\t\t\tfloat theta = float( sampleIndex ) * goldenAngle + phi;\n\t\t\treturn vec2( cos( theta ), sin( theta ) ) * r;\n\n\t\t}\n\n\t#endif\n\n\t#if defined( SHADOWMAP_TYPE_PCF )\n\n\t\tfloat getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {\n\n\t\t\tfloat shadow = 1.0;\n\n\t\t\tshadowCoord.xyz /= shadowCoord.w;\n\n\t\t\t#ifdef USE_REVERSED_DEPTH_BUFFER\n\n\t\t\t\tshadowCoord.z -= shadowBias;\n\n\t\t\t#else\n\n\t\t\t\tshadowCoord.z += shadowBias;\n\n\t\t\t#endif\n\n\t\t\tbool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;\n\t\t\tbool frustumTest = inFrustum && shadowCoord.z <= 1.0;\n\n\t\t\tif ( frustumTest ) {\n\n\t\t\t\t// Hardware PCF with LinearFilter gives us 4-tap filtering per sample\n\t\t\t\t// 5 samples using Vogel disk + IGN = effectively 20 filtered taps with better distribution\n\t\t\t\tvec2 texelSize = vec2( 1.0 ) / shadowMapSize;\n\t\t\t\tvec2 radius = shadowRadius * texelSize;\n\n\t\t\t\t// Use IGN to rotate sampling pattern per pixel\n\t\t\t\tfloat phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;\n\n\t\t\t\tshadow = (\n\t\t\t\t\ttexture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +\n\t\t\t\t\ttexture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +\n\t\t\t\t\ttexture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +\n\t\t\t\t\ttexture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +\n\t\t\t\t\ttexture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )\n\t\t\t\t) * 0.2;\n\n\t\t\t}\n\n\t\t\treturn mix( 1.0, shadow, shadowIntensity );\n\n\t\t}\n\n\t#elif defined( SHADOWMAP_TYPE_VSM )\n\n\t\tfloat getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {\n\n\t\t\tfloat shadow = 1.0;\n\n\t\t\tshadowCoord.xyz /= shadowCoord.w;\n\n\t\t\t#ifdef USE_REVERSED_DEPTH_BUFFER\n\n\t\t\t\tshadowCoord.z -= shadowBias;\n\n\t\t\t#else\n\n\t\t\t\tshadowCoord.z += shadowBias;\n\n\t\t\t#endif\n\n\t\t\tbool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;\n\t\t\tbool frustumTest = inFrustum && shadowCoord.z <= 1.0;\n\n\t\t\tif ( frustumTest ) {\n\n\t\t\t\tvec2 distribution = textureLod( shadowMap, shadowCoord.xy, 0.0 ).rg;\n\n\t\t\t\tfloat mean = distribution.x;\n\t\t\t\tfloat variance = distribution.y * distribution.y;\n\n\t\t\t\t#ifdef USE_REVERSED_DEPTH_BUFFER\n\n\t\t\t\t\tfloat hard_shadow = step( mean, shadowCoord.z );\n\n\t\t\t\t#else\n\n\t\t\t\t\tfloat hard_shadow = step( shadowCoord.z, mean );\n\n\t\t\t\t#endif\n\t\t\t\t\n\t\t\t\t// Early return if fully lit\n\t\t\t\tif ( hard_shadow == 1.0 ) {\n\n\t\t\t\t\tshadow = 1.0;\n\n\t\t\t\t} else {\n\n\t\t\t\t\t// Variance must be non-zero to avoid division by zero\n\t\t\t\t\tvariance = max( variance, 0.0000001 );\n\n\t\t\t\t\t// Distance from mean\n\t\t\t\t\tfloat d = shadowCoord.z - mean;\n\n\t\t\t\t\t// Chebyshev's inequality for upper bound on probability\n\t\t\t\t\tfloat p_max = variance / ( variance + d * d );\n\n\t\t\t\t\t// Reduce light bleeding by remapping [amount, 1] to [0, 1]\n\t\t\t\t\tp_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );\n\n\t\t\t\t\tshadow = max( hard_shadow, p_max );\n\n\t\t\t\t}\n\n\t\t\t}\n\n\t\t\treturn mix( 1.0, shadow, shadowIntensity );\n\n\t\t}\n\n\t#else // SHADOWMAP_TYPE_BASIC\n\n\t\tfloat getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {\n\n\t\t\tfloat shadow = 1.0;\n\n\t\t\tshadowCoord.xyz /= shadowCoord.w;\n\n\t\t\t#ifdef USE_REVERSED_DEPTH_BUFFER\n\n\t\t\t\tshadowCoord.z -= shadowBias;\n\n\t\t\t#else\n\n\t\t\t\tshadowCoord.z += shadowBias;\n\n\t\t\t#endif\n\n\t\t\tbool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;\n\t\t\tbool frustumTest = inFrustum && shadowCoord.z <= 1.0;\n\n\t\t\tif ( frustumTest ) {\n\n\t\t\t\tfloat depth = textureLod( shadowMap, shadowCoord.xy, 0.0 ).r;\n\n\t\t\t\t#ifdef USE_REVERSED_DEPTH_BUFFER\n\n\t\t\t\t\tshadow = step( depth, shadowCoord.z );\n\n\t\t\t\t#else\n\n\t\t\t\t\tshadow = step( shadowCoord.z, depth );\n\n\t\t\t\t#endif\n\n\t\t\t}\n\n\t\t\treturn mix( 1.0, shadow, shadowIntensity );\n\n\t\t}\n\n\t#endif\n\n\t#if NUM_SUN_LIGHT_SHADOWS > 0\n\n\t\tfloat getSunShadow(\n\t\t\t#if defined( SHADOWMAP_TYPE_PCF )\n\t\t\t\tsampler2DShadow shadowMap,\n\t\t\t#else\n\t\t\t\tsampler2D shadowMap,\n\t\t\t#endif\n\t\t\tSunLightShadow sunLightShadow,\n\t\t\tint shadowIndex\n\t\t) {\n\n\t\t\tvec4 shadowWorldPosition = vec4( vSunShadowWorldPosition.xyz + vSunShadowWorldNormal * sunLightShadow.shadowNormalBias, 1.0 );\n\t\t\tfloat viewDepth = vSunShadowWorldPosition.w;\n\t\t\tint cascadeOffset = shadowIndex * SUN_LIGHT_CASCADES;\n\n\t\t\tfloat shadow = 1.0;\n\n\t\t\t// walk the cascades back to front so each fade band can blend with the shadow behind it\n\n\t\t\tfor ( int i = SUN_LIGHT_CASCADES - 1; i >= 0; i -- ) {\n\n\t\t\t\t// ( begin, end, fade start ) view depths of the cascade\n\n\t\t\t\tvec4 cascade = sunShadowCascade[ cascadeOffset + i ];\n\n\t\t\t\tif ( viewDepth >= cascade.x && viewDepth < cascade.y ) {\n\n\t\t\t\t\tfloat cascadeShadow = getShadow(\n\t\t\t\t\t\tshadowMap,\n\t\t\t\t\t\tsunLightShadow.shadowMapSize,\n\t\t\t\t\t\tsunLightShadow.shadowIntensity,\n\t\t\t\t\t\tsunLightShadow.shadowBias,\n\t\t\t\t\t\tsunLightShadow.shadowRadius,\n\t\t\t\t\t\tsunShadowMatrix[ cascadeOffset + i ] * shadowWorldPosition\n\t\t\t\t\t);\n\n\t\t\t\t\tshadow = mix( cascadeShadow, shadow, smoothstep( cascade.z, cascade.y, viewDepth ) );\n\n\t\t\t\t}\n\n\t\t\t}\n\n\t\t\treturn shadow;\n\n\t\t}\n\n\t#endif\n\n\t#if NUM_POINT_LIGHT_SHADOWS > 0\n\n\t#if defined( SHADOWMAP_TYPE_PCF )\n\n\tfloat getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {\n\n\t\tfloat shadow = 1.0;\n\n\t\t// for point lights, the uniform @vShadowCoord is re-purposed to hold\n\t\t// the vector from the light to the world-space position of the fragment.\n\t\tvec3 lightToPosition = shadowCoord.xyz;\n\n\t\t// Direction from light to fragment\n\t\tvec3 bd3D = normalize( lightToPosition );\n\n\t\t// For cube shadow maps, depth is stored as distance along each face's view axis, not radial distance\n\t\t// The view-space depth is the maximum component of the direction vector (which face is sampled)\n\t\tvec3 absVec = abs( lightToPosition );\n\t\tfloat viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );\n\n\t\tif ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {\n\n\t\t\t// Each sample is compared against the depth of the point next to the fragment in its direction,\n\t\t\t// on whichever face of the cube it falls, so the comparison is the same on every face. A face\n\t\t\t// stores the depth along its own axis, which for the point at direction * lightDistance is\n\t\t\t// lightDistance * max( |x|, |y|, |z| ) of the direction, and perspective depth is a + b / z.\n\t\t\tfloat lightDistance = length( lightToPosition );\n\t\t\tfloat depthRange = shadowCameraFar - shadowCameraNear;\n\n\t\t\t#ifdef USE_REVERSED_DEPTH_BUFFER\n\n\t\t\t\tfloat depthA = - shadowCameraNear / depthRange - shadowBias;\n\t\t\t\tfloat depthB = shadowCameraNear * shadowCameraFar / ( depthRange * lightDistance );\n\n\t\t\t#else\n\n\t\t\t\tfloat depthA = shadowCameraFar / depthRange + shadowBias;\n\t\t\t\tfloat depthB = - shadowCameraNear * shadowCameraFar / ( depthRange * lightDistance );\n\n\t\t\t#endif\n\n\t\t\t// Hardware PCF with LinearFilter gives us 4-tap filtering per sample\n\t\t\t// Use Vogel disk + IGN sampling for better quality\n\t\t\tfloat texelSize = shadowRadius / shadowMapSize.x;\n\n\t\t\t// Build a tangent-space coordinate system for applying offsets\n\t\t\tvec3 absDir = abs( bd3D );\n\t\t\tvec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );\n\t\t\ttangent = normalize( cross( bd3D, tangent ) );\n\t\t\tvec3 bitangent = cross( bd3D, tangent );\n\n\t\t\t// Use IGN to rotate sampling pattern per pixel\n\t\t\tfloat phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;\n\n\t\t\tshadow = 0.0;\n\n\t\t\tfor ( int i = 0; i < 5; i ++ ) {\n\n\t\t\t\tvec2 offset = vogelDiskSample( i, 5, phi );\n\t\t\t\tvec3 direction = bd3D + ( tangent * offset.x + bitangent * offset.y ) * texelSize;\n\t\t\t\tvec3 absDirection = abs( direction );\n\n\t\t\t\tfloat dp = depthA + depthB / max( max( absDirection.x, absDirection.y ), absDirection.z );\n\n\t\t\t\tshadow += texture( shadowMap, vec4( direction, dp ) );\n\n\t\t\t}\n\n\t\t\tshadow *= 0.2;\n\n\t\t}\n\n\t\treturn mix( 1.0, shadow, shadowIntensity );\n\n\t}\n\n\t#elif defined( SHADOWMAP_TYPE_BASIC )\n\n\tfloat getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {\n\n\t\tfloat shadow = 1.0;\n\n\t\t// for point lights, the uniform @vShadowCoord is re-purposed to hold\n\t\t// the vector from the light to the world-space position of the fragment.\n\t\tvec3 lightToPosition = shadowCoord.xyz;\n\n\t\t// For cube shadow maps, depth is stored as distance along each face's view axis, not radial distance\n\t\t// The view-space depth is the maximum component of the direction vector (which face is sampled)\n\t\tvec3 absVec = abs( lightToPosition );\n\t\tfloat viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );\n\n\t\tif ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {\n\n\t\t\t// viewZ to perspective depth\n\n\t\t\tfloat dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );\n\t\t\tdp += shadowBias;\n\n\t\t\t// Direction from light to fragment\n\t\t\tvec3 bd3D = normalize( lightToPosition );\n\n\t\t\tfloat depth = textureLod( shadowMap, bd3D, 0.0 ).r;\n\n\t\t\t#ifdef USE_REVERSED_DEPTH_BUFFER\n\n\t\t\t\tdepth = 1.0 - depth;\n\n\t\t\t#endif\n\n\t\t\tshadow = step( dp, depth );\n\n\t\t}\n\n\t\treturn mix( 1.0, shadow, shadowIntensity );\n\n\t}\n\n\t#endif\n\n\t#endif\n\n#endif\n",
    'shadowmap_pars_vertex': '\n\n#if NUM_SPOT_LIGHT_COORDS > 0\n\n\tuniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];\n\tvarying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];\n\n#endif\n\n#ifdef USE_SHADOWMAP\n\n\t#if NUM_SUN_LIGHT_SHADOWS > 0\n\n\t\t// cascade selection and shadow coordinates are computed per fragment\n\n\t\tvarying vec4 vSunShadowWorldPosition;\n\t\tvarying vec3 vSunShadowWorldNormal;\n\n\t#endif\n\n\t#if NUM_DIR_LIGHT_SHADOWS > 0\n\n\t\tuniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];\n\t\tvarying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];\n\n\t\tstruct DirectionalLightShadow {\n\t\t\tfloat shadowIntensity;\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t};\n\n\t\tuniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];\n\n\t#endif\n\n\t#if NUM_SPOT_LIGHT_SHADOWS > 0\n\n\t\tstruct SpotLightShadow {\n\t\t\tfloat shadowIntensity;\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t};\n\n\t\tuniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];\n\n\t#endif\n\n\t#if NUM_POINT_LIGHT_SHADOWS > 0\n\n\t\tuniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];\n\t\tvarying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];\n\n\t\tstruct PointLightShadow {\n\t\t\tfloat shadowIntensity;\n\t\t\tfloat shadowBias;\n\t\t\tfloat shadowNormalBias;\n\t\t\tfloat shadowRadius;\n\t\t\tvec2 shadowMapSize;\n\t\t\tfloat shadowCameraNear;\n\t\t\tfloat shadowCameraFar;\n\t\t};\n\n\t\tuniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];\n\n\t#endif\n\n\t/*\n\t#if NUM_RECT_AREA_LIGHTS > 0\n\n\t\t// TODO (abelnation): uniforms for area light shadows\n\n\t#endif\n\t*/\n\n#endif\n',
    'shadowmap_vertex': '\n\n#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_SUN_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )\n\n\t#ifdef HAS_NORMAL\n\n\t\t// Offsetting the position used for querying occlusion along the world normal can be used to reduce shadow acne.\n\n\t\tvec3 shadowWorldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );\n\n\t#else\n\n\t\tvec3 shadowWorldNormal = vec3( 0.0 ); // fallback, see #21483\n\n\t#endif\n\n\tvec4 shadowWorldPosition;\n\n#endif\n\n#if defined( USE_SHADOWMAP )\n\n\t#if NUM_SUN_LIGHT_SHADOWS > 0\n\n\t\tvSunShadowWorldPosition = vec4( worldPosition.xyz, - mvPosition.z );\n\t\tvSunShadowWorldNormal = shadowWorldNormal;\n\n\t#endif\n\n\t#if NUM_DIR_LIGHT_SHADOWS > 0\n\n\t\t#pragma unroll_loop_start\n\t\tfor ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {\n\n\t\t\tshadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );\n\t\t\tvDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;\n\n\t\t}\n\t\t#pragma unroll_loop_end\n\n\t#endif\n\n\t#if NUM_POINT_LIGHT_SHADOWS > 0\n\n\t\t#pragma unroll_loop_start\n\t\tfor ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {\n\n\t\t\tshadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );\n\t\t\tvPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;\n\n\t\t}\n\t\t#pragma unroll_loop_end\n\n\t#endif\n\n\t/*\n\t#if NUM_RECT_AREA_LIGHTS > 0\n\n\t\t// TODO (abelnation): update vAreaShadowCoord with area light info\n\n\t#endif\n\t*/\n\n#endif\n\n// spot lights can be evaluated without active shadow mapping (when SpotLight.map is used)\n\n#if NUM_SPOT_LIGHT_COORDS > 0\n\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {\n\n\t\tshadowWorldPosition = worldPosition;\n\t\t#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )\n\t\t\tshadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;\n\t\t#endif\n\t\tvSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;\n\n\t}\n\t#pragma unroll_loop_end\n\n#endif\n\n\n',
    'shadowmask_pars_fragment': '\nfloat getShadowMask() {\n\n\tfloat shadow = 1.0;\n\n\t#ifdef USE_SHADOWMAP\n\n\t#if NUM_SUN_LIGHT_SHADOWS > 0\n\n\tSunLightShadow sunLight;\n\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_SUN_LIGHT_SHADOWS; i ++ ) {\n\n\t\tsunLight = sunLightShadows[ i ];\n\t\tshadow *= receiveShadow ? getSunShadow( sunShadowMap[ i ], sunLight, UNROLLED_LOOP_INDEX ) : 1.0;\n\n\t}\n\t#pragma unroll_loop_end\n\n\t#endif\n\n\t#if NUM_DIR_LIGHT_SHADOWS > 0\n\n\tDirectionalLightShadow directionalLight;\n\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {\n\n\t\tdirectionalLight = directionalLightShadows[ i ];\n\t\tshadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;\n\n\t}\n\t#pragma unroll_loop_end\n\n\t#endif\n\n\t#if NUM_SPOT_LIGHT_SHADOWS > 0\n\n\tSpotLightShadow spotLight;\n\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {\n\n\t\tspotLight = spotLightShadows[ i ];\n\t\tshadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;\n\n\t}\n\t#pragma unroll_loop_end\n\n\t#endif\n\n\t#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )\n\n\tPointLightShadow pointLight;\n\n\t#pragma unroll_loop_start\n\tfor ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {\n\n\t\tpointLight = pointLightShadows[ i ];\n\t\tshadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;\n\n\t}\n\t#pragma unroll_loop_end\n\n\t#endif\n\n\t/*\n\t#if NUM_RECT_AREA_LIGHTS > 0\n\n\t\t// TODO (abelnation): update shadow for Area light\n\n\t#endif\n\t*/\n\n\t#endif\n\n\treturn shadow;\n\n}\n',
    'skinbase_vertex': '\n#ifdef USE_SKINNING\n\n\tmat4 boneMatX = getBoneMatrix( skinIndex.x );\n\tmat4 boneMatY = getBoneMatrix( skinIndex.y );\n\tmat4 boneMatZ = getBoneMatrix( skinIndex.z );\n\tmat4 boneMatW = getBoneMatrix( skinIndex.w );\n\n#endif\n',
    'skinning_pars_vertex': '\n#ifdef USE_SKINNING\n\n\tuniform mat4 bindMatrix;\n\tuniform mat4 bindMatrixInverse;\n\n\tuniform highp sampler2D boneTexture;\n\n\tmat4 getBoneMatrix( const in float i ) {\n\n\t\tint size = textureSize( boneTexture, 0 ).x;\n\t\tint j = int( i ) * 4;\n\t\tint x = j % size;\n\t\tint y = j / size;\n\t\tvec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );\n\t\tvec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );\n\t\tvec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );\n\t\tvec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );\n\n\t\treturn mat4( v1, v2, v3, v4 );\n\n\t}\n\n#endif\n',
    'skinning_vertex': '\n#ifdef USE_SKINNING\n\n\tvec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );\n\n\tvec4 skinned = vec4( 0.0 );\n\tskinned += boneMatX * skinVertex * skinWeight.x;\n\tskinned += boneMatY * skinVertex * skinWeight.y;\n\tskinned += boneMatZ * skinVertex * skinWeight.z;\n\tskinned += boneMatW * skinVertex * skinWeight.w;\n\n\ttransformed = ( bindMatrixInverse * skinned ).xyz;\n\n#endif\n',
    'skinnormal_vertex': '\n#ifdef USE_SKINNING\n\n\tmat4 skinMatrix = mat4( 0.0 );\n\tskinMatrix += skinWeight.x * boneMatX;\n\tskinMatrix += skinWeight.y * boneMatY;\n\tskinMatrix += skinWeight.z * boneMatZ;\n\tskinMatrix += skinWeight.w * boneMatW;\n\tskinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;\n\n\tobjectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;\n\n\t#ifdef USE_TANGENT\n\n\t\tobjectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;\n\n\t#endif\n\n#endif\n',
    'specularmap_fragment': '\nfloat specularStrength;\n\n#ifdef USE_SPECULARMAP\n\n\tvec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );\n\tspecularStrength = texelSpecular.r;\n\n#else\n\n\tspecularStrength = 1.0;\n\n#endif\n',
    'specularmap_pars_fragment': '\n#ifdef USE_SPECULARMAP\n\n\tuniform sampler2D specularMap;\n\n#endif\n',
    'tonemapping_fragment': '\n#if defined( TONE_MAPPING )\n\n\tgl_FragColor.rgb = toneMapping( gl_FragColor.rgb );\n\n#endif\n',
    'tonemapping_pars_fragment': "\n#ifndef saturate\n// <common> may have defined saturate() already\n#define saturate( a ) clamp( a, 0.0, 1.0 )\n#endif\n\nuniform float toneMappingExposure;\n\n// exposure only\nvec3 LinearToneMapping( vec3 color ) {\n\n\treturn saturate( toneMappingExposure * color );\n\n}\n\n// source: https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf\nvec3 ReinhardToneMapping( vec3 color ) {\n\n\tcolor *= toneMappingExposure;\n\treturn saturate( color / ( vec3( 1.0 ) + color ) );\n\n}\n\n// source: http://filmicworlds.com/blog/filmic-tonemapping-operators/\nvec3 CineonToneMapping( vec3 color ) {\n\n\t// filmic operator by Jim Hejl and Richard Burgess-Dawson\n\tcolor *= toneMappingExposure;\n\tcolor = max( vec3( 0.0 ), color - 0.004 );\n\treturn pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );\n\n}\n\n// source: https://github.com/selfshadow/ltc_code/blob/master/webgl/shaders/ltc/ltc_blit.fs\nvec3 RRTAndODTFit( vec3 v ) {\n\n\tvec3 a = v * ( v + 0.0245786 ) - 0.000090537;\n\tvec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;\n\treturn a / b;\n\n}\n\n// this implementation of ACES is modified to accommodate a brighter viewing environment.\n// the scale factor of 1/0.6 is subjective. see discussion in #19621.\n\nvec3 ACESFilmicToneMapping( vec3 color ) {\n\n\t// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT\n\tconst mat3 ACESInputMat = mat3(\n\t\tvec3( 0.59719, 0.07600, 0.02840 ), // transposed from source\n\t\tvec3( 0.35458, 0.90834, 0.13383 ),\n\t\tvec3( 0.04823, 0.01566, 0.83777 )\n\t);\n\n\t// ODT_SAT => XYZ => D60_2_D65 => sRGB\n\tconst mat3 ACESOutputMat = mat3(\n\t\tvec3(  1.60475, -0.10208, -0.00327 ), // transposed from source\n\t\tvec3( -0.53108,  1.10813, -0.07276 ),\n\t\tvec3( -0.07367, -0.00605,  1.07602 )\n\t);\n\n\tcolor *= toneMappingExposure / 0.6;\n\n\tcolor = ACESInputMat * color;\n\n\t// Apply RRT and ODT\n\tcolor = RRTAndODTFit( color );\n\n\tcolor = ACESOutputMat * color;\n\n\t// Clamp to [0, 1]\n\treturn saturate( color );\n\n}\n\n// Matrices for rec 2020 <> rec 709 color space conversion\n// matrix provided in row-major order so it has been transposed\n// https://www.itu.int/pub/R-REP-BT.2407-2017\nconst mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(\n\tvec3( 1.6605, - 0.1246, - 0.0182 ),\n\tvec3( - 0.5876, 1.1329, - 0.1006 ),\n\tvec3( - 0.0728, - 0.0083, 1.1187 )\n);\n\nconst mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(\n\tvec3( 0.6274, 0.0691, 0.0164 ),\n\tvec3( 0.3293, 0.9195, 0.0880 ),\n\tvec3( 0.0433, 0.0113, 0.8956 )\n);\n\n// https://iolite-engine.com/blog_posts/minimal_agx_implementation\n// Mean error^2: 3.6705141e-06\nvec3 agxDefaultContrastApprox( vec3 x ) {\n\n\tvec3 x2 = x * x;\n\tvec3 x4 = x2 * x2;\n\n\treturn + 15.5 * x4 * x2\n\t\t- 40.14 * x4 * x\n\t\t+ 31.96 * x4\n\t\t- 6.868 * x2 * x\n\t\t+ 0.4298 * x2\n\t\t+ 0.1191 * x\n\t\t- 0.00232;\n\n}\n\n// AgX Tone Mapping implementation based on Filament, which in turn is based\n// on Blender's implementation using rec 2020 primaries\n// https://github.com/google/filament/pull/7236\n// Inputs and outputs are encoded as Linear-sRGB.\n\nvec3 AgXToneMapping( vec3 color ) {\n\n\t// AgX constants\n\tconst mat3 AgXInsetMatrix = mat3(\n\t\tvec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),\n\t\tvec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),\n\t\tvec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )\n\t);\n\n\t// explicit AgXOutsetMatrix generated from Filaments AgXOutsetMatrixInv\n\tconst mat3 AgXOutsetMatrix = mat3(\n\t\tvec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),\n\t\tvec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),\n\t\tvec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )\n\t);\n\n\t// LOG2_MIN      = -10.0\n\t// LOG2_MAX      =  +6.5\n\t// MIDDLE_GRAY   =  0.18\n\tconst float AgxMinEv = - 12.47393;  // log2( pow( 2, LOG2_MIN ) * MIDDLE_GRAY )\n\tconst float AgxMaxEv = 4.026069;    // log2( pow( 2, LOG2_MAX ) * MIDDLE_GRAY )\n\n\tcolor *= toneMappingExposure;\n\n\tcolor = LINEAR_SRGB_TO_LINEAR_REC2020 * color;\n\n\tcolor = AgXInsetMatrix * color;\n\n\t// Log2 encoding\n\tcolor = max( color, 1e-10 ); // avoid 0 or negative numbers for log2\n\tcolor = log2( color );\n\tcolor = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );\n\n\tcolor = clamp( color, 0.0, 1.0 );\n\n\t// Apply sigmoid\n\tcolor = agxDefaultContrastApprox( color );\n\n\t// Apply AgX look\n\t// v = agxLook(v, look);\n\n\tcolor = AgXOutsetMatrix * color;\n\n\t// Linearize\n\tcolor = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );\n\n\tcolor = LINEAR_REC2020_TO_LINEAR_SRGB * color;\n\n\t// Gamut mapping. Simple clamp for now.\n\tcolor = clamp( color, 0.0, 1.0 );\n\n\treturn color;\n\n}\n\n// https://modelviewer.dev/examples/tone-mapping\n\nvec3 NeutralToneMapping( vec3 color ) {\n\n\tconst float StartCompression = 0.8 - 0.04;\n\tconst float Desaturation = 0.15;\n\n\tcolor *= toneMappingExposure;\n\n\tfloat x = min( color.r, min( color.g, color.b ) );\n\n\tfloat offset = x < 0.08 ? x - 6.25 * x * x : 0.04;\n\n\tcolor -= offset;\n\n\tfloat peak = max( color.r, max( color.g, color.b ) );\n\n\tif ( peak < StartCompression ) return color;\n\n\tfloat d = 1. - StartCompression;\n\n\tfloat newPeak = 1. - d * d / ( peak + d - StartCompression );\n\n\tcolor *= newPeak / peak;\n\n\tfloat g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );\n\n\treturn mix( color, vec3( newPeak ), g );\n\n}\n\nvec3 CustomToneMapping( vec3 color ) { return color; }\n",
    'transmission_fragment': '\n#ifdef USE_TRANSMISSION\n\n\tmaterial.transmission = transmission;\n\tmaterial.transmissionAlpha = 1.0;\n\tmaterial.thickness = thickness;\n\tmaterial.attenuationDistance = attenuationDistance;\n\tmaterial.attenuationColor = attenuationColor;\n\n\t#ifdef USE_TRANSMISSIONMAP\n\n\t\tmaterial.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;\n\n\t#endif\n\n\t#ifdef USE_THICKNESSMAP\n\n\t\tmaterial.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;\n\n\t#endif\n\n\tvec3 pos = vWorldPosition;\n\tvec3 v = normalize( cameraPosition - pos );\n\tvec3 n = transformNormalByInverseViewMatrix( normal, viewMatrix );\n\n\tvec4 transmitted = getIBLVolumeRefraction(\n\t\tn, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,\n\t\tpos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,\n\t\tmaterial.attenuationColor, material.attenuationDistance );\n\n\tmaterial.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );\n\n\ttotalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );\n\n#endif\n',
    'transmission_pars_fragment': "\n#ifdef USE_TRANSMISSION\n\n\t// Transmission code is based on glTF-Sampler-Viewer\n\t// https://github.com/KhronosGroup/glTF-Sample-Viewer\n\n\tuniform float transmission;\n\tuniform float thickness;\n\tuniform float attenuationDistance;\n\tuniform vec3 attenuationColor;\n\n\t#ifdef USE_TRANSMISSIONMAP\n\n\t\tuniform sampler2D transmissionMap;\n\n\t#endif\n\n\t#ifdef USE_THICKNESSMAP\n\n\t\tuniform sampler2D thicknessMap;\n\n\t#endif\n\n\tuniform vec2 transmissionSamplerSize;\n\tuniform sampler2D transmissionSamplerMap;\n\n\tuniform mat4 modelMatrix;\n\tuniform mat4 projectionMatrix;\n\n\tvarying vec3 vWorldPosition;\n\n\t// Mipped Bicubic Texture Filtering by N8\n\t// https://www.shadertoy.com/view/Dl2SDW\n\n\tfloat w0( float a ) {\n\n\t\treturn ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );\n\n\t}\n\n\tfloat w1( float a ) {\n\n\t\treturn ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );\n\n\t}\n\n\tfloat w2( float a ){\n\n\t\treturn ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );\n\n\t}\n\n\tfloat w3( float a ) {\n\n\t\treturn ( 1.0 / 6.0 ) * ( a * a * a );\n\n\t}\n\n\t// g0 and g1 are the two amplitude functions\n\tfloat g0( float a ) {\n\n\t\treturn w0( a ) + w1( a );\n\n\t}\n\n\tfloat g1( float a ) {\n\n\t\treturn w2( a ) + w3( a );\n\n\t}\n\n\t// h0 and h1 are the two offset functions\n\tfloat h0( float a ) {\n\n\t\treturn - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );\n\n\t}\n\n\tfloat h1( float a ) {\n\n\t\treturn 1.0 + w3( a ) / ( w2( a ) + w3( a ) );\n\n\t}\n\n\tvec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {\n\n\t\tuv = uv * texelSize.zw + 0.5;\n\n\t\tvec2 iuv = floor( uv );\n\t\tvec2 fuv = fract( uv );\n\n\t\tfloat g0x = g0( fuv.x );\n\t\tfloat g1x = g1( fuv.x );\n\t\tfloat h0x = h0( fuv.x );\n\t\tfloat h1x = h1( fuv.x );\n\t\tfloat h0y = h0( fuv.y );\n\t\tfloat h1y = h1( fuv.y );\n\n\t\tvec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;\n\t\tvec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;\n\t\tvec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;\n\t\tvec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;\n\n\t\treturn g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +\n\t\t\tg1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );\n\n\t}\n\n\tvec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {\n\n\t\tvec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );\n\t\tvec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );\n\t\tvec2 fLodSizeInv = 1.0 / fLodSize;\n\t\tvec2 cLodSizeInv = 1.0 / cLodSize;\n\t\tvec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );\n\t\tvec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );\n\t\treturn mix( fSample, cSample, fract( lod ) );\n\n\t}\n\n\tvec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {\n\n\t\t// Direction of refracted light.\n\t\tvec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );\n\n\t\t// Compute rotation-independent scaling of the model matrix.\n\t\tvec3 modelScale;\n\t\tmodelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );\n\t\tmodelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );\n\t\tmodelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );\n\n\t\t// The thickness is specified in local space.\n\t\treturn normalize( refractionVector ) * thickness * modelScale;\n\n\t}\n\n\tfloat applyIorToRoughness( const in float roughness, const in float ior ) {\n\n\t\t// Scale roughness with IOR so that an IOR of 1.0 results in no microfacet refraction and\n\t\t// an IOR of 1.5 results in the default amount of microfacet refraction.\n\t\treturn roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );\n\n\t}\n\n\tvec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {\n\n\t\tfloat lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );\n\t\treturn textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );\n\n\t}\n\n\tvec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {\n\n\t\tif ( isinf( attenuationDistance ) ) {\n\n\t\t\t// Attenuation distance is +∞, i.e. the transmitted color is not attenuated at all.\n\t\t\treturn vec3( 1.0 );\n\n\t\t} else {\n\n\t\t\t// Compute light attenuation using Beer's law.\n\t\t\tvec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;\n\t\t\tvec3 transmittance = exp( - attenuationCoefficient * transmissionDistance ); // Beer's law\n\t\t\treturn transmittance;\n\n\t\t}\n\n\t}\n\n\tvec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,\n\t\tconst in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,\n\t\tconst in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,\n\t\tconst in vec3 attenuationColor, const in float attenuationDistance ) {\n\n\t\tvec4 transmittedLight;\n\t\tvec3 transmittance;\n\n\t\t#ifdef USE_DISPERSION\n\n\t\t\tfloat halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;\n\t\t\tvec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );\n\n\t\t\tfor ( int i = 0; i < 3; i ++ ) {\n\n\t\t\t\tvec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );\n\t\t\t\tvec3 refractedRayExit = position + transmissionRay;\n\n\t\t\t\t// Project refracted vector on the framebuffer, while mapping to normalized device coordinates.\n\t\t\t\tvec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );\n\t\t\t\tvec2 refractionCoords = ndcPos.xy / ndcPos.w;\n\t\t\t\trefractionCoords += 1.0;\n\t\t\t\trefractionCoords /= 2.0;\n\n\t\t\t\t// Sample framebuffer to get pixel the refracted ray hits.\n\t\t\t\tvec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );\n\t\t\t\ttransmittedLight[ i ] = transmissionSample[ i ];\n\t\t\t\ttransmittedLight.a += transmissionSample.a;\n\n\t\t\t\ttransmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];\n\n\t\t\t}\n\n\t\t\ttransmittedLight.a /= 3.0;\n\n\t\t#else\n\n\t\t\tvec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );\n\t\t\tvec3 refractedRayExit = position + transmissionRay;\n\n\t\t\t// Project refracted vector on the framebuffer, while mapping to normalized device coordinates.\n\t\t\tvec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );\n\t\t\tvec2 refractionCoords = ndcPos.xy / ndcPos.w;\n\t\t\trefractionCoords += 1.0;\n\t\t\trefractionCoords /= 2.0;\n\n\t\t\t// Sample framebuffer to get pixel the refracted ray hits.\n\t\t\ttransmittedLight = getTransmissionSample( refractionCoords, roughness, ior );\n\t\t\ttransmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );\n\n\t\t#endif\n\n\t\tvec3 attenuatedColor = transmittance * transmittedLight.rgb;\n\n\t\t// Get the specular component.\n\t\tvec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );\n\n\t\t// As less light is transmitted, the opacity should be increased. This simple approximation does a decent job\n\t\t// of modulating a CSS background, and has no effect when the buffer is opaque, due to a solid object or clear color.\n\t\tfloat transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;\n\n\t\treturn vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );\n\n\t}\n#endif\n",
    'uv_pars_fragment': '\n#if defined( USE_UV ) || defined( USE_ANISOTROPY )\n\n\tvarying vec2 vUv;\n\n#endif\n#ifdef USE_MAP\n\n\tvarying vec2 vMapUv;\n\n#endif\n#ifdef USE_ALPHAMAP\n\n\tvarying vec2 vAlphaMapUv;\n\n#endif\n#ifdef USE_LIGHTMAP\n\n\tvarying vec2 vLightMapUv;\n\n#endif\n#ifdef USE_AOMAP\n\n\tvarying vec2 vAoMapUv;\n\n#endif\n#ifdef USE_BUMPMAP\n\n\tvarying vec2 vBumpMapUv;\n\n#endif\n#ifdef USE_NORMALMAP\n\n\tvarying vec2 vNormalMapUv;\n\n#endif\n#ifdef USE_EMISSIVEMAP\n\n\tvarying vec2 vEmissiveMapUv;\n\n#endif\n#ifdef USE_METALNESSMAP\n\n\tvarying vec2 vMetalnessMapUv;\n\n#endif\n#ifdef USE_ROUGHNESSMAP\n\n\tvarying vec2 vRoughnessMapUv;\n\n#endif\n#ifdef USE_DIFFUSE_ROUGHNESSMAP\n\n\tvarying vec2 vDiffuseRoughnessMapUv;\n\n#endif\n#ifdef USE_ANISOTROPYMAP\n\n\tvarying vec2 vAnisotropyMapUv;\n\n#endif\n#ifdef USE_CLEARCOATMAP\n\n\tvarying vec2 vClearcoatMapUv;\n\n#endif\n#ifdef USE_CLEARCOAT_NORMALMAP\n\n\tvarying vec2 vClearcoatNormalMapUv;\n\n#endif\n#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n\n\tvarying vec2 vClearcoatRoughnessMapUv;\n\n#endif\n#ifdef USE_IRIDESCENCEMAP\n\n\tvarying vec2 vIridescenceMapUv;\n\n#endif\n#ifdef USE_IRIDESCENCE_THICKNESSMAP\n\n\tvarying vec2 vIridescenceThicknessMapUv;\n\n#endif\n#ifdef USE_SHEEN_COLORMAP\n\n\tvarying vec2 vSheenColorMapUv;\n\n#endif\n#ifdef USE_SHEEN_ROUGHNESSMAP\n\n\tvarying vec2 vSheenRoughnessMapUv;\n\n#endif\n#ifdef USE_SPECULARMAP\n\n\tvarying vec2 vSpecularMapUv;\n\n#endif\n#ifdef USE_SPECULAR_COLORMAP\n\n\tvarying vec2 vSpecularColorMapUv;\n\n#endif\n#ifdef USE_SPECULAR_INTENSITYMAP\n\n\tvarying vec2 vSpecularIntensityMapUv;\n\n#endif\n#ifdef USE_TRANSMISSIONMAP\n\n\tuniform mat3 transmissionMapTransform;\n\tvarying vec2 vTransmissionMapUv;\n\n#endif\n#ifdef USE_THICKNESSMAP\n\n\tuniform mat3 thicknessMapTransform;\n\tvarying vec2 vThicknessMapUv;\n\n#endif\n',
    'uv_pars_vertex': '\n#if defined( USE_UV ) || defined( USE_ANISOTROPY )\n\n\tvarying vec2 vUv;\n\n#endif\n#ifdef USE_MAP\n\n\tuniform mat3 mapTransform;\n\tvarying vec2 vMapUv;\n\n#endif\n#ifdef USE_ALPHAMAP\n\n\tuniform mat3 alphaMapTransform;\n\tvarying vec2 vAlphaMapUv;\n\n#endif\n#ifdef USE_LIGHTMAP\n\n\tuniform mat3 lightMapTransform;\n\tvarying vec2 vLightMapUv;\n\n#endif\n#ifdef USE_AOMAP\n\n\tuniform mat3 aoMapTransform;\n\tvarying vec2 vAoMapUv;\n\n#endif\n#ifdef USE_BUMPMAP\n\n\tuniform mat3 bumpMapTransform;\n\tvarying vec2 vBumpMapUv;\n\n#endif\n#ifdef USE_NORMALMAP\n\n\tuniform mat3 normalMapTransform;\n\tvarying vec2 vNormalMapUv;\n\n#endif\n#ifdef USE_DISPLACEMENTMAP\n\n\tuniform mat3 displacementMapTransform;\n\tvarying vec2 vDisplacementMapUv;\n\n#endif\n#ifdef USE_EMISSIVEMAP\n\n\tuniform mat3 emissiveMapTransform;\n\tvarying vec2 vEmissiveMapUv;\n\n#endif\n#ifdef USE_METALNESSMAP\n\n\tuniform mat3 metalnessMapTransform;\n\tvarying vec2 vMetalnessMapUv;\n\n#endif\n#ifdef USE_ROUGHNESSMAP\n\n\tuniform mat3 roughnessMapTransform;\n\tvarying vec2 vRoughnessMapUv;\n\n#endif\n#ifdef USE_DIFFUSE_ROUGHNESSMAP\n\n\tuniform mat3 diffuseRoughnessMapTransform;\n\tvarying vec2 vDiffuseRoughnessMapUv;\n\n#endif\n#ifdef USE_ANISOTROPYMAP\n\n\tuniform mat3 anisotropyMapTransform;\n\tvarying vec2 vAnisotropyMapUv;\n\n#endif\n#ifdef USE_CLEARCOATMAP\n\n\tuniform mat3 clearcoatMapTransform;\n\tvarying vec2 vClearcoatMapUv;\n\n#endif\n#ifdef USE_CLEARCOAT_NORMALMAP\n\n\tuniform mat3 clearcoatNormalMapTransform;\n\tvarying vec2 vClearcoatNormalMapUv;\n\n#endif\n#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n\n\tuniform mat3 clearcoatRoughnessMapTransform;\n\tvarying vec2 vClearcoatRoughnessMapUv;\n\n#endif\n#ifdef USE_SHEEN_COLORMAP\n\n\tuniform mat3 sheenColorMapTransform;\n\tvarying vec2 vSheenColorMapUv;\n\n#endif\n#ifdef USE_SHEEN_ROUGHNESSMAP\n\n\tuniform mat3 sheenRoughnessMapTransform;\n\tvarying vec2 vSheenRoughnessMapUv;\n\n#endif\n#ifdef USE_IRIDESCENCEMAP\n\n\tuniform mat3 iridescenceMapTransform;\n\tvarying vec2 vIridescenceMapUv;\n\n#endif\n#ifdef USE_IRIDESCENCE_THICKNESSMAP\n\n\tuniform mat3 iridescenceThicknessMapTransform;\n\tvarying vec2 vIridescenceThicknessMapUv;\n\n#endif\n#ifdef USE_SPECULARMAP\n\n\tuniform mat3 specularMapTransform;\n\tvarying vec2 vSpecularMapUv;\n\n#endif\n#ifdef USE_SPECULAR_COLORMAP\n\n\tuniform mat3 specularColorMapTransform;\n\tvarying vec2 vSpecularColorMapUv;\n\n#endif\n#ifdef USE_SPECULAR_INTENSITYMAP\n\n\tuniform mat3 specularIntensityMapTransform;\n\tvarying vec2 vSpecularIntensityMapUv;\n\n#endif\n#ifdef USE_TRANSMISSIONMAP\n\n\tuniform mat3 transmissionMapTransform;\n\tvarying vec2 vTransmissionMapUv;\n\n#endif\n#ifdef USE_THICKNESSMAP\n\n\tuniform mat3 thicknessMapTransform;\n\tvarying vec2 vThicknessMapUv;\n\n#endif\n',
    'uv_vertex': '\n#if defined( USE_UV ) || defined( USE_ANISOTROPY )\n\n\tvUv = vec3( uv, 1 ).xy;\n\n#endif\n#ifdef USE_MAP\n\n\tvMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_ALPHAMAP\n\n\tvAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_LIGHTMAP\n\n\tvLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_AOMAP\n\n\tvAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_BUMPMAP\n\n\tvBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_NORMALMAP\n\n\tvNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_DISPLACEMENTMAP\n\n\tvDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_EMISSIVEMAP\n\n\tvEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_METALNESSMAP\n\n\tvMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_ROUGHNESSMAP\n\n\tvRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_DIFFUSE_ROUGHNESSMAP\n\n\tvDiffuseRoughnessMapUv = ( diffuseRoughnessMapTransform * vec3( DIFFUSE_ROUGHNESSMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_ANISOTROPYMAP\n\n\tvAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_CLEARCOATMAP\n\n\tvClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_CLEARCOAT_NORMALMAP\n\n\tvClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_CLEARCOAT_ROUGHNESSMAP\n\n\tvClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_IRIDESCENCEMAP\n\n\tvIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_IRIDESCENCE_THICKNESSMAP\n\n\tvIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_SHEEN_COLORMAP\n\n\tvSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_SHEEN_ROUGHNESSMAP\n\n\tvSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_SPECULARMAP\n\n\tvSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_SPECULAR_COLORMAP\n\n\tvSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_SPECULAR_INTENSITYMAP\n\n\tvSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_TRANSMISSIONMAP\n\n\tvTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;\n\n#endif\n#ifdef USE_THICKNESSMAP\n\n\tvThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;\n\n#endif\n',
    'worldpos_vertex': '\n#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0\n\n\tvec4 worldPosition = vec4( transformed, 1.0 );\n\n\t#ifdef USE_BATCHING\n\n\t\tworldPosition = batchingMatrix * worldPosition;\n\n\t#endif\n\n\t#ifdef USE_INSTANCING\n\n\t\tworldPosition = instanceMatrix * worldPosition;\n\n\t#endif\n\n\tworldPosition = modelMatrix * worldPosition;\n\n#endif\n',
}


def load_chunks(dirpath=None):
    """{chunk name: GLSL source} for #include resolution.

    The base is the embedded ShaderChunk texts (_EMBEDDED_CHUNKS), so
    the full set is available in standalone deployments where the
    three.js clone is not on disk (e.g. Code & Magic).  When a chunk
    directory is given, or the default clone location exists, the
    on-disk ShaderChunk/*.glsl.js extractions are overlaid on top
    (on-disk texts win), so local development keeps tracking the
    clone exactly."""
    chunks = dict(_EMBEDDED_CHUNKS)
    d = Path(dirpath) if dirpath is not None else _default_chunk_dir()
    if d.is_dir():
        for f in sorted(d.glob("*.glsl.js")):
            text = f.read_text(encoding="utf-8")
            body, _ = _extract_template(text)
            chunks[f.name[:-len(".glsl.js")]] = body
    return chunks


def load_shader_lib(dirpath=None):
    """{lib name: {export name: GLSL source}} from ShaderLib files."""
    d = Path(dirpath) if dirpath is not None else _default_lib_dir()
    lib = {}
    if not d.is_dir():
        return lib
    for f in sorted(d.glob("*.glsl.js")):
        text = f.read_text(encoding="utf-8")
        entry = {}
        for m in re.finditer(r"export\s+const\s+(\w+)\s*=", text):
            body, _ = _extract_template(text, m.end())
            entry[m.group(1)] = body
        if entry:
            lib[f.name[:-len(".glsl.js")]] = entry
    return lib


# ---------------------------------------------------------------------------
# Tokenizer (shared by the preprocessor's macro expansion and the parser)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<num>0[xX][0-9a-fA-F]+[uU]?|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuU]?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op><<=|>>=|\+\+|--|\+=|-=|\*=|/=|%=|&=|\^=|\|=|<<|>>|<=|>=|==|!=|&&|\|\||
          [+\-*/%<>=!~&^|?:;,.(){}\[\]])
""", re.X | re.S)


class Tok:
    __slots__ = ("kind", "text", "line")

    def __init__(self, kind, text, line=0):
        self.kind, self.text, self.line = kind, text, line

    def __repr__(self):
        return f"<{self.kind}:{self.text}>"


def _scan_tokens(src):
    """Tokenize GLSL text (comments/whitespace dropped, no EOF mark)."""
    toks = []
    line = 1
    pos = 0
    for m in _TOKEN_RE.finditer(src):
        if m.start() > pos and src[pos:m.start()].strip():
            raise GLSLError(
                f"lex error near {src[pos:pos + 20]!r} (line {line})")
        line += src[pos:m.end()].count("\n")
        pos = m.end()
        if m.lastgroup in ("ws", "comment"):
            continue
        toks.append(Tok(m.lastgroup, m.group(), line))
    if src[pos:].strip():
        raise GLSLError(f"lex error near {src[pos:pos + 20]!r} (line {line})")
    return toks


def lex(src):
    """Tokenize preprocessed GLSL source; returns tokens + EOF token.
    Lines whose first non-space character is '#' are dropped first, so
    parse() tolerates un-preprocessed text (directives then do
    nothing -- see the module docstring)."""
    src = re.sub(r"(?m)^[ \t]*#[^\n]*", "", src)
    toks = _scan_tokens(src)
    toks.append(Tok("eof", "", toks[-1].line if toks else 1))
    return toks


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class _Macro:
    __slots__ = ("params", "body")

    def __init__(self, params, body):
        self.params = params  # None: object-like; else [param names]
        self.body = body


def _strip_comments(src):
    """Replace // and /* */ comments with spaces (newlines kept).
    GLSL ES 1.00 has no string literals, so this is safe."""
    out = []
    i, n = 0, len(src)
    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i)
            if j < 0:
                break
            out.append(" ")
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            if j < 0:
                raise GLSLError("unterminated /* comment")
            out.append("".join("\n" if c == "\n" else " "
                               for c in src[i:j + 2]))
            i = j + 2
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def _collect_args(toks, open_idx):
    """toks[open_idx] is '('.  Return (args, index_after_close) where
    args is a list of token lists split at top-level commas."""
    args = []
    cur = []
    depth = 0
    i = open_idx
    while i < len(toks):
        t = toks[i]
        if t.text == "(":
            depth += 1
            if depth > 1:
                cur.append(t)
        elif t.text == ")":
            depth -= 1
            if depth == 0:
                if cur or args:
                    args.append(cur)
                return args, i + 1
            cur.append(t)
        elif t.text == "," and depth == 1:
            args.append(cur)
            cur = []
        else:
            cur.append(t)
        i += 1
    raise GLSLError("unterminated macro argument list")


def _expand_tokens(toks, macros, disabled=frozenset()):
    """Expand object- and function-like macros in a token list.
    ``disabled`` is the set of macro names currently being expanded
    (prevents self-recursion, as in cpp)."""
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "ident" and t.text in macros and t.text not in disabled:
            mac = macros[t.text]
            if mac.params is None:
                body = _scan_tokens(mac.body) if mac.body.strip() else []
                out.extend(_expand_tokens(body, macros, disabled | {t.text}))
                i += 1
                continue
            if i + 1 < len(toks) and toks[i + 1].text == "(":
                args, nxt = _collect_args(toks, i + 1)
                if len(args) == len(mac.params):
                    amap = {}
                    for pname, atoks in zip(mac.params, args):
                        amap[pname] = _expand_tokens(atoks, macros, disabled)
                    body = []
                    for bt in _scan_tokens(mac.body):
                        if bt.kind == "ident" and bt.text in amap:
                            body.extend(amap[bt.text])
                        else:
                            body.append(bt)
                    out.extend(_expand_tokens(body, macros,
                                              disabled | {t.text}))
                    i = nxt
                    continue
            out.append(t)
            i += 1
            continue
        out.append(t)
        i += 1
    return out


# -- #if expression evaluation (integer recursive descent, no eval) ---------

class _PPEval:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i].text if self.i < len(self.toks) else ""

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        v = self.p_or()
        if self.i != len(self.toks):
            raise GLSLError("trailing tokens in #if expression")
        return v

    def _bin(self, sub, ops):
        v = sub()
        while self.peek() in ops:
            op = self.next()
            r = sub()
            if op == "||":
                v = 1 if (v or r) else 0
            elif op == "&&":
                v = 1 if (v and r) else 0
            elif op == "|":
                v = v | r
            elif op == "^":
                v = v ^ r
            elif op == "&":
                v = v & r
            elif op == "==":
                v = 1 if v == r else 0
            elif op == "!=":
                v = 1 if v != r else 0
            elif op == "<":
                v = 1 if v < r else 0
            elif op == "<=":
                v = 1 if v <= r else 0
            elif op == ">":
                v = 1 if v > r else 0
            elif op == ">=":
                v = 1 if v >= r else 0
            elif op == "<<":
                v = v << r
            elif op == ">>":
                v = v >> r
            elif op == "+":
                v = v + r
            elif op == "-":
                v = v - r
            elif op == "*":
                v = v * r
            elif op == "/":
                v = int(v / r) if r else 0
            elif op == "%":
                v = v % r if r else 0
        return v

    def p_or(self):
        return self._bin(self.p_and, ("||",))

    def p_and(self):
        return self._bin(self.p_bitor, ("&&",))

    def p_bitor(self):
        return self._bin(self.p_bitxor, ("|",))

    def p_bitxor(self):
        return self._bin(self.p_bitand, ("^",))

    def p_bitand(self):
        return self._bin(self.p_eq, ("&",))

    def p_eq(self):
        return self._bin(self.p_rel, ("==", "!="))

    def p_rel(self):
        return self._bin(self.p_shift, ("<", "<=", ">", ">="))

    def p_shift(self):
        return self._bin(self.p_add, ("<<", ">>"))

    def p_add(self):
        return self._bin(self.p_mul, ("+", "-"))

    def p_mul(self):
        return self._bin(self.p_unary, ("*", "/", "%"))

    def p_unary(self):
        t = self.peek()
        if t == "!":
            self.next()
            return 0 if self.p_unary() else 1
        if t == "~":
            self.next()
            return ~self.p_unary()
        if t == "-":
            self.next()
            return -self.p_unary()
        if t == "+":
            self.next()
            return self.p_unary()
        return self.p_primary()

    def p_primary(self):
        t = self.next()
        if t == "(":
            v = self.p_or()
            if self.next() != ")":
                raise GLSLError("unbalanced '(' in #if expression")
            return v
        if t == "":
            raise GLSLError("empty #if expression")
        try:
            return int(t, 0) if not t.startswith("0") or len(t) == 1 \
                else int(t, 8)
        except ValueError:
            pass
        try:
            return int(float(t))
        except ValueError:
            raise GLSLError(f"bad token {t!r} in #if expression")


def _eval_pp_expr(expr, macros):
    toks = _scan_tokens(expr)
    # Resolve defined(X) / defined X BEFORE macro expansion.
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "ident" and t.text == "defined":
            j = i + 1
            if j < len(toks) and toks[j].text == "(":
                name = toks[j + 1].text if j + 1 < len(toks) else ""
                out.append(Tok("num", "1" if name in macros else "0"))
                i = j + 3
            elif j < len(toks):
                out.append(Tok("num", "1" if toks[j].text in macros else "0"))
                i = j + 1
            else:
                raise GLSLError("defined without an argument in #if")
        else:
            out.append(t)
            i += 1
    toks = _expand_tokens(out, macros)
    norm = []
    for t in toks:
        if t.kind == "ident":
            norm.append(Tok("num", "1" if t.text == "true" else "0"))
        else:
            norm.append(t)
    if not norm:
        return 0
    return _PPEval(norm).parse()


_INCLUDE_RE = re.compile(r"""<\s*([^>]+?)\s*>|"\s*([^"]+?)\s*\"""")


def _parse_define(body_rest):
    """Parse the text after '#define': return (name, _Macro)."""
    s = body_rest
    i = 0
    while i < len(s) and s[i] in " \t":
        i += 1
    j = i
    while j < len(s) and (s[j].isalnum() or s[j] == "_"):
        j += 1
    name = s[i:j]
    if not name:
        raise GLSLError(f"malformed #define: {body_rest!r}")
    if j < len(s) and s[j] == "(":  # '(' abutting the name: function-like
        k = s.find(")", j)
        if k < 0:
            raise GLSLError(f"malformed #define: {body_rest!r}")
        plist = s[j + 1:k].strip()
        params = [p.strip() for p in plist.split(",")] if plist else []
        return name, _Macro(params, s[k + 1:].strip())
    return name, _Macro(None, s[j:].strip())


def _pp_lines(lines, macros, chunks, out, depth):
    """Process source lines, appending expanded code lines to ``out``.
    Conditional frames: [parent_active, active, taken, seen_else]."""
    stack = []

    def active():
        return stack[-1][1] if stack else True

    for raw in lines:
        stripped = raw.lstrip()
        if not stripped.startswith("#"):
            if active():
                toks = _expand_tokens(_scan_tokens(raw), macros)
                out.append(" ".join(t.text for t in toks))
            else:
                out.append("")
            continue
        body = stripped[1:].lstrip()
        m = re.match(r"([A-Za-z]+)", body)
        word = m.group(1) if m else ""
        rest = body[m.end():].strip() if m else ""
        if word in ("if", "ifdef", "ifndef"):
            parent = active()
            if not parent:
                val = False
            elif word == "if":
                val = bool(_eval_pp_expr(rest, macros))
            else:
                nm = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", rest)
                defined = bool(nm) and nm.group(1) in macros
                val = defined if word == "ifdef" else not defined
            stack.append([parent, parent and val, val, False])
        elif word == "elif":
            if not stack:
                raise GLSLError("#elif without #if")
            fr = stack[-1]
            if fr[3]:
                raise GLSLError("#elif after #else")
            if fr[2]:
                fr[1] = False
            else:
                val = bool(_eval_pp_expr(rest, macros)) if fr[0] else False
                fr[1] = fr[0] and val
                fr[2] = val
        elif word == "else":
            if not stack:
                raise GLSLError("#else without #if")
            fr = stack[-1]
            if fr[3]:
                raise GLSLError("duplicate #else")
            fr[1] = fr[0] and not fr[2]
            fr[2] = True
            fr[3] = True
        elif word == "endif":
            if not stack:
                raise GLSLError("#endif without #if")
            stack.pop()
        elif not active():
            continue  # other directives are inert in dead regions
        elif word == "define":
            name, mac = _parse_define(body[m.end():])
            macros[name] = mac
        elif word == "undef":
            nm = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", rest)
            if nm:
                macros.pop(nm.group(1), None)
        elif word == "include":
            im = _INCLUDE_RE.search(rest)
            if not im:
                raise GLSLError(f"malformed #include: {rest!r}")
            inc = im.group(1) or im.group(2)
            if inc not in chunks:
                raise GLSLError(f"cannot resolve #include <{inc}>")
            if depth >= 64:
                raise GLSLError("#include nested too deeply")
            _pp_lines(_strip_comments(chunks[inc]).split("\n"), macros,
                      chunks, out, depth + 1)
        elif word == "error":
            raise GLSLError(f"#error {rest}")
        elif word in ("version", "extension", "pragma", "line", ""):
            continue  # accepted and dropped (see module docstring)
        else:
            continue  # unknown directive: tolerate
    if stack and depth == 0:
        raise GLSLError("unterminated #if block")


def preprocess(source, defines=None, chunks=None):
    """Run the preprocessor; returns the expanded source text."""
    src = source.replace("\\\r\n", "").replace("\\\n", "")
    src = _strip_comments(src)
    macros = {"GL_ES": _Macro(None, "1"),
              "__VERSION__": _Macro(None, "100")}
    if defines:
        for name, value in defines.items():
            if value is False:
                continue
            if value is True:
                macros[name] = _Macro(None, "true")
            else:
                macros[name] = _Macro(None, str(value))
    out = []
    _pp_lines(src.split("\n"), macros, chunks if chunks is not None else {},
              out, 0)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# WebGLProgram prefix reproduction (ShaderMaterial, ES 1.00 form)
# ---------------------------------------------------------------------------

def _generate_precision(precision):
    lines = [f"precision {precision} {t};" for t in (
        "float", "int", "sampler2D", "samplerCube", "sampler3D",
        "sampler2DArray", "sampler2DShadow", "samplerCubeShadow",
        "sampler2DArrayShadow", "isampler2D", "isampler3D",
        "isamplerCube", "isampler2DArray", "usampler2D", "usampler3D",
        "usamplerCube", "usampler2DArray")]
    macro = {"highp": "HIGH_PRECISION", "mediump": "MEDIUM_PRECISION",
             "lowp": "LOW_PRECISION"}.get(precision)
    if macro:
        lines.append(f"#define {macro}")
    return "\n".join(lines)


def _generate_defines(defines):
    out = []
    for name, value in (defines or {}).items():
        if value is False:
            continue
        if value is True:
            value = "true"
        out.append(f"#define {name} {value}")
    return "\n".join(out)


def _d(lines, cond, text):
    if cond:
        lines.append(text)


_VERTEX_DEFAULTS = dict(
    precision="highp", shader_type="ShaderMaterial", shader_name="",
    extension_clip_cull_distance=False,
    batching=False, batching_color=False,
    instancing=False, instancing_color=False, instancing_morph=False,
    fog=False, fog_exp2=False,
    map=False, envmap=False, envmap_mode="ENVMAP_MODE_REFLECTION",
    lightmap=False, aomap=False, bumpmap=False, normalmap=False,
    normalmap_objectspace=False, normalmap_tangentspace=False,
    displacementmap=False, emissivemap=False,
    anisotropy=False, anisotropy_map=False,
    clearcoat_map=False, clearcoat_roughness_map=False,
    clearcoat_normal_map=False, diffuse_roughness_map=False,
    iridescence_map=False, iridescence_thickness_map=False,
    specular_map=False, specular_color_map=False,
    specular_intensity_map=False,
    roughness_map=False, metalness_map=False, alphamap=False,
    alphahash=False, transmission=False, transmission_map=False,
    thickness_map=False, sheen_color_map=False, sheen_roughness_map=False,
    map_uv=None, alphamap_uv=None, lightmap_uv=None, aomap_uv=None,
    emissivemap_uv=None, bumpmap_uv=None, normalmap_uv=None,
    displacementmap_uv=None, metalnessmap_uv=None, roughnessmap_uv=None,
    anisotropy_map_uv=None, clearcoat_map_uv=None,
    clearcoat_normal_map_uv=None, clearcoat_roughness_map_uv=None,
    diffuse_roughness_map_uv=None, iridescence_map_uv=None,
    iridescence_thickness_map_uv=None, sheen_colormap_uv=None,
    sheen_roughness_map_uv=None, specular_map_uv=None,
    specular_colormap_uv=None, specular_intensity_map_uv=None,
    transmissionmap_uv=None, thicknessmap_uv=None,
    vertex_tangents=False, vertex_normals=True, vertex_colors=False,
    vertex_alphas=False, vertex_uv1s=False, vertex_uv2s=False,
    vertex_uv3s=False, points_uvs=False, flat_shading=False,
    skinning=False, morph_targets=False, morph_normals=False,
    morph_colors=False, morph_targets_count=0, morph_texture_stride=0,
    double_sided=False, flip_sided=False,
    shadowmap_enabled=False, shadowmap_type="SHADOWMAP_TYPE_BASIC",
    size_attenuation=False, num_light_probes=0,
    logarithmic_depth_buffer=False, reversed_depth_buffer=False,
)

_FRAGMENT_DEFAULTS = dict(
    precision="highp", shader_type="ShaderMaterial", shader_name="",
    fog=False, fog_exp2=False, alpha_to_coverage=False,
    map=False, matcap=False, envmap=False,
    envmap_type="ENVMAP_TYPE_CUBE", envmap_mode="ENVMAP_MODE_REFLECTION",
    envmap_blending="ENVMAP_BLENDING_NONE", envmap_pmrem=False,
    envmap_max_lod=0, envmap_size=0,
    lightmap=False, aomap=False, bumpmap=False, normalmap=False,
    normalmap_objectspace=False, normalmap_tangentspace=False,
    packed_normalmap=False, emissivemap=False,
    anisotropy=False, anisotropy_map=False,
    clearcoat=False, clearcoat_map=False, clearcoat_roughness_map=False,
    clearcoat_normal_map=False, diffuse_roughness=False,
    diffuse_roughness_map=False, dispersion=False, retroreflection=False,
    iridescence=False, iridescence_map=False,
    iridescence_thickness_map=False,
    specular_map=False, specular_color_map=False,
    specular_intensity_map=False,
    roughness_map=False, metalness_map=False,
    alphamap=False, alphatest=False, alphahash=False,
    sheen=False, sheen_color_map=False, sheen_roughness_map=False,
    transmission=False, transmission_map=False, thickness_map=False,
    vertex_tangents=False, vertex_colors=False, vertex_alphas=False,
    instancing_color=False, batching_color=False,
    vertex_uv1s=False, vertex_uv2s=False, vertex_uv3s=False,
    points_uvs=False, gradientmap=False, flat_shading=False,
    double_sided=False, flip_sided=False,
    shadowmap_enabled=False, shadowmap_type="SHADOWMAP_TYPE_BASIC",
    premultiplied_alpha=False, num_light_probes=0,
    decode_video_texture=False, decode_video_texture_emissive=False,
    logarithmic_depth_buffer=False, reversed_depth_buffer=False,
    tone_mapping=None, dithering=False, opaque=False,
    output_color_space="srgb", use_depth_packing=None,
)

_UV_CHANNELS = (
    ("map_uv", "MAP_UV"), ("alphamap_uv", "ALPHAMAP_UV"),
    ("lightmap_uv", "LIGHTMAP_UV"), ("aomap_uv", "AOMAP_UV"),
    ("emissivemap_uv", "EMISSIVEMAP_UV"), ("bumpmap_uv", "BUMPMAP_UV"),
    ("normalmap_uv", "NORMALMAP_UV"),
    ("displacementmap_uv", "DISPLACEMENTMAP_UV"),
    ("metalnessmap_uv", "METALNESSMAP_UV"),
    ("roughnessmap_uv", "ROUGHNESSMAP_UV"),
    ("anisotropy_map_uv", "ANISOTROPYMAP_UV"),
    ("clearcoat_map_uv", "CLEARCOATMAP_UV"),
    ("clearcoat_normal_map_uv", "CLEARCOAT_NORMALMAP_UV"),
    ("clearcoat_roughness_map_uv", "CLEARCOAT_ROUGHNESSMAP_UV"),
    ("diffuse_roughness_map_uv", "DIFFUSE_ROUGHNESSMAP_UV"),
    ("iridescence_map_uv", "IRIDESCENCEMAP_UV"),
    ("iridescence_thickness_map_uv", "IRIDESCENCE_THICKNESSMAP_UV"),
    ("sheen_colormap_uv", "SHEEN_COLORMAP_UV"),
    ("sheen_roughness_map_uv", "SHEEN_ROUGHNESSMAP_UV"),
    ("specular_map_uv", "SPECULARMAP_UV"),
    ("specular_colormap_uv", "SPECULAR_COLORMAP_UV"),
    ("specular_intensity_map_uv", "SPECULAR_INTENSITYMAP_UV"),
    ("transmissionmap_uv", "TRANSMISSIONMAP_UV"),
    ("thicknessmap_uv", "THICKNESSMAP_UV"),
)

_TONE_MAPPING_NAMES = {
    "linear": "Linear", "reinhard": "Reinhard", "cineon": "Cineon",
    "aces": "ACESFilmic", "acesfilmic": "ACESFilmic", "agx": "AgX",
    "neutral": "Neutral", "custom": "Custom",
}


def _chunk_text(chunks, name):
    if chunks and name in chunks:
        return chunks[name]
    return f"#include <{name}>"


def vertex_prefix(defines=None, parameters=None):
    """The prefix WebGLProgram prepends to a ShaderMaterial's vertex
    shader (ES 1.00 form).  See the module docstring for parameters."""
    p = dict(_VERTEX_DEFAULTS)
    if parameters:
        p.update(parameters)
    lines = [_generate_precision(p["precision"]),
             f"#define SHADER_TYPE {p['shader_type']}",
             f"#define SHADER_NAME {p['shader_name']}",
             _generate_defines(defines)]
    _d(lines, p["extension_clip_cull_distance"], "#define USE_CLIP_DISTANCE")
    _d(lines, p["batching"], "#define USE_BATCHING")
    _d(lines, p["batching_color"], "#define USE_BATCHING_COLOR")
    _d(lines, p["instancing"], "#define USE_INSTANCING")
    _d(lines, p["instancing_color"], "#define USE_INSTANCING_COLOR")
    _d(lines, p["instancing_morph"], "#define USE_INSTANCING_MORPH")
    _d(lines, p["fog"], "#define USE_FOG")
    _d(lines, p["fog"] and p["fog_exp2"], "#define FOG_EXP2")
    _d(lines, p["map"], "#define USE_MAP")
    if p["envmap"]:
        lines.append("#define USE_ENVMAP")
        lines.append(f"#define {p['envmap_mode']}")
    _d(lines, p["lightmap"], "#define USE_LIGHTMAP")
    _d(lines, p["aomap"], "#define USE_AOMAP")
    _d(lines, p["bumpmap"], "#define USE_BUMPMAP")
    _d(lines, p["normalmap"], "#define USE_NORMALMAP")
    _d(lines, p["normalmap_objectspace"], "#define USE_NORMALMAP_OBJECTSPACE")
    _d(lines, p["normalmap_tangentspace"],
       "#define USE_NORMALMAP_TANGENTSPACE")
    _d(lines, p["displacementmap"], "#define USE_DISPLACEMENTMAP")
    _d(lines, p["emissivemap"], "#define USE_EMISSIVEMAP")
    _d(lines, p["anisotropy"], "#define USE_ANISOTROPY")
    _d(lines, p["anisotropy_map"], "#define USE_ANISOTROPYMAP")
    _d(lines, p["clearcoat_map"], "#define USE_CLEARCOATMAP")
    _d(lines, p["clearcoat_roughness_map"], "#define USE_CLEARCOAT_ROUGHNESSMAP")
    _d(lines, p["clearcoat_normal_map"], "#define USE_CLEARCOAT_NORMALMAP")
    _d(lines, p["diffuse_roughness_map"], "#define USE_DIFFUSE_ROUGHNESSMAP")
    _d(lines, p["iridescence_map"], "#define USE_IRIDESCENCEMAP")
    _d(lines, p["iridescence_thickness_map"],
       "#define USE_IRIDESCENCE_THICKNESSMAP")
    _d(lines, p["specular_map"], "#define USE_SPECULARMAP")
    _d(lines, p["specular_color_map"], "#define USE_SPECULAR_COLORMAP")
    _d(lines, p["specular_intensity_map"], "#define USE_SPECULAR_INTENSITYMAP")
    _d(lines, p["roughness_map"], "#define USE_ROUGHNESSMAP")
    _d(lines, p["metalness_map"], "#define USE_METALNESSMAP")
    _d(lines, p["alphamap"], "#define USE_ALPHAMAP")
    _d(lines, p["alphahash"], "#define USE_ALPHAHASH")
    _d(lines, p["transmission"], "#define USE_TRANSMISSION")
    _d(lines, p["transmission_map"], "#define USE_TRANSMISSIONMAP")
    _d(lines, p["thickness_map"], "#define USE_THICKNESSMAP")
    _d(lines, p["sheen_color_map"], "#define USE_SHEEN_COLORMAP")
    _d(lines, p["sheen_roughness_map"], "#define USE_SHEEN_ROUGHNESSMAP")
    for key, macro in _UV_CHANNELS:
        if p.get(key):
            lines.append(f"#define {macro} {p[key]}")
    _d(lines, p["vertex_tangents"] and not p["flat_shading"],
       "#define USE_TANGENT")
    _d(lines, p["vertex_normals"], "#define HAS_NORMAL")
    _d(lines, p["vertex_colors"], "#define USE_COLOR")
    _d(lines, p["vertex_alphas"], "#define USE_COLOR_ALPHA")
    _d(lines, p["vertex_uv1s"], "#define USE_UV1")
    _d(lines, p["vertex_uv2s"], "#define USE_UV2")
    _d(lines, p["vertex_uv3s"], "#define USE_UV3")
    _d(lines, p["points_uvs"], "#define USE_POINTS_UV")
    _d(lines, p["flat_shading"], "#define FLAT_SHADED")
    _d(lines, p["skinning"], "#define USE_SKINNING")
    _d(lines, p["morph_targets"], "#define USE_MORPHTARGETS")
    _d(lines, p["morph_normals"] and not p["flat_shading"],
       "#define USE_MORPHNORMALS")
    _d(lines, p["morph_colors"], "#define USE_MORPHCOLORS")
    if p["morph_targets_count"] > 0:
        lines.append("#define MORPHTARGETS_TEXTURE_STRIDE "
                     f"{p['morph_texture_stride']}")
        lines.append(f"#define MORPHTARGETS_COUNT {p['morph_targets_count']}")
    _d(lines, p["double_sided"], "#define DOUBLE_SIDED")
    _d(lines, p["flip_sided"], "#define FLIP_SIDED")
    if p["shadowmap_enabled"]:
        lines.append("#define USE_SHADOWMAP")
        lines.append(f"#define {p['shadowmap_type']}")
    _d(lines, p["size_attenuation"], "#define USE_SIZEATTENUATION")
    _d(lines, p["num_light_probes"] > 0, "#define USE_LIGHT_PROBES")
    _d(lines, p["logarithmic_depth_buffer"],
       "#define USE_LOGARITHMIC_DEPTH_BUFFER")
    _d(lines, p["reversed_depth_buffer"], "#define USE_REVERSED_DEPTH_BUFFER")
    lines += [
        "uniform mat4 modelMatrix;",
        "uniform mat4 modelViewMatrix;",
        "uniform mat4 projectionMatrix;",
        "uniform mat4 viewMatrix;",
        "uniform mat3 normalMatrix;",
        "uniform vec3 cameraPosition;",
        "uniform bool isOrthographic;",
        "#ifdef USE_INSTANCING",
        "\tattribute mat4 instanceMatrix;",
        "#endif",
        "#ifdef USE_INSTANCING_COLOR",
        "\tattribute vec3 instanceColor;",
        "#endif",
        "#ifdef USE_INSTANCING_MORPH",
        "\tuniform sampler2D morphTexture;",
        "#endif",
        "attribute vec3 position;",
        "attribute vec3 normal;",
        "attribute vec2 uv;",
        "#ifdef USE_UV1",
        "\tattribute vec2 uv1;",
        "#endif",
        "#ifdef USE_UV2",
        "\tattribute vec2 uv2;",
        "#endif",
        "#ifdef USE_UV3",
        "\tattribute vec2 uv3;",
        "#endif",
        "#ifdef USE_TANGENT",
        "\tattribute vec4 tangent;",
        "#endif",
        "#if defined( USE_COLOR_ALPHA )",
        "\tattribute vec4 color;",
        "#elif defined( USE_COLOR )",
        "\tattribute vec3 color;",
        "#endif",
        "#ifdef USE_SKINNING",
        "\tattribute vec4 skinIndex;",
        "\tattribute vec4 skinWeight;",
        "#endif",
    ]
    return "\n".join(x for x in lines if x != "") + "\n"


def _texel_encoding_function(output_color_space):
    if output_color_space == "srgb":
        oetf = "sRGBTransferOETF"
    else:
        oetf = "LinearTransferOETF"
    matrix = "mat3( 1.0000,0.0000,0.0000,0.0000,1.0000,0.0000," \
             "0.0000,0.0000,1.0000 )"
    return ("vec4 linearToOutputTexel( vec4 value ) {\n"
            f"\treturn {oetf}( vec4( {matrix} * value.rgb, value.a ) );\n"
            "}")


_LUMINANCE_FUNCTION = (
    "float luminance( const in vec3 rgb ) {\n"
    "\tconst vec3 weights = vec3( 0.2126, 0.7152, 0.0722 );\n"
    "\treturn dot( weights, rgb );\n"
    "}")


def fragment_prefix(defines=None, parameters=None, chunks=None):
    """The prefix WebGLProgram prepends to a ShaderMaterial's fragment
    shader (ES 1.00 form).  See the module docstring for parameters."""
    p = dict(_FRAGMENT_DEFAULTS)
    if parameters:
        p.update(parameters)
    lines = [_generate_precision(p["precision"]),
             f"#define SHADER_TYPE {p['shader_type']}",
             f"#define SHADER_NAME {p['shader_name']}",
             _generate_defines(defines)]
    _d(lines, p["fog"], "#define USE_FOG")
    _d(lines, p["fog"] and p["fog_exp2"], "#define FOG_EXP2")
    _d(lines, p["alpha_to_coverage"], "#define ALPHA_TO_COVERAGE")
    _d(lines, p["map"], "#define USE_MAP")
    _d(lines, p["matcap"], "#define USE_MATCAP")
    if p["envmap"]:
        lines.append("#define USE_ENVMAP")
        lines.append(f"#define {p['envmap_type']}")
        lines.append(f"#define {p['envmap_mode']}")
        lines.append(f"#define {p['envmap_blending']}")
    if p["envmap_pmrem"]:
        lines.append(f"#define ENVMAP_MAX_LOD {p['envmap_max_lod']}.0")
        lines.append(f"#define ENVMAP_SIZE {p['envmap_size']}.0")
    _d(lines, p["lightmap"], "#define USE_LIGHTMAP")
    _d(lines, p["aomap"], "#define USE_AOMAP")
    _d(lines, p["bumpmap"], "#define USE_BUMPMAP")
    _d(lines, p["normalmap"], "#define USE_NORMALMAP")
    _d(lines, p["normalmap_objectspace"], "#define USE_NORMALMAP_OBJECTSPACE")
    _d(lines, p["normalmap_tangentspace"],
       "#define USE_NORMALMAP_TANGENTSPACE")
    _d(lines, p["packed_normalmap"], "#define USE_PACKED_NORMALMAP")
    _d(lines, p["emissivemap"], "#define USE_EMISSIVEMAP")
    _d(lines, p["anisotropy"], "#define USE_ANISOTROPY")
    _d(lines, p["anisotropy_map"], "#define USE_ANISOTROPYMAP")
    _d(lines, p["clearcoat"], "#define USE_CLEARCOAT")
    _d(lines, p["clearcoat_map"], "#define USE_CLEARCOATMAP")
    _d(lines, p["clearcoat_roughness_map"], "#define USE_CLEARCOAT_ROUGHNESSMAP")
    _d(lines, p["clearcoat_normal_map"], "#define USE_CLEARCOAT_NORMALMAP")
    _d(lines, p["diffuse_roughness"], "#define USE_DIFFUSE_ROUGHNESS")
    _d(lines, p["diffuse_roughness_map"], "#define USE_DIFFUSE_ROUGHNESSMAP")
    _d(lines, p["dispersion"], "#define USE_DISPERSION")
    _d(lines, p["retroreflection"], "#define USE_RETROREFLECTION")
    _d(lines, p["iridescence"], "#define USE_IRIDESCENCE")
    _d(lines, p["iridescence_map"], "#define USE_IRIDESCENCEMAP")
    _d(lines, p["iridescence_thickness_map"],
       "#define USE_IRIDESCENCE_THICKNESSMAP")
    _d(lines, p["specular_map"], "#define USE_SPECULARMAP")
    _d(lines, p["specular_color_map"], "#define USE_SPECULAR_COLORMAP")
    _d(lines, p["specular_intensity_map"], "#define USE_SPECULAR_INTENSITYMAP")
    _d(lines, p["roughness_map"], "#define USE_ROUGHNESSMAP")
    _d(lines, p["metalness_map"], "#define USE_METALNESSMAP")
    _d(lines, p["alphamap"], "#define USE_ALPHAMAP")
    _d(lines, p["alphatest"], "#define USE_ALPHATEST")
    _d(lines, p["alphahash"], "#define USE_ALPHAHASH")
    _d(lines, p["sheen"], "#define USE_SHEEN")
    _d(lines, p["sheen_color_map"], "#define USE_SHEEN_COLORMAP")
    _d(lines, p["sheen_roughness_map"], "#define USE_SHEEN_ROUGHNESSMAP")
    _d(lines, p["transmission"], "#define USE_TRANSMISSION")
    _d(lines, p["transmission_map"], "#define USE_TRANSMISSIONMAP")
    _d(lines, p["thickness_map"], "#define USE_THICKNESSMAP")
    _d(lines, p["vertex_tangents"] and not p["flat_shading"],
       "#define USE_TANGENT")
    _d(lines, p["vertex_colors"] or p["instancing_color"],
       "#define USE_COLOR")
    _d(lines, p["vertex_alphas"] or p["batching_color"],
       "#define USE_COLOR_ALPHA")
    _d(lines, p["vertex_uv1s"], "#define USE_UV1")
    _d(lines, p["vertex_uv2s"], "#define USE_UV2")
    _d(lines, p["vertex_uv3s"], "#define USE_UV3")
    _d(lines, p["points_uvs"], "#define USE_POINTS_UV")
    _d(lines, p["gradientmap"], "#define USE_GRADIENTMAP")
    _d(lines, p["flat_shading"], "#define FLAT_SHADED")
    _d(lines, p["double_sided"], "#define DOUBLE_SIDED")
    _d(lines, p["flip_sided"], "#define FLIP_SIDED")
    if p["shadowmap_enabled"]:
        lines.append("#define USE_SHADOWMAP")
        lines.append(f"#define {p['shadowmap_type']}")
    _d(lines, p["premultiplied_alpha"], "#define PREMULTIPLIED_ALPHA")
    _d(lines, p["num_light_probes"] > 0, "#define USE_LIGHT_PROBES")
    _d(lines, p["decode_video_texture"], "#define DECODE_VIDEO_TEXTURE")
    _d(lines, p["decode_video_texture_emissive"],
       "#define DECODE_VIDEO_TEXTURE_EMISSIVE")
    _d(lines, p["logarithmic_depth_buffer"],
       "#define USE_LOGARITHMIC_DEPTH_BUFFER")
    _d(lines, p["reversed_depth_buffer"], "#define USE_REVERSED_DEPTH_BUFFER")
    lines += [
        "uniform mat4 viewMatrix;",
        "uniform vec3 cameraPosition;",
        "uniform bool isOrthographic;",
    ]
    if p["tone_mapping"]:
        name = _TONE_MAPPING_NAMES.get(str(p["tone_mapping"]).lower())
        if name is None:
            name = "Linear"
        lines.append("#define TONE_MAPPING")
        lines.append(_chunk_text(chunks, "tonemapping_pars_fragment"))
        lines.append(f"vec3 toneMapping( vec3 color ) "
                     f"{{ return {name}ToneMapping( color ); }}")
    _d(lines, p["dithering"], "#define DITHERING")
    _d(lines, p["opaque"], "#define OPAQUE")
    lines.append(_chunk_text(chunks, "colorspace_pars_fragment"))
    lines.append(_texel_encoding_function(p["output_color_space"]))
    lines.append(_LUMINANCE_FUNCTION)
    if p["use_depth_packing"] is not None:
        lines.append(f"#define DEPTH_PACKING {p['use_depth_packing']}")
    return "\n".join(x for x in lines if x != "") + "\n"


def raw_prefix(defines=None, parameters=None):
    """The prefix for RawShaderMaterial: SHADER_TYPE / SHADER_NAME and
    the material's custom defines only (both stages)."""
    p = {"shader_type": "RawShaderMaterial", "shader_name": ""}
    if parameters:
        p.update(parameters)
    lines = [f"#define SHADER_TYPE {p['shader_type']}",
             f"#define SHADER_NAME {p['shader_name']}",
             _generate_defines(defines)]
    text = "\n".join(x for x in lines if x != "")
    return text + "\n" if text else ""


# ---------------------------------------------------------------------------
# Parser (recursive descent; AST contract in the module docstring)
# ---------------------------------------------------------------------------

BUILTIN_TYPES = {
    "float", "int", "uint", "bool", "void",
    "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4",
    "uvec2", "uvec3", "uvec4", "bvec2", "bvec3", "bvec4",
    "mat2", "mat3", "mat4",
    "sampler2D", "samplerCube", "sampler3D", "sampler2DArray",
    "sampler2DShadow", "samplerCubeShadow", "sampler2DArrayShadow",
    "isampler2D", "isampler3D", "isamplerCube", "isampler2DArray",
    "usampler2D", "usampler3D", "usamplerCube", "usampler2DArray",
}

QUALIFIERS = {
    "uniform", "varying", "attribute", "in", "out", "inout", "const",
    "highp", "mediump", "lowp", "invariant",
}


class Program:
    """A parsed shader: top-level items + struct table (see the
    module docstring for the AST contract)."""

    def __init__(self, items, structs, stage=None):
        self.items = items
        self.structs = structs
        self.stage = stage

    def functions(self):
        return [it for it in self.items if it[0] == "func"]

    def function_names(self):
        return [it[2] for it in self.functions()]

    def globals(self):
        return [it for it in self.items if it[0] == "global"]

    def _by_qual(self, qual):
        return [(it[2], it[3], it[5]) for it in self.globals()
                if qual in it[1]]

    def uniforms(self):
        return self._by_qual("uniform")

    def varyings(self):
        return self._by_qual("varying")

    def attributes(self):
        return self._by_qual("attribute")


class Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0
        self.structs = {}

    # -- token plumbing ----------------------------------------------------
    def peek(self, k=0):
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self):
        t = self.toks[self.i]
        if t.kind != "eof":
            self.i += 1
        return t

    def expect(self, text):
        t = self.next()
        if t.text != text:
            raise GLSLError(
                f"expected {text!r}, got {t.text!r} (line {t.line})")

    def accept(self, text):
        if self.peek().text == text:
            self.i += 1
            return True
        return False

    def _is_type(self, text):
        return text in BUILTIN_TYPES or text in self.structs

    # -- qualifiers ----------------------------------------------------------
    def _parse_qualifiers(self):
        quals = []
        while True:
            t = self.peek().text
            if t in QUALIFIERS:
                quals.append(self.next().text)
            elif t == "layout":
                self.next()
                self.expect("(")
                inner = []
                depth = 1
                while depth:
                    tk = self.next()
                    if tk.kind == "eof":
                        raise GLSLError("unterminated layout(...)")
                    if tk.text == "(":
                        depth += 1
                    elif tk.text == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    inner.append(tk.text)
                quals.append(("layout", " ".join(inner)))
            else:
                return quals

    def _skip_qualifiers(self, j):
        """Index just past any qualifiers starting at token index j."""
        n = len(self.toks)
        while j < n:
            t = self.toks[j].text
            if t in QUALIFIERS:
                j += 1
            elif t == "layout":
                depth = 0
                j += 1
                while j < n:
                    if self.toks[j].text == "(":
                        depth += 1
                    elif self.toks[j].text == ")":
                        depth -= 1
                        if depth == 0:
                            j += 1
                            break
                    j += 1
            else:
                break
        return j

    # -- top level -----------------------------------------------------------
    def parse_program(self):
        items = []
        while self.peek().kind != "eof":
            if self.accept(";"):
                continue
            if self.peek().text == "struct":
                items.extend(self._parse_struct())
                continue
            if self.peek().text == "precision":
                self.next()
                qual = self.next().text
                typ = self.next().text
                self.expect(";")
                items.append(("precision", qual, typ))
                continue
            quals = self._parse_qualifiers()
            ttype = self.next()
            if ttype.kind != "ident":
                raise GLSLError(
                    f"expected a type at top level, got {ttype.text!r} "
                    f"(line {ttype.line})")
            name = self.next()
            if name.kind != "ident":
                raise GLSLError(
                    f"expected a name after type {ttype.text!r} "
                    f"(line {name.line})")
            if self.accept("("):
                params = self._parse_params()
                self.expect(")")
                if self.accept(";"):
                    items.append(("funcdecl", ttype.text, name.text, params))
                else:
                    body = self.parse_block()
                    items.append(("func", ttype.text, name.text, params,
                                  body))
            else:
                items.extend(self._parse_declarators(
                    quals, ttype.text, name.text, "global"))
        return Program(items, self.structs)

    def _parse_declarators(self, quals, typ, first_name, kind):
        """Parse ``name [arr] [= init], ... ;`` given the first name.
        kind "global" -> one item per declarator; kind "decl" -> a
        single statement node holding all declarators."""
        decls = []
        name = first_name
        while True:
            arr = self._parse_array_size()
            init = None
            if self.accept("="):
                init = self.parse_expr()
            decls.append((name, init, arr))
            if self.accept(","):
                nt = self.next()
                if nt.kind != "ident":
                    raise GLSLError(
                        f"expected a declarator name (line {nt.line})")
                name = nt.text
                continue
            break
        self.expect(";")
        if kind == "global":
            return [("global", quals, typ, n, init, arr)
                    for (n, init, arr) in decls]
        return [("decl", quals, typ, decls)]

    def _parse_array_size(self):
        if not self.accept("["):
            return None
        if self.accept("]"):
            return UNSIZED
        size = self.parse_expr()
        self.expect("]")
        return size

    def _parse_struct(self):
        self.expect("struct")
        name_tok = self.next()
        if name_tok.kind != "ident":
            raise GLSLError(
                f"expected a struct name (line {name_tok.line})")
        name = name_tok.text
        self.expect("{")
        fields = []
        while self.peek().text != "}":
            self._parse_qualifiers()  # precision on fields: parsed, dropped
            ftype = self.next().text
            while True:
                fname = self.next()
                if fname.kind != "ident":
                    raise GLSLError(
                        f"expected a field name (line {fname.line})")
                arr = self._parse_array_size()
                fields.append((ftype, fname.text, arr))
                if not self.accept(","):
                    break
            self.expect(";")
        self.expect("}")
        self.structs[name] = fields
        items = [("structdef", name, fields)]
        # Optional instance declarator: ``struct S { ... } s;``
        if self.peek().kind == "ident":
            inst = self.next().text
            items.extend(self._parse_declarators([], name, inst, "global"))
        else:
            self.expect(";")
        return items

    def _parse_params(self):
        params = []
        if self.peek().text == ")":
            return params
        if self.peek().text == "void" and self.peek(1).text == ")":
            self.next()
            return params
        while True:
            quals = self._parse_qualifiers()
            ttype = self.next()
            if ttype.kind != "ident":
                raise GLSLError(
                    f"expected a parameter type, got {ttype.text!r} "
                    f"(line {ttype.line})")
            name = None
            if self.peek().kind == "ident":
                name = self.next().text
            arr = self._parse_array_size()
            params.append((quals, ttype.text, name, arr))
            if not self.accept(","):
                break
        return params

    # -- statements ----------------------------------------------------------
    def parse_block(self):
        self.expect("{")
        stmts = []
        while self.peek().text != "}":
            if self.peek().kind == "eof":
                raise GLSLError("unterminated block")
            stmts.append(self.parse_stmt())
        self.expect("}")
        return ("block", stmts)

    def parse_stmt(self):
        t = self.peek().text
        if t == "{":
            return self.parse_block()
        if t == "if":
            self.next()
            self.expect("(")
            cond = self.parse_expr()
            self.expect(")")
            then = self.parse_stmt()
            els = None
            if self.peek().text == "else":
                self.next()
                els = self.parse_stmt()
            return ("if", cond, then, els)
        if t == "for":
            self.next()
            self.expect("(")
            init = None
            if self.peek().text != ";":
                if self._looks_like_decl():
                    init = self._parse_local_decl()
                else:
                    init = ("exprstmt", self.parse_expr())
                    self.expect(";")
            else:
                self.next()
            cond = None
            if self.peek().text != ";":
                cond = self.parse_expr()
            self.expect(";")
            step = None
            if self.peek().text != ")":
                step = self.parse_expr()
            self.expect(")")
            body = self.parse_stmt()
            return ("for", init, cond, step, body)
        if t == "while":
            self.next()
            self.expect("(")
            cond = self.parse_expr()
            self.expect(")")
            return ("while", cond, self.parse_stmt())
        if t == "do":
            self.next()
            body = self.parse_stmt()
            self.expect("while")
            self.expect("(")
            cond = self.parse_expr()
            self.expect(")")
            self.expect(";")
            return ("dowhile", body, cond)
        if t == "return":
            self.next()
            e = None
            if self.peek().text != ";":
                e = self.parse_expr()
            self.expect(";")
            return ("return", e)
        if t == "break":
            self.next()
            self.expect(";")
            return ("break",)
        if t == "continue":
            self.next()
            self.expect(";")
            return ("continue",)
        if t == "discard":
            self.next()
            self.expect(";")
            return ("discard",)
        if t == ";":
            self.next()
            return ("block", [])
        if self._looks_like_decl():
            return self._parse_local_decl()
        e = self.parse_expr()
        self.expect(";")
        return ("exprstmt", e)

    def _looks_like_decl(self):
        j = self._skip_qualifiers(self.i)
        if j >= len(self.toks):
            return False
        if not self._is_type(self.toks[j].text):
            return False
        return j + 1 < len(self.toks) and self.toks[j + 1].kind == "ident"

    def _parse_local_decl(self):
        quals = self._parse_qualifiers()
        ttype = self.next().text
        name = self.next()
        if name.kind != "ident":
            raise GLSLError(
                f"expected a variable name (line {name.line})")
        return self._parse_declarators(quals, ttype, name.text, "decl")[0]

    # -- expressions ---------------------------------------------------------
    def parse_expr(self):
        return self._parse_assign()

    def _parse_assign(self):
        lhs = self._parse_ternary()
        if self.peek().text in ("=", "+=", "-=", "*=", "/=", "%=",
                                "<<=", ">>=", "&=", "^=", "|="):
            op = self.next().text
            rhs = self._parse_assign()
            return ("assign", op, lhs, rhs)
        return lhs

    def _parse_ternary(self):
        e = self._parse_or()
        if self.accept("?"):
            a = self.parse_expr()
            self.expect(":")
            b = self._parse_ternary()
            return ("ternary", e, a, b)
        return e

    def _bin(self, sub, ops):
        e = sub()
        while self.peek().text in ops:
            op = self.next().text
            e = ("bin", op, e, sub())
        return e

    def _parse_or(self):
        return self._bin(self._parse_and, ("||",))

    def _parse_and(self):
        return self._bin(self._parse_bitor, ("&&",))

    def _parse_bitor(self):
        return self._bin(self._parse_bitxor, ("|",))

    def _parse_bitxor(self):
        return self._bin(self._parse_bitand, ("^",))

    def _parse_bitand(self):
        return self._bin(self._parse_eq, ("&",))

    def _parse_eq(self):
        return self._bin(self._parse_rel, ("==", "!="))

    def _parse_rel(self):
        return self._bin(self._parse_shift, ("<", ">", "<=", ">="))

    def _parse_shift(self):
        return self._bin(self._parse_add, ("<<", ">>"))

    def _parse_add(self):
        return self._bin(self._parse_mul, ("+", "-"))

    def _parse_mul(self):
        return self._bin(self._parse_unary, ("*", "/", "%"))

    def _parse_unary(self):
        t = self.peek().text
        if t in ("-", "+", "!", "~", "++", "--"):
            self.next()
            return ("unary", t, self._parse_unary())
        return self._parse_postfix()

    def _parse_postfix(self):
        e = self._parse_primary()
        while True:
            t = self.peek().text
            if t == "(":
                self.next()
                args = self._parse_args()
                self.expect(")")
                e = ("call", e, args)
            elif t == "[":
                # Array constructor: Type [ size ] ( args )
                if e[0] == "ident" and self._is_type(e[1]):
                    save = self.i
                    self.next()
                    size = None
                    if self.peek().text != "]":
                        size = self.parse_expr()
                    if self.accept("]") and self.peek().text == "(":
                        self.next()
                        args = self._parse_args()
                        self.expect(")")
                        e = ("arrctor", e[1], size, args)
                        continue
                    self.i = save
                self.next()
                idx = self.parse_expr()
                self.expect("]")
                e = ("index", e, idx)
            elif t == ".":
                self.next()
                nt = self.next()
                if nt.kind != "ident":
                    raise GLSLError(
                        f"expected a member name (line {nt.line})")
                e = ("field", e, nt.text)
            elif t in ("++", "--"):
                self.next()
                e = ("postfix", t, e)
            else:
                break
        return e

    def _parse_args(self):
        args = []
        if self.peek().text == ")":
            return args
        while True:
            args.append(self.parse_expr())
            if not self.accept(","):
                break
        return args

    def _parse_primary(self):
        t = self.next()
        if t.kind == "num":
            return _num_node(t.text)
        if t.text == "(":
            e = self.parse_expr()
            self.expect(")")
            return e
        if t.text == "true":
            return ("num", True, "bool")
        if t.text == "false":
            return ("num", False, "bool")
        if t.kind == "ident":
            return ("ident", t.text)
        raise GLSLError(f"unexpected token {t.text!r} (line {t.line})")


def _num_node(text):
    if text[:2] in ("0x", "0X"):
        return ("num", int(text.rstrip("uU"), 16), "int")
    if text.endswith(("u", "U")) and not any(c in text for c in ".eE"):
        return ("num", int(text.rstrip("uU")), "int")
    if any(c in text for c in ".eE") or text.endswith(("f", "F")):
        return ("num", float(text.rstrip("fF")), "float")
    return ("num", int(text), "int")


def parse(source, stage=None):
    """Parse preprocessed GLSL source into a Program (see the module
    docstring for the AST contract)."""
    parser = Parser(lex(source))
    prog = parser.parse_program()
    prog.stage = stage
    return prog


def parse_shader(source, stage="fragment", defines=None, chunks=None,
                 raw=False, parameters=None):
    """Full ShaderMaterial pipeline: prefix + preprocess + parse."""
    if chunks is None:
        chunks = load_chunks()
    if raw:
        text = raw_prefix(defines, parameters) + source
    elif stage == "vertex":
        text = vertex_prefix(defines, parameters) + source
    else:
        text = fragment_prefix(defines, parameters, chunks=chunks) + source
    return parse(preprocess(text, defines=defines, chunks=chunks),
                 stage=stage)

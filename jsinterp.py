# jsinterp.py -- a small, self-written JavaScript subset interpreter.
#
# Written from scratch (lexer / recursive-descent parser / tree-walking
# evaluator) to run three.js introductory-sample level code inside Pyxel,
# where no JS engine and no exec/eval are available.
#
# Interface:
#   JSInterpreter(three_module, host)
#     .run(source: str)   -- parse and execute a program
#     .call_raf()         -- run every registered requestAnimationFrame
#                            callback once (callbacks may re-register)
#   Host globals provided: THREE, requestAnimationFrame, document (stub),
#   window (stub), console.log.  All JS errors are raised as JSError.
#
# Supported grammar (core set, plus a few harmless extras marked *):
#   import statements (resolved: 'three' binds from the host module,
#   a small registry of 'three/addons/...' paths binds built-ins such
#   as OrbitControls; unknown modules keep the historical skip
#   behaviour), export lines (skipped)*, const/let/var, function
#   declarations,
#   function expressions*, arrow functions, new, member access/assignment,
#   index access/assignment, calls, object literals, array literals,
#   hex/decimal numbers, strings, template literals (full ${}
#   expression interpolation), binary + - * / % == != === !==
#   < > <= >= && ||, unary - ! + typeof*, compound assignment
#   += -= *= /= %=*, ++/--*, ternary ?:*, if/else, for, for-of/for-in*,
#   while, do-while*, return, break/continue*, // and /* */ comments,
#   missing semicolons (ASI-tolerant).
#   Also: class declarations/expressions (extends,
#   constructor, methods, statics, get/set, field initializers, super
#   calls and super member access), try/catch/finally, throw,
#   destructuring declarations/parameters/assignments (array and
#   object patterns, defaults, rest), spread in array literals, call
#   arguments and object literals, rest parameters, Array prototype
#   methods (map/filter/forEach/reduce/find/findIndex/includes/
#   indexOf/slice/join/concat/reverse/sort) and String methods
#   (split/includes/startsWith/endsWith/indexOf/slice/substring/
#   charAt/trim/replace/repeat/padStart/...).
#
# Not supported (raises JSError): switch, generators, async/await,
# getters/setters in object literals, labels, regex literals,
# optional chaining.

import array
import datetime
import json
import math
import re
import struct
import time


class JSError(Exception):
    """Any JavaScript-level error (lex, parse, runtime)."""


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _ThrowJS(Exception):
    """A JS-level ``throw`` of an arbitrary value (not an interpreter
    error).  Caught by JS try/catch; converted to JSError if it escapes
    to the host."""

    def __init__(self, value):
        super().__init__("uncaught throw")
        self.value = value


UNDEFINED = None  # JS undefined / null are both represented by None.


# --------------------------------------------------------------------------
# Lexer
# --------------------------------------------------------------------------

_KEYWORDS = {
    "const", "let", "var", "function", "return", "if", "else", "for",
    "while", "do", "new", "typeof", "true", "false", "null", "undefined",
    "this", "break", "continue", "in", "of", "import", "export", "from",
    "class", "switch", "try", "catch", "finally", "throw", "delete",
    "void", "instanceof",
}

# Longest first so prefix matching works.
_PUNCTS = [
    "===", "!==", "...", "**=", ">>>=", "<<=", ">>=",
    "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "%=",
    "&=", "|=", "^=",
    "++", "--", "=>", "**", "<<", ">>>", ">>", "?.",
    "+", "-", "*", "/", "%", "=", "<", ">", "!", "?", ":", ".", ",",
    "&", "|", "^", "~",
    ";", "(", ")", "{", "}", "[", "]",
]


class Token:
    __slots__ = ("kind", "value", "line")

    def __init__(self, kind, value, line):
        self.kind = kind      # 'num' | 'str' | 'ident' | 'kw' | 'punct' | 'eof'
        self.value = value
        self.line = line

    def __repr__(self):
        return f"Token({self.kind},{self.value!r},L{self.line})"


_REGEX_PREV_PUNCTS = frozenset({
    "(", "[", "{", ",", ";", ":", "?", "=", "+=", "-=", "*=", "/=", "%=",
    "==", "===", "!=", "!==", "<", "<=", ">", ">=", "&&", "||", "??",
    "!", "~", "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", ">>>",
    "=>", "...",
})
_REGEX_PREV_KWS = frozenset({
    "return", "throw", "typeof", "void", "delete", "in", "instanceof",
    "case", "else",
})


def _can_start_regex(toks):
    if not toks:
        return True
    prev = toks[-1]
    if prev.kind == "punct":
        return prev.value in _REGEX_PREV_PUNCTS
    if prev.kind == "kw":
        return prev.value in _REGEX_PREV_KWS
    if prev.kind == "ident" and prev.value == "await":
        return True
    return False


def _lex(source):
    toks = []
    i, n, line = 0, len(source), 1
    while i < n:
        c = source[i]
        if c in " \t\r":
            i += 1
            continue
        if c == "\n":
            line += 1
            i += 1
            continue
        # comments
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            while i < n and source[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and source[i + 1] == "*":
            j = source.find("*/", i + 2)
            if j < 0:
                raise JSError(f"line {line}: unterminated /* comment")
            line += source.count("\n", i, j)
            i = j + 2
            continue
        # regex literal: /pattern/flags when '/' cannot be division
        if c == "/" and _can_start_regex(toks):
            j = i + 1
            in_class = False
            pat_buf = []
            closed = False
            while j < n:
                rc = source[j]
                if rc == "\n":
                    break
                if rc == "\\" and j + 1 < n:
                    pat_buf.append(rc)
                    pat_buf.append(source[j + 1])
                    j += 2
                    continue
                if rc == "[":
                    in_class = True
                elif rc == "]" and in_class:
                    in_class = False
                elif rc == "/" and not in_class:
                    j += 1
                    closed = True
                    break
                pat_buf.append(rc)
                j += 1
            if closed:
                fstart = j
                while j < n and source[j].isalpha():
                    j += 1
                flags = source[fstart:j]
                toks.append(Token("regex", ("".join(pat_buf), flags), line))
                i = j
                continue
        # numbers (decimal, float, exponent, hex 0x, binary 0b, octal 0o)
        if c.isdigit() or (c == "." and i + 1 < n and source[i + 1].isdigit()):
            start = i
            if c == "0" and i + 1 < n and source[i + 1] in "xX":
                i += 2
                hs = i
                while i < n and source[i] in "0123456789abcdefABCDEF":
                    i += 1
                if i == hs:
                    raise JSError(f"line {line}: malformed hex number")
                toks.append(Token("num", float(int(source[hs:i], 16)), line))
                continue
            if c == "0" and i + 1 < n and source[i + 1] in "bB":
                i += 2
                hs = i
                while i < n and source[i] in "01":
                    i += 1
                toks.append(Token("num", float(int(source[hs:i] or "0", 2)), line))
                continue
            if c == "0" and i + 1 < n and source[i + 1] in "oO":
                i += 2
                hs = i
                while i < n and source[i] in "01234567":
                    i += 1
                toks.append(Token("num", float(int(source[hs:i] or "0", 8)), line))
                continue
            while i < n and (source[i].isdigit() or source[i] == "_"):
                i += 1
            if i < n and source[i] == ".":
                i += 1
                while i < n and source[i].isdigit():
                    i += 1
            if i < n and source[i] in "eE":
                j = i + 1
                if j < n and source[j] in "+-":
                    j += 1
                if j < n and source[j].isdigit():
                    i = j
                    while i < n and source[i].isdigit():
                        i += 1
            text = source[start:i].replace("_", "")
            try:
                val = float(text)
            except ValueError:
                raise JSError(f"line {line}: malformed number {text!r}")
            toks.append(Token("num", val, line))
            continue
        # strings
        if c in "\"'":
            quote = c
            i += 1
            out = []
            while i < n and source[i] != quote:
                ch = source[i]
                if ch == "\n":
                    raise JSError(f"line {line}: unterminated string")
                if ch == "\\" and i + 1 < n:
                    i += 1
                    esc = source[i]
                    out.append({"n": "\n", "t": "\t", "r": "\r", "0": "\0",
                                "\\": "\\", "'": "'", '"': '"', "`": "`",
                                "/": "/"}.get(esc, esc))
                    i += 1
                    continue
                out.append(ch)
                i += 1
            if i >= n:
                raise JSError(f"line {line}: unterminated string")
            i += 1
            toks.append(Token("str", "".join(out), line))
            continue
        # template literal: kept as a string token with raw ${...} spans
        # recorded; interpolation of simple expressions is done by parser.
        if c == "`":
            i += 1
            parts = []  # list of ('lit', str) or ('expr', source-str)
            buf = []
            closed = False
            while i < n:
                ch = source[i]
                if ch == "`":
                    i += 1
                    closed = True
                    break
                if ch == "\\" and i + 1 < n:
                    esc = source[i + 1]
                    buf.append({"n": "\n", "t": "\t", "r": "\r", "0": "\0",
                                "\\": "\\", "'": "'", '"': '"', "`": "`",
                                "/": "/", "$": "$"}.get(esc, esc))
                    i += 2
                    continue
                if ch == "$" and i + 1 < n and source[i + 1] == "{":
                    depth = 1
                    j = i + 2
                    while j < n and depth:
                        cj = source[j]
                        if cj in "\"'`":
                            # Skip a string / nested template span so
                            # braces inside it do not disturb the count.
                            quote = cj
                            j += 1
                            while j < n and source[j] != quote:
                                if source[j] == "\\":
                                    j += 1
                                j += 1
                            j += 1
                            continue
                        if cj == "{":
                            depth += 1
                        elif cj == "}":
                            depth -= 1
                        j += 1
                    parts.append(("lit", "".join(buf)))
                    buf = []
                    parts.append(("expr", source[i + 2:j - 1]))
                    i = j
                    continue
                if ch == "\n":
                    line += 1
                buf.append(ch)
                i += 1
            if not closed:
                raise JSError(f"line {line}: unterminated template literal")
            parts.append(("lit", "".join(buf)))
            toks.append(Token("tmpl", parts, line))
            continue
        # identifiers / keywords
        if c.isalpha() or c in "_$":
            start = i
            while i < n and (source[i].isalnum() or source[i] in "_$"):
                i += 1
            word = source[start:i]
            kind = "kw" if word in _KEYWORDS else "ident"
            toks.append(Token(kind, word, line))
            continue
        # punctuation
        matched = None
        for p in _PUNCTS:
            if source.startswith(p, i):
                matched = p
                break
        if matched is None:
            raise JSError(f"line {line}: unexpected character {c!r}")
        if matched == "?.":
            raise JSError(f"line {line}: optional chaining (?.) not supported")
        toks.append(Token("punct", matched, line))
        i += len(matched)
    toks.append(Token("eof", None, line))
    return toks


# --------------------------------------------------------------------------
# AST nodes are plain tuples: (tag, ...)
# --------------------------------------------------------------------------

class Parser:
    def __init__(self, toks):
        self.toks = toks
        self.pos = 0

    # -- token helpers ------------------------------------------------------
    def peek(self, k=0):
        return self.toks[min(self.pos + k, len(self.toks) - 1)]

    def next(self):
        t = self.toks[self.pos]
        if self.pos < len(self.toks) - 1:
            self.pos += 1
        return t

    def at_punct(self, p):
        t = self.peek()
        return t.kind == "punct" and t.value == p

    def at_kw(self, w):
        t = self.peek()
        return t.kind == "kw" and t.value == w

    def eat_punct(self, p):
        if self.at_punct(p):
            self.next()
            return True
        return False

    def expect_punct(self, p):
        t = self.next()
        if t.kind != "punct" or t.value != p:
            raise JSError(f"line {t.line}: expected {p!r}, got {t.value!r}")
        return t

    def expect_ident(self):
        t = self.next()
        if t.kind != "ident":
            raise JSError(f"line {t.line}: expected identifier, got {t.value!r}")
        return t.value

    def semi(self):
        # Automatic semicolon insertion, simplified: eat any ';'.
        while self.at_punct(";"):
            self.next()

    # -- program / statements ----------------------------------------------
    def parse_program(self):
        body = []
        while self.peek().kind != "eof":
            stmt = self.parse_statement(top=True)
            if stmt is not None:
                body.append(stmt)
        return ("program", body)

    def parse_statement(self, top=False):
        t = self.peek()
        if t.kind == "punct" and t.value == ";":
            self.next()
            return None
        if t.kind == "ident" and t.value == "async" and \
                self.peek(1).kind == "kw" and self.peek(1).value == "function":
            self.next()
            self.next()
            name = self.expect_ident()
            params, body = self.parse_function_rest()
            return ("funcdecl", name, params, body)
        if t.kind == "kw":
            w = t.value
            if w == "import":
                return self.parse_import()
            if w == "export":
                self._skip_export()
                return self.parse_statement(top=top)
            if w in ("const", "let", "var"):
                return self.parse_var_decl()
            if w == "function":
                self.next()
                name = self.expect_ident()
                params, body = self.parse_function_rest()
                return ("funcdecl", name, params, body)
            if w == "if":
                return self.parse_if()
            if w == "for":
                return self.parse_for()
            if w == "while":
                return self.parse_while()
            if w == "do":
                return self.parse_do_while()
            if w == "return":
                self.next()
                if self.at_punct(";") or self.at_punct("}") or \
                        self.peek().kind == "eof":
                    self.semi()
                    return ("return", None)
                expr = self.parse_expression()
                self.semi()
                return ("return", expr)
            if w == "break":
                self.next()
                self.semi()
                return ("break",)
            if w == "continue":
                self.next()
                self.semi()
                return ("continue",)
            if w == "class":
                return self.parse_class_decl()
            if w == "try":
                return self.parse_try()
            if w == "throw":
                self.next()
                expr = self.parse_expression()
                self.semi()
                return ("throw", expr)
            if w == "switch":
                return self.parse_switch()
        if t.kind == "punct" and t.value == "{":
            return self.parse_block()
        expr = self.parse_expression()
        self.semi()
        return ("exprstmt", expr)

    def _skip_line_statement(self):
        # import ... ;  -- imports occupy one source line in sample code;
        # skip to ';' or to end of the starting line, whichever is first.
        start = self.next()
        while self.peek().kind != "eof":
            t = self.peek()
            if t.kind == "punct" and t.value == ";":
                self.next()
                return
            if t.line > start.line:
                return
            self.next()

    def parse_import(self):
        # import ... from '...' ;   /   import '...' ;
        # Produces ("import", bindings, source) where each binding is
        # (kind, imported_name, local_name) with kind in
        # "namespace" / "default" / "named".  Resolution happens at run
        # time in JSInterpreter._exec_import.
        self.next()  # consume 'import'
        bindings = []
        t = self.peek()
        if t.kind == "str":
            source = self.next().value
            self.semi()
            return ("import", bindings, source)
        if t.kind == "punct" and t.value == "*":
            self.next()
            self._expect_word("as")
            bindings.append(("namespace", None, self.expect_ident()))
        elif t.kind == "punct" and t.value == "{":
            self._parse_import_names(bindings)
        else:
            bindings.append(("default", None, self.expect_ident()))
            if self.eat_punct(","):
                if self.at_punct("*"):
                    self.next()
                    self._expect_word("as")
                    bindings.append(("namespace", None, self.expect_ident()))
                elif self.at_punct("{"):
                    self._parse_import_names(bindings)
        if not self.at_kw("from"):
            tt = self.peek()
            raise JSError(
                f"line {tt.line}: expected 'from' in import statement")
        self.next()
        src = self.next()
        if src.kind != "str":
            raise JSError(
                f"line {src.line}: expected module string in import")
        self.semi()
        return ("import", bindings, src.value)

    def _expect_word(self, word):
        t = self.next()
        if t.kind != "ident" or t.value != word:
            raise JSError(f"line {t.line}: expected {word!r}, got {t.value!r}")

    def _parse_import_names(self, bindings):
        self.expect_punct("{")
        while not self.at_punct("}"):
            imported = self.expect_ident()
            local = imported
            t = self.peek()
            if t.kind == "ident" and t.value == "as":
                self.next()
                local = self.expect_ident()
            bindings.append(("named", imported, local))
            if not self.eat_punct(","):
                break
        self.expect_punct("}")

    def _skip_export(self):
        # 'export' prefix: drop the keyword (and a possible 'default'),
        # then parse the following declaration normally.
        self.next()
        if self.peek().kind == "ident" and self.peek().value == "default":
            self.next()

    def parse_block(self):
        self.expect_punct("{")
        body = []
        while not self.at_punct("}"):
            if self.peek().kind == "eof":
                raise JSError("unexpected end of input inside block")
            stmt = self.parse_statement()
            if stmt is not None:
                body.append(stmt)
        self.expect_punct("}")
        return ("block", body)

    def parse_var_decl(self):
        kind = self.next().value  # const / let / var
        decls = []
        while True:
            pattern = self.parse_pattern()
            init = None
            if self.eat_punct("="):
                init = self.parse_assignment()
            decls.append((pattern, init))
            if not self.eat_punct(","):
                break
        self.semi()
        return ("vardecl", kind, decls)

    # -- patterns (destructuring) -------------------------------------------
    # Pattern AST:
    #   ("pident", name)
    #   ("parr", [None | (subpattern, default_node|None) | ("prest", name)])
    #   ("pobj", [(key, subpattern, default_node|None)], rest_name|None)
    #   ("pmember", target_expr_node)   -- only in assignment destructuring
    def parse_pattern(self):
        t = self.peek()
        if t.kind == "punct" and t.value == "[":
            self.next()
            elems = []
            while not self.at_punct("]"):
                if self.eat_punct(","):
                    elems.append(None)  # hole
                    continue
                if self.at_punct("..."):
                    self.next()
                    elems.append(("prest", self.expect_ident()))
                    break
                sub = self.parse_pattern()
                default = None
                if self.eat_punct("="):
                    default = self.parse_assignment()
                elems.append((sub, default))
                if not self.eat_punct(","):
                    break
            self.expect_punct("]")
            return ("parr", elems)
        if t.kind == "punct" and t.value == "{":
            self.next()
            props = []
            rest = None
            while not self.at_punct("}"):
                if self.at_punct("..."):
                    self.next()
                    rest = self.expect_ident()
                    break
                kt = self.next()
                if kt.kind in ("ident", "kw"):
                    key = kt.value
                elif kt.kind == "str":
                    key = kt.value
                elif kt.kind == "num":
                    key = _num_to_str(kt.value)
                else:
                    raise JSError(f"line {kt.line}: bad pattern key")
                if self.eat_punct(":"):
                    sub = self.parse_pattern()
                else:
                    sub = ("pident", key)
                default = None
                if self.eat_punct("="):
                    default = self.parse_assignment()
                props.append((key, sub, default))
                if not self.eat_punct(","):
                    break
            self.expect_punct("}")
            return ("pobj", props, rest)
        return ("pident", self.expect_ident())

    def parse_param_list(self):
        # Current position is just after '('.  Parses patterns with
        # optional defaults and a trailing ...rest, consuming the ')'.
        # Returns a list of (pattern, default_node|None); a rest
        # parameter is the pattern ("prest", name).
        params = []
        if not self.at_punct(")"):
            while True:
                if self.at_punct("..."):
                    self.next()
                    params.append((("prest", self.expect_ident()), None))
                    break
                pattern = self.parse_pattern()
                default = None
                if self.eat_punct("="):
                    default = self.parse_assignment()
                params.append((pattern, default))
                if not self.eat_punct(","):
                    break
                if self.at_punct(")"):
                    break  # trailing comma (ES2017)
        self.expect_punct(")")
        return params

    def parse_function_rest(self):
        self.expect_punct("(")
        params = self.parse_param_list()
        block = self.parse_block()
        return params, block[1]

    def parse_if(self):
        self.next()  # if
        self.expect_punct("(")
        cond = self.parse_expression()
        self.expect_punct(")")
        then = self.parse_statement()
        otherwise = None
        if self.at_kw("else"):
            self.next()
            otherwise = self.parse_statement()
        return ("if", cond, then, otherwise)

    def parse_for(self):
        self.next()  # for
        self.expect_punct("(")
        init = None
        if self.at_punct(";"):
            init = None
        elif self.peek().kind == "kw" and self.peek().value in ("const", "let", "var"):
            kw = self.next().value
            pattern = self.parse_pattern()
            if self.at_kw("of") or self.at_kw("in"):
                mode = self.next().value
                iterable = self.parse_expression()
                self.expect_punct(")")
                body = self.parse_statement()
                tag = "forof" if mode == "of" else "forin"
                return (tag, kw, pattern, iterable, body)
            first_init = None
            if self.eat_punct("="):
                first_init = self.parse_assignment()
            decls = [(pattern, first_init)]
            while self.eat_punct(","):
                pat = self.parse_pattern()
                iv = None
                if self.eat_punct("="):
                    iv = self.parse_assignment()
                decls.append((pat, iv))
            init = ("vardecl", kw, decls)
        else:
            init = ("exprstmt", self.parse_expression())
        self.expect_punct(";")
        cond = None
        if not self.at_punct(";"):
            cond = self.parse_expression()
        self.expect_punct(";")
        update = None
        if not self.at_punct(")"):
            update = self.parse_expression()
        self.expect_punct(")")
        body = self.parse_statement()
        return ("for", init, cond, update, body)

    def parse_while(self):
        self.next()
        self.expect_punct("(")
        cond = self.parse_expression()
        self.expect_punct(")")
        body = self.parse_statement()
        return ("while", cond, body)

    def parse_do_while(self):
        self.next()
        body = self.parse_statement()
        if not self.at_kw("while"):
            raise JSError("expected 'while' after do-body")
        self.next()
        self.expect_punct("(")
        cond = self.parse_expression()
        self.expect_punct(")")
        self.semi()
        return ("dowhile", body, cond)

    def parse_try(self):
        self.next()  # try
        body = self.parse_block()
        catch_param = None
        catch_body = None
        finally_body = None
        if self.at_kw("catch"):
            self.next()
            if self.eat_punct("("):
                if not self.at_punct(")"):
                    catch_param = self.parse_pattern()
                self.expect_punct(")")
            catch_body = self.parse_block()
        if self.at_kw("finally"):
            self.next()
            finally_body = self.parse_block()
        if catch_body is None and finally_body is None:
            raise JSError("try requires a catch or finally block")
        return ("try", body, catch_param, catch_body, finally_body)

    def parse_switch(self):
        self.next()  # 'switch'
        self.expect_punct("(")
        disc = self.parse_expression()
        self.expect_punct(")")
        self.expect_punct("{")
        clauses = []
        def _at_case():
            tk = self.peek()
            return tk.kind in ("kw", "ident") and tk.value == "case"

        def _at_default():
            tk = self.peek()
            return tk.kind in ("kw", "ident") and tk.value == "default"

        while not self.at_punct("}"):
            if self.peek().kind == "eof":
                raise JSError("unterminated switch block")
            if _at_case():
                self.next()
                test = self.parse_expression()
                self.expect_punct(":")
            elif _at_default():
                self.next()
                test = None
                self.expect_punct(":")
            else:
                raise JSError(f"line {self.peek().line}: expected 'case' or 'default' in switch")
            stmts = []
            while not self.at_punct("}") and not _at_case() and not _at_default():
                if self.peek().kind == "eof":
                    raise JSError("unterminated switch block")
                st = self.parse_statement()
                if st is not None:
                    stmts.append(st)
            clauses.append((test, stmts))
        self.expect_punct("}")
        return ("switch", disc, clauses)

    # -- classes --------------------------------------------------------------
    def parse_class_decl(self):
        cls = self.parse_class()
        if cls[1] is None:
            t = self.peek()
            raise JSError(f"line {t.line}: class declaration requires a name")
        self.semi()
        return ("classdecl", cls)

    def parse_class(self):
        # Current token is 'class'.  Returns
        # ("class", name|None, super_expr|None, members, fields) where
        # members = [(kind, name, params, body, is_static)] with kind in
        # "constructor"/"method"/"get"/"set", and fields =
        # [(name, init_node|None, is_static)].
        self.next()  # class
        name = None
        if self.peek().kind == "ident":
            name = self.next().value
        super_expr = None
        t = self.peek()
        if t.kind in ("ident", "kw") and t.value == "extends":
            self.next()
            super_expr = self.parse_class_super()
        self.expect_punct("{")
        members = []
        fields = []
        while not self.at_punct("}"):
            if self.peek().kind == "eof":
                raise JSError("unexpected end of input inside class body")
            if self.eat_punct(";"):
                continue
            is_static = False
            t = self.peek()
            t2 = self.peek(1)
            if t.kind == "ident" and t.value == "static" and \
                    not (t2.kind == "punct" and t2.value == "("):
                if t2.kind == "punct" and t2.value == "{":
                    raise JSError(
                        f"line {t.line}: static blocks are not supported")
                self.next()
                is_static = True
                t = self.peek()
                t2 = self.peek(1)
            if self.at_punct("*"):
                raise JSError(
                    f"line {self.peek().line}: generator methods "
                    "are not supported")
            kind = "method"
            if t.kind == "ident" and t.value in ("get", "set") and \
                    t2.kind in ("ident", "kw", "str") and \
                    self.peek(2).kind == "punct" and self.peek(2).value == "(":
                kind = t.value
                self.next()
                t = self.peek()
            if t.kind == "ident" and t.value == "async" and \
                    t2.kind in ("ident", "kw"):
                # async methods run synchronously in this subset.
                self.next()
                t = self.peek()
            name_tok = self.next()
            if name_tok.kind in ("ident", "kw"):
                mname = name_tok.value
            elif name_tok.kind == "str":
                mname = name_tok.value
            elif name_tok.kind == "num":
                mname = _num_to_str(name_tok.value)
            else:
                raise JSError(
                    f"line {name_tok.line}: bad class member "
                    f"{name_tok.value!r}")
            if self.at_punct("("):
                params, body = self.parse_function_rest()
                if mname == "constructor" and not is_static:
                    kind = "constructor"
                members.append((kind, mname, params, body, is_static))
            elif kind == "method" and \
                    (self.at_punct("=") or self.at_punct(";") or
                     self.at_punct("}")):
                init = None
                if self.eat_punct("="):
                    init = self.parse_assignment()
                fields.append((mname, init, is_static))
                self.eat_punct(";")
            else:
                raise JSError(
                    f"line {name_tok.line}: unsupported class member "
                    f"{mname!r}")
        self.expect_punct("}")
        return ("class", name, super_expr, members, fields)

    def parse_class_super(self):
        # The extends expression: a primary plus member/index chain
        # (no calls), e.g. THREE.Object3D or ns.Base.
        expr = self.parse_primary()
        while True:
            if self.at_punct("."):
                self.next()
                nm = self.next()
                if nm.kind not in ("ident", "kw"):
                    raise JSError(
                        f"line {nm.line}: expected property name")
                expr = ("member", expr, nm.value)
            elif self.at_punct("["):
                self.next()
                idx = self.parse_expression()
                self.expect_punct("]")
                expr = ("index", expr, idx)
            else:
                return expr

    # -- expressions ---------------------------------------------------------
    def parse_expression(self):
        return self.parse_assignment()

    def parse_assignment(self):
        left = self.parse_conditional()
        t = self.peek()
        if t.kind == "punct" and t.value in (
                "=", "+=", "-=", "*=", "/=", "%=", "**=",
                "&=", "|=", "^=", "<<=", ">>=", ">>>="):
            op = self.next().value
            if op == "=" and left[0] in ("array", "object"):
                # Destructuring assignment: [a, b] = v / ({a} = v)
                pattern = _literal_to_pattern(left)
                right = self.parse_assignment()
                return ("destruct", pattern, right)
            if left[0] not in ("ident", "member", "index"):
                raise JSError(f"line {t.line}: invalid assignment target")
            right = self.parse_assignment()
            return ("assign", op, left, right)
        return left

    def parse_conditional(self):
        expr = self.parse_logical_or()
        if self.at_punct("?"):
            self.next()
            yes = self.parse_assignment()
            self.expect_punct(":")
            no = self.parse_assignment()
            return ("cond", expr, yes, no)
        return expr

    def _binary(self, sub, ops):
        expr = sub()
        while True:
            t = self.peek()
            if t.kind == "punct" and t.value in ops:
                op = self.next().value
                right = sub()
                expr = ("binary", op, expr, right)
            else:
                return expr

    def parse_logical_or(self):
        return self._binary(self.parse_logical_and, ("||",))

    def parse_logical_and(self):
        return self._binary(self.parse_bitwise_or, ("&&",))

    def parse_bitwise_or(self):
        return self._binary(self.parse_bitwise_xor, ("|",))

    def parse_bitwise_xor(self):
        return self._binary(self.parse_bitwise_and, ("^",))

    def parse_bitwise_and(self):
        return self._binary(self.parse_equality, ("&",))

    def parse_equality(self):
        return self._binary(self.parse_relational, ("==", "!=", "===", "!=="))

    def parse_relational(self):
        expr = self.parse_shift()
        while True:
            t = self.peek()
            if t.kind == "punct" and t.value in ("<", ">", "<=", ">="):
                op = self.next().value
                right = self.parse_shift()
                expr = ("binary", op, expr, right)
            elif t.kind == "kw" and t.value == "instanceof":
                self.next()
                right = self.parse_shift()
                expr = ("binary", "instanceof", expr, right)
            else:
                return expr

    def parse_shift(self):
        return self._binary(self.parse_additive, ("<<", ">>", ">>>"))

    def parse_additive(self):
        return self._binary(self.parse_multiplicative, ("+", "-"))

    def parse_multiplicative(self):
        return self._binary(self.parse_unary, ("*", "/", "%", "**"))

    def parse_unary(self):
        t = self.peek()
        if t.kind == "punct" and t.value in ("-", "!", "+", "~"):
            op = self.next().value
            return ("unary", op, self.parse_unary())
        if t.kind == "punct" and t.value in ("++", "--"):
            op = self.next().value
            operand = self.parse_unary()
            return ("update", op, operand, True)
        if t.kind == "kw" and t.value in ("typeof", "void", "delete"):
            op = self.next().value
            return ("unary", op, self.parse_unary())
        if t.kind == "ident" and t.value == "await":
            self.next()
            return ("await", self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_call_member()
        t = self.peek()
        if t.kind == "punct" and t.value in ("++", "--"):
            op = self.next().value
            return ("update", op, expr, False)
        return expr

    def parse_call_member(self):
        expr = self.parse_primary()
        while True:
            if self.at_punct("."):
                self.next()
                name_tok = self.next()
                if name_tok.kind not in ("ident", "kw"):
                    raise JSError(f"line {name_tok.line}: expected property name")
                expr = ("member", expr, name_tok.value)
            elif self.at_punct("["):
                self.next()
                idx = self.parse_expression()
                self.expect_punct("]")
                expr = ("index", expr, idx)
            elif self.at_punct("("):
                args = self.parse_args()
                expr = ("call", expr, args)
            else:
                return expr

    def parse_args(self):
        self.expect_punct("(")
        args = []
        if not self.at_punct(")"):
            while True:
                if self.at_punct("..."):
                    self.next()
                    args.append(("spread", self.parse_assignment()))
                else:
                    args.append(self.parse_assignment())
                if not self.eat_punct(","):
                    break
                if self.at_punct(")"):
                    break  # trailing comma (ES2017)
        self.expect_punct(")")
        return args

    def parse_primary(self):
        t = self.next()
        if t.kind == "num":
            return ("num", t.value)
        if t.kind == "regex":
            return ("regex", t.value[0], t.value[1])
        if t.kind == "str":
            return ("str", t.value)
        if t.kind == "tmpl":
            parts = []
            for kind, val in t.value:
                if kind == "lit":
                    parts.append(("str", val))
                else:
                    sub = Parser(_lex(val))
                    expr = sub.parse_expression()
                    tail = sub.peek()
                    if tail.kind != "eof":
                        raise JSError(
                            f"line {t.line}: bad template expression "
                            f"{val!r}")
                    parts.append(expr)
            return ("tmpl", parts)
        if t.kind == "ident":
            if t.value == "async":
                nt = self.peek()
                if nt.kind == "kw" and nt.value == "function":
                    self.next()
                    name = None
                    if self.peek().kind == "ident":
                        name = self.next().value
                    params, body = self.parse_function_rest()
                    return ("funcexpr", name, params, body)
                if nt.kind == "punct" and nt.value == "(":
                    save = self.pos
                    self.next()
                    if self._scan_arrow_params():
                        params = self.parse_param_list()
                        self.expect_punct("=>")
                        return self.parse_arrow_body(params)
                    self.pos = save
                if nt.kind == "ident" and self.peek(1).kind == "punct" and \
                        self.peek(1).value == "=>":
                    pname = self.next().value
                    self.next()
                    return self.parse_arrow_body([(("pident", pname), None)])
            if t.value == "super":
                return ("super",)
            # single-parameter arrow without parentheses: x => ...
            if self.at_punct("=>"):
                self.next()
                return self.parse_arrow_body([(("pident", t.value), None)])
            return ("ident", t.value)
        if t.kind == "ident" and t.value == "async" and \
                self.peek(1).kind == "kw" and self.peek(1).value == "function":
            self.next()
            self.next()
            name = self.expect_ident()
            params, body = self.parse_function_rest()
            return ("funcdecl", name, params, body)
        if t.kind == "kw":
            w = t.value
            if w == "true":
                return ("bool", True)
            if w == "false":
                return ("bool", False)
            if w in ("null", "undefined"):
                return ("undef",)
            if w == "this":
                return ("this",)
            if w == "class":
                self.pos -= 1  # parse_class expects to consume 'class'
                return ("classexpr", self.parse_class())
            if w == "function":
                name = None
                if self.peek().kind == "ident":
                    name = self.next().value
                params, body = self.parse_function_rest()
                return ("funcexpr", name, params, body)
            if w == "new":
                # new Callee(args) -- callee may be member/index chain
                callee = self.parse_primary()
                while True:
                    if self.at_punct("."):
                        self.next()
                        nm = self.next()
                        callee = ("member", callee, nm.value)
                    elif self.at_punct("["):
                        self.next()
                        idx = self.parse_expression()
                        self.expect_punct("]")
                        callee = ("index", callee, idx)
                    else:
                        break
                args = []
                if self.at_punct("("):
                    args = self.parse_args()
                return ("new", callee, args)
            raise JSError(f"line {t.line}: unexpected keyword {w!r}")
        if t.kind == "punct":
            if t.value == "(":
                # arrow function parameter list, or grouping
                save = self.pos
                if self._scan_arrow_params():
                    params = self.parse_param_list()
                    self.expect_punct("=>")
                    return self.parse_arrow_body(params)
                self.pos = save
                expr = self.parse_expression()
                self.expect_punct(")")
                return expr
            if t.value == "{":
                return self.parse_object_literal()
            if t.value == "[":
                return self.parse_array_literal()
            if t.value == "/":
                raise JSError(f"line {t.line}: regex literals not supported")
        raise JSError(f"line {t.line}: unexpected token {t.value!r}")

    def _scan_arrow_params(self):
        # Current position is just after '('.  Token-level lookahead:
        # scan to the matching ')' (tracking (), [] and {} depth) and
        # report whether '=>' follows it.  Does not move self.pos.
        toks = self.toks
        p = self.pos
        depth = 0
        while p < len(toks):
            tk = toks[p]
            if tk.kind == "eof":
                return False
            if tk.kind == "punct":
                if tk.value in ("(", "[", "{"):
                    depth += 1
                elif tk.value in (")", "]", "}"):
                    if depth == 0:
                        if tk.value != ")":
                            return False
                        q = p + 1
                        return q < len(toks) and toks[q].kind == "punct" \
                            and toks[q].value == "=>"
                    depth -= 1
            p += 1
        return False

    def parse_arrow_body(self, params):
        if self.at_punct("{"):
            block = self.parse_block()
            return ("funcexpr", None, params, block[1])
        expr = self.parse_assignment()
        return ("arrowexpr", params, expr)

    def parse_object_literal(self):
        pairs = []
        if not self.at_punct("}"):
            while True:
                if self.at_punct("..."):
                    self.next()
                    pairs.append((None, ("spread", self.parse_assignment())))
                else:
                    kt = self.next()
                    if kt.kind in ("ident", "kw"):
                        key = kt.value
                    elif kt.kind == "str":
                        key = kt.value
                    elif kt.kind == "num":
                        key = _num_to_str(kt.value)
                    else:
                        raise JSError(f"line {kt.line}: bad object key {kt.value!r}")
                    if self.eat_punct(":"):
                        val = self.parse_assignment()
                    else:
                        # shorthand {x} means {x: x}
                        val = ("ident", key)
                    pairs.append((key, val))
                if not self.eat_punct(","):
                    break
                if self.at_punct("}"):
                    break
        self.expect_punct("}")
        return ("object", pairs)

    def parse_array_literal(self):
        elems = []
        if not self.at_punct("]"):
            while True:
                if self.at_punct(","):
                    elems.append(("undef",))
                elif self.at_punct("..."):
                    self.next()
                    elems.append(("spread", self.parse_assignment()))
                else:
                    elems.append(self.parse_assignment())
                if not self.eat_punct(","):
                    break
                if self.at_punct("]"):
                    break
        self.expect_punct("]")
        return ("array", elems)


def _literal_to_pattern(node):
    """Convert an array/object literal AST into a destructuring
    pattern (used for assignment destructuring: [a, b] = v)."""
    if node[0] == "array":
        elems = []
        raw = node[1]
        for i, el in enumerate(raw):
            if el[0] == "undef":
                elems.append(None)
            elif el[0] == "spread":
                if i != len(raw) - 1 or el[1][0] != "ident":
                    raise JSError("invalid rest element in pattern")
                elems.append(("prest", el[1][1]))
            elif el[0] == "assign" and el[1] == "=":
                elems.append((_literal_to_pattern_target(el[2]), el[3]))
            else:
                elems.append((_literal_to_pattern_target(el), None))
        return ("parr", elems)
    if node[0] == "object":
        props = []
        rest = None
        for key, val in node[1]:
            if key is None:  # {...rest} spread entry
                if val[0] != "spread" or val[1][0] != "ident":
                    raise JSError("invalid rest element in pattern")
                rest = val[1][1]
                continue
            if val[0] == "assign" and val[1] == "=":
                props.append((key, _literal_to_pattern_target(val[2]),
                              val[3]))
            else:
                props.append((key, _literal_to_pattern_target(val), None))
        return ("pobj", props, rest)
    raise JSError("invalid destructuring target")


def _literal_to_pattern_target(node):
    if node[0] == "ident":
        return ("pident", node[1])
    if node[0] in ("array", "object"):
        return _literal_to_pattern(node)
    if node[0] in ("member", "index"):
        return ("pmember", node)
    raise JSError("invalid destructuring target")


# --------------------------------------------------------------------------
# Runtime values
# --------------------------------------------------------------------------

class JSObject:
    """A plain JS object: attribute access is backed by a dict."""

    def __init__(self, props=None):
        object.__setattr__(self, "_props", dict(props or {}))

    def __getattr__(self, name):
        props = object.__getattribute__(self, "_props")
        if name in props:
            return props[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_props")[name] = value

    def __repr__(self):
        return f"JSObject({object.__getattribute__(self, '_props')!r})"


class _JSURL:
    """The URL constructor, modelled just far enough for page code
    that reads query parameters off ``document.location``
    (``new URL(document.location).searchParams.has('vr')``) or mints
    object URLs for downloads (``URL.createObjectURL`` — the blob
    itself never leaves the host here, so a stub string is honest).
    Parsing is by hand: urllib is outside the host's import
    whitelist."""

    def __init__(self, url="", base=None):
        if isinstance(url, str):
            href = url
        elif isinstance(url, dict):
            href = str(url.get("href", ""))
        else:
            href = str(getattr(url, "href", "") or "")
        rest = href
        hash_ = ""
        if "#" in rest:
            rest, frag = rest.split("#", 1)
            hash_ = "#" + frag
        search = ""
        if "?" in rest:
            rest, query = rest.split("?", 1)
            search = "?" + query
        protocol = ""
        host = ""
        pathname = rest
        if "://" in rest:
            scheme, after = rest.split("://", 1)
            protocol = scheme + ":"
            if "/" in after:
                host, path = after.split("/", 1)
                pathname = "/" + path
            else:
                host, pathname = after, "/"
        self.href = href
        self.protocol = protocol
        self.host = host
        self.hostname = host.split(":")[0] if host else ""
        self.port = (host.split(":", 1)[1]
                     if ":" in host else "")
        self.pathname = pathname
        self.search = search
        self.hash = hash_
        self.origin = (protocol + "//" + host) if protocol else "null"
        params = {}
        if search:
            for pair in search[1:].split("&"):
                if not pair:
                    continue
                if "=" in pair:
                    key, value = pair.split("=", 1)
                else:
                    key, value = pair, ""
                params.setdefault(key, []).append(value)
        self.searchParams = JSObject({
            "get": lambda name=None: (
                params.get(str(name), [None])[0]),
            "has": lambda name=None: str(name) in params,
            "getAll": lambda name=None: list(
                params.get(str(name), [])),
            "toString": lambda: search[1:],
        })

    @staticmethod
    def createObjectURL(blob=None):
        return "blob:host-stub"

    @staticmethod
    def revokeObjectURL(url=None):
        return None


def _map_key(key):
    """Normalise a JS Map/Set key to a hashable, type-tagged tuple
    (JS distinguishes 1 from '1' and from true; object keys compare
    by identity)."""
    if isinstance(key, bool):
        return ("b", key)
    if isinstance(key, (int, float)):
        return ("n", float(key))
    if isinstance(key, str):
        return ("s", key)
    if key is None:
        return ("null",)
    if key is UNDEFINED:
        return ("undef",)
    return ("o", id(key))


class JSMap:
    """The JS ``Map`` builtin: insertion-ordered key/value store.
    Iteration helpers return plain lists (the interpreter's for-of
    works over lists); ``forEach`` invokes the callback directly,
    which works for JS functions (they are Python-callable)."""

    def __init__(self, iterable=None):
        self._entries = []  # list of [key, value]
        self._index = {}    # _map_key -> position in _entries
        if iterable is not None:
            for pair in iterable:
                self.set(pair[0], pair[1])

    @property
    def size(self):
        return len(self._entries)

    def set(self, key, value):
        norm = _map_key(key)
        pos = self._index.get(norm)
        if pos is None:
            self._index[norm] = len(self._entries)
            self._entries.append([key, value])
        else:
            self._entries[pos][1] = value
        return self

    def get(self, key):
        pos = self._index.get(_map_key(key))
        if pos is None:
            return UNDEFINED
        return self._entries[pos][1]

    def has(self, key):
        return _map_key(key) in self._index

    def delete(self, key):
        norm = _map_key(key)
        pos = self._index.get(norm)
        if pos is None:
            return False
        del self._entries[pos]
        self._index = {_map_key(k): i
                       for i, (k, _v) in enumerate(self._entries)}
        return True

    def clear(self):
        self._entries = []
        self._index = {}
        return None

    def forEach(self, callback, this_arg=UNDEFINED):
        for key, value in list(self._entries):
            callback(value, key, self)
        return None

    def keys(self):
        return [k for k, _v in self._entries]

    def values(self):
        return [v for _k, v in self._entries]

    def entries(self):
        return [[k, v] for k, v in self._entries]


class JSSet:
    """The JS ``Set`` builtin (see JSMap)."""

    def __init__(self, iterable=None):
        self._entries = []  # list of values
        self._index = {}
        if iterable is not None:
            for value in iterable:
                self.add(value)

    @property
    def size(self):
        return len(self._entries)

    def add(self, value):
        norm = _map_key(value)
        if norm not in self._index:
            self._index[norm] = len(self._entries)
            self._entries.append(value)
        return self

    def has(self, value):
        return _map_key(value) in self._index

    def delete(self, value):
        norm = _map_key(value)
        pos = self._index.get(norm)
        if pos is None:
            return False
        del self._entries[pos]
        self._index = {_map_key(v): i
                       for i, v in enumerate(self._entries)}
        return True

    def clear(self):
        self._entries = []
        self._index = {}
        return None

    def forEach(self, callback, this_arg=UNDEFINED):
        for value in list(self._entries):
            callback(value, value, self)
        return None

    def keys(self):
        return list(self._entries)

    def values(self):
        return list(self._entries)

    def entries(self):
        return [[v, v] for v in self._entries]

    def __iter__(self):
        return iter(self._entries)


class JSPromise:
    """Synchronous Promise implementation for async/await and Promise.all."""

    def __init__(self, executor=None):
        self._state = "pending"
        self._value = UNDEFINED
        if callable(executor):
            def _resolve(val=UNDEFINED):
                if self._state != "pending":
                    return
                if isinstance(val, JSPromise) or (
                    hasattr(val, "_state") and hasattr(val, "_value")
                ):
                    self._state = val._state
                    self._value = val._value
                else:
                    self._state = "fulfilled"
                    self._value = val

            def _reject(err=UNDEFINED):
                if self._state != "pending":
                    return
                self._state = "rejected"
                self._value = err

            try:
                executor(_resolve, _reject)
            except Exception as exc:
                _reject(exc)
        if self._state == "pending":
            self._state = "fulfilled"

    @classmethod
    def resolve(cls, val=UNDEFINED):
        if isinstance(val, JSPromise) or (
            hasattr(val, "_state") and hasattr(val, "_value")
        ):
            return val
        p = cls()
        p._state = "fulfilled"
        p._value = val
        return p

    @classmethod
    def reject(cls, err=UNDEFINED):
        p = cls()
        p._state = "rejected"
        p._value = err
        return p

    @classmethod
    def all(cls, items):
        out = []
        for it in (items or []):
            if hasattr(it, "_state") and hasattr(it, "_value"):
                if it._state == "rejected":
                    return cls.reject(it._value)
                out.append(it._value)
            else:
                out.append(it)
        return cls.resolve(out)

    @classmethod
    def allSettled(cls, items):
        out = []
        for it in (items or []):
            if hasattr(it, "_state") and hasattr(it, "_value"):
                if it._state == "rejected":
                    out.append(JSObject({"status": "rejected", "reason": it._value}))
                else:
                    out.append(JSObject({"status": "fulfilled", "value": it._value}))
            else:
                out.append(JSObject({"status": "fulfilled", "value": it}))
        return cls.resolve(out)

    @classmethod
    def race(cls, items):
        for it in (items or []):
            return cls.resolve(it)
        return cls.resolve(UNDEFINED)

    def then(self, on_fulfilled=None, on_rejected=None):
        if self._state == "fulfilled":
            if not callable(on_fulfilled):
                return JSPromise.resolve(self._value)
            try:
                return JSPromise.resolve(on_fulfilled(self._value))
            except Exception as exc:
                return JSPromise.reject(exc)
        if not callable(on_rejected):
            return JSPromise.reject(self._value)
        try:
            return JSPromise.resolve(on_rejected(self._value))
        except Exception as exc:
            return JSPromise.reject(exc)

    def catch_(self, on_rejected=None):
        return self.then(None, on_rejected)

    def finally_(self, callback=None):
        if callable(callback):
            try:
                callback()
            except Exception as exc:
                return JSPromise.reject(exc)
        return JSPromise.reject(self._value) if self._state == "rejected" else JSPromise.resolve(self._value)


JSPromise.catch = JSPromise.catch_
setattr(JSPromise, "finally", JSPromise.finally_)


class JSRegExp:
    """Minimal RegExp wrapper backed by Python's re module."""

    def __init__(self, pattern="", flags=""):
        self.source = str(pattern)
        self.flags = str(flags)
        self.lastIndex = 0
        py_flags = 0
        if "i" in self.flags:
            py_flags |= re.IGNORECASE
        if "m" in self.flags:
            py_flags |= re.MULTILINE
        if "s" in self.flags:
            py_flags |= re.DOTALL
        try:
            self._re = re.compile(self.source, py_flags)
        except re.error as exc:
            raise JSError(f"invalid RegExp /{self.source}/{self.flags}: {exc}")

    def test(self, s=""):
        return self._re.search(_to_str(s)) is not None

    def exec(self, s=""):
        m = self._re.search(_to_str(s))
        if m is None:
            return None
        return [m.group(0)] + list(m.groups())


def _js_parse_int(value, radix=0):
    s = _to_str(value).lstrip()
    sign = 1.0
    if s[:1] in ("+", "-"):
        if s[0] == "-":
            sign = -1.0
        s = s[1:]
    try:
        radix = int(radix)
    except Exception:
        radix = 0
    if radix == 0:
        if s[:2].lower() == "0x":
            radix = 16
            s = s[2:]
        else:
            radix = 10
    elif radix == 16 and s[:2].lower() == "0x":
        s = s[2:]
    if radix < 2 or radix > 36:
        return float("nan")
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    valid = digits[:radix]
    out = ""
    for ch in s:
        if ch.lower() in valid:
            out += ch
        else:
            break
    if not out:
        return float("nan")
    return sign * float(int(out, radix))


def _js_parse_float(value):
    s = _to_str(value).lstrip()
    if s.startswith("Infinity") or s.startswith("+Infinity"):
        return float("inf")
    if s.startswith("-Infinity"):
        return float("-inf")
    i = 0
    seen_digit = False
    seen_dot = False
    while i < len(s):
        ch = s[i]
        if ch.isdigit():
            seen_digit = True
            i += 1
        elif ch == "." and not seen_dot:
            seen_dot = True
            i += 1
        elif ch in "+-" and i == 0:
            i += 1
        else:
            break
    if not seen_digit:
        return float("nan")
    if i < len(s) and s[i] in "eE":
        j = i + 1
        if j < len(s) and s[j] in "+-":
            j += 1
        k = j
        while k < len(s) and s[k].isdigit():
            k += 1
        if k > j:
            i = k
    try:
        return float(s[:i])
    except ValueError:
        return float("nan")


_URI_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789-_.!~*'()")
_URI_RESERVED = frozenset(";,/?:@&=+$#")


def _encode_uri(text, extra_allowed):
    allowed = _URI_UNRESERVED | extra_allowed
    out = []
    for ch in _to_str(text):
        if ch in allowed:
            out.append(ch)
        else:
            for byte in ch.encode("utf-8"):
                out.append("%%%02X" % byte)
    return "".join(out)


def _decode_uri(text, keep_reserved):
    s = _to_str(text)
    raw = bytearray()
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "%" and i + 2 < len(s):
            hexpart = s[i + 1:i + 3]
            try:
                byte = int(hexpart, 16)
            except ValueError:
                byte = None
            if byte is not None and len(hexpart) == 2:
                if keep_reserved and chr(byte) in _URI_RESERVED:
                    raw.extend(ch.encode("utf-8"))
                    i += 1
                    continue
                raw.append(byte)
                i += 3
                continue
        raw.extend(ch.encode("utf-8"))
        i += 1
    return raw.decode("utf-8", "replace")


class Env:
    __slots__ = ("vars", "parent")

    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def declare(self, name, value):
        self.vars[name] = value

    def lookup(self, name):
        env = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.parent
        raise JSError(f"{name} is not defined")

    def assign(self, name, value):
        env = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        # Sloppy-mode style: assigning an unknown name creates a global.
        root = self
        while root.parent is not None:
            root = root.parent
        root.vars[name] = value


class JSFunction:
    def __init__(self, interp, name, params, body, closure, expr_body=None,
                 home_class=None, is_constructor=False):
        self.interp = interp
        self.name = name
        # params: list of (pattern, default_node|None); a rest parameter
        # has the pattern ("prest", name).
        self.params = params
        self.body = body            # list of statements, or None
        self.closure = closure
        self.expr_body = expr_body  # AST for expression-bodied arrows
        # The class this function was defined in as a method (drives
        # super resolution), or None for plain functions.
        self.home_class = home_class
        self.is_constructor = is_constructor
        self.bound_this = None      # set on bound method copies

    def bind(self, this_value, *bound_args):
        """A copy of this function with ``this`` fixed (method
        extraction: const f = obj.m; f())."""
        clone = JSFunction(self.interp, self.name, self.params, self.body,
                           self.closure, self.expr_body,
                           home_class=self.home_class,
                           is_constructor=self.is_constructor)
        clone.bound_this = this_value
        return clone

    def call(self, this_value=UNDEFINED, *args):
        return self.interp._invoke(self, list(args), this_value=this_value)

    def apply(self, this_value=UNDEFINED, args=None):
        arg_list = list(args) if args is not None else []
        return self.interp._invoke(self, arg_list, this_value=this_value)

    def __call__(self, *args, **kwargs):
        return self.interp._invoke(self, list(args), this_value=UNDEFINED)

    def __repr__(self):
        return f"<JSFunction {self.name or '(anonymous)'}>"


class JSBuiltinMethod:
    """A built-in method (Array/String prototype surface).  Wrapped
    so _call_value passes arguments through unconverted (unlike plain
    host callables, whose arguments go through _to_python)."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)

    def __repr__(self):
        return "<JSBuiltinMethod>"


class JSClass:
    """A JS ``class``: constructor, prototype methods, statics,
    getters/setters and field initializers, with single inheritance.
    The superclass is another JSClass, a Python class (e.g. a three.py
    class), or None."""

    def __init__(self, interp, name, superclass, ctor, methods, statics,
                 getters, setters, fields, static_fields):
        self.interp = interp
        self.name = name
        self.superclass = superclass
        self.ctor = ctor                    # JSFunction or None
        self.methods = methods              # name -> JSFunction
        self.statics = statics              # name -> JSFunction
        self.getters = getters              # name -> JSFunction
        self.setters = setters              # name -> JSFunction
        self.fields = fields                # [(name, init_node|None)]
        self.static_fields = static_fields  # name -> value (evaluated)
        self.static_props = {}              # assigned later from JS
        self.static_getters = {}            # name -> JSFunction
        self.static_setters = {}            # name -> JSFunction
        self.def_env = None                 # env the class was defined in

    def chain(self):
        cls = self
        while isinstance(cls, JSClass):
            yield cls
            cls = cls.superclass

    def lookup(self, table, name):
        for cls in self.chain():
            fn = getattr(cls, table).get(name)
            if fn is not None:
                return fn
        return None

    def has_member(self, name):
        for cls in self.chain():
            if name in cls.methods or name in cls.getters or \
                    name in cls.setters:
                return True
        return False

    def __repr__(self):
        return f"<JSClass {self.name or '(anonymous)'}>"


class JSInstance:
    """An instance of a JSClass.  Own properties live in ``_props``;
    member lookup falls through to the class chain (methods bound,
    getters invoked) and finally to ``_base`` -- the wrapped Python
    object when the class extends a Python class, so renderer-style
    duck typing (getattr of position/children/...) keeps working."""

    def __init__(self, cls, interp):
        object.__setattr__(self, "_props", {})
        object.__setattr__(self, "_class", cls)
        object.__setattr__(self, "_interp", interp)
        object.__setattr__(self, "_base", None)

    def __getattr__(self, name):
        if name in ("_props", "_class", "_interp", "_base") or (
            name.startswith("__") and name.endswith("__")
        ):
            raise AttributeError(name)
        props = object.__getattribute__(self, "_props")
        if name in props:
            return props[name]
        cls = object.__getattribute__(self, "_class")
        interp = object.__getattribute__(self, "_interp")
        getter = cls.lookup("getters", name)
        if getter is not None:
            return interp._invoke(getter, [], this_value=self)
        method = cls.lookup("methods", name)
        if method is not None:
            return method.bind(self)
        base = object.__getattribute__(self, "_base")
        if base is not None:
            return getattr(base, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ("_props", "_class", "_interp", "_base"):
            object.__setattr__(self, name, value)
            return
        cls = object.__getattribute__(self, "_class")
        interp = object.__getattribute__(self, "_interp")
        setter = cls.lookup("setters", name)
        if setter is not None:
            interp._invoke(setter, [value], this_value=self)
            return
        props = object.__getattribute__(self, "_props")
        base = object.__getattribute__(self, "_base")
        if base is not None and name not in props and \
                not cls.has_member(name) and hasattr(base, name):
            setattr(base, name, value)
            return
        props[name] = value

    def __repr__(self):
        cls = object.__getattribute__(self, "_class")
        return f"<{cls.name or 'anonymous'} instance>"


def _num_to_str(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "undefined"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        if v == int(v) and abs(v) < 1e21:
            return str(int(v))
        return repr(v)
    return str(v)


def _to_str(v):
    if v is None:
        return "undefined"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _num_to_str(float(v))
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return ",".join(_to_str(x) for x in v)
    return str(v)


def _truthy(v):
    if v is None or v is False:
        return False
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v != 0 and not (isinstance(v, float) and math.isnan(v))
    if isinstance(v, str):
        return len(v) > 0
    return True


def _to_number(v):
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return 0.0
        try:
            return float(s)
        except ValueError:
            try:
                return float(int(s, 16))
            except ValueError:
                return float("nan")
    return float("nan")


def _to_int32(v):
    """JS ToInt32: truncate toward zero, wrap modulo 2**32, sign."""
    n = _to_number(v)
    if math.isnan(n) or math.isinf(n):
        return 0
    i = math.trunc(n) % (1 << 32)
    return i - (1 << 32) if i >= (1 << 31) else i


def _to_uint32(v):
    """JS ToUint32: truncate toward zero, wrap modulo 2**32."""
    n = _to_number(v)
    if math.isnan(n) or math.isinf(n):
        return 0
    return math.trunc(n) % (1 << 32)


def _bitwise_binary(op, left, right):
    """The JS bitwise/shift operators, on ToInt32/ToUint32 values;
    results come back as floats (the interpreter's number type)."""
    if op == "&":
        return float(_to_int32(left) & _to_int32(right))
    if op == "|":
        return float(_to_int32(left) | _to_int32(right))
    if op == "^":
        return float(_to_int32(left) ^ _to_int32(right))
    if op == "<<":
        return float(_to_int32(_to_int32(left) << (_to_uint32(right) & 31)))
    if op == ">>":
        return float(_to_int32(left) >> (_to_uint32(right) & 31))
    if op == ">>>":
        return float(_to_uint32(left) >> (_to_uint32(right) & 31))
    raise JSError(f"unsupported operator {op!r}")


# --------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------

def _float32(v):
    try:
        return struct.unpack("f", struct.pack("f", v))[0]
    except OverflowError:
        return math.inf if v > 0 else -math.inf


def _float16(v):
    try:
        return struct.unpack("e", struct.pack("e", v))[0]
    except OverflowError:
        return math.inf if v > 0 else -math.inf


def _ta_index(v, n):
    """JS relative index (for subarray/slice/fill bounds)."""
    num = _to_number(v)
    i = 0 if math.isnan(num) else int(num)
    return max(0, n + i) if i < 0 else min(i, n)


class JSArrayBuffer:
    """Minimal ArrayBuffer: a bytearray with byteLength and slice."""

    def __init__(self, byte_length=0):
        n = int(byte_length) if isinstance(byte_length, (int, float)) \
            and not isinstance(byte_length, bool) else 0
        self._bytes = bytearray(max(0, n))

    @property
    def byteLength(self):
        return len(self._bytes)

    def slice(self, begin=0, end=None):
        n = len(self._bytes)
        b = _ta_index(begin, n)
        e = n if end is None else _ta_index(end, n)
        out = JSArrayBuffer(0)
        out._bytes = bytearray(self._bytes[b:e])
        return out


class JSTypedArray:
    """JS typed array backed by a stdlib ``array.array`` (the module
    stays numpy-free; ``__array__`` bridges to numpy consumers such
    as three.py's BufferAttribute).  Element writes follow the JS
    conversion rules for the kind (wrap for integers, round-half-even
    + clamp for Uint8Clamped, float32/float16 rounding for the float
    kinds).  subarray/slice return copies rather than shared views,
    and construction from an ArrayBuffer copies the bytes — the same
    copy semantics the port already has for plain JS arrays."""

    _typecode = "f"
    _bpe = 4
    _conv_kind = "f32"
    _bits = 32
    _signed = True
    BYTES_PER_ELEMENT = 4

    def __init__(self, *args):
        self._data = array.array(self._typecode)
        if not args:
            return
        a0 = args[0]
        if isinstance(a0, (int, float)) and not isinstance(a0, bool):
            n = int(a0)
            if n > 0:
                self._data = array.array(
                    self._typecode, bytes(n * self._data.itemsize))
            return
        if isinstance(a0, JSArrayBuffer):
            raw = bytes(a0._bytes)
            offset = int(args[1]) if len(args) > 1 else 0
            if len(args) > 2 and args[2] is not None:
                count = int(args[2])
                raw = raw[offset:offset + count * self._data.itemsize]
            else:
                raw = raw[offset:]
            usable = len(raw) // self._data.itemsize * self._data.itemsize
            self._data.frombytes(raw[:usable])
            return
        if isinstance(a0, JSTypedArray):
            values = list(a0._data)
        else:
            try:
                values = list(a0)
            except TypeError:
                values = []
        self._data = array.array(
            self._typecode, [self._convert(v) for v in values])

    # -- JS conversion on write ----------------------------------------
    def _convert(self, v):
        kind = self._conv_kind
        if kind == "f64":
            return _to_number(v)
        if kind == "f32":
            return _float32(_to_number(v))
        if kind == "f16":
            return _float16(_to_number(v))
        if kind == "clamp":
            n = _to_number(v)
            if math.isnan(n):
                return 0
            # Python's round() is round-half-even, as the JS spec wants.
            return int(round(min(255.0, max(0.0, n))))
        n = _to_number(v)
        if math.isnan(n) or math.isinf(n):
            return 0
        i = math.trunc(n) % (1 << self._bits)
        if self._signed and i >= (1 << (self._bits - 1)):
            i -= 1 << self._bits
        return i

    # -- properties ------------------------------------------------------
    @property
    def length(self):
        return len(self._data)

    @property
    def byteLength(self):
        return len(self._data) * self._bpe

    @property
    def byteOffset(self):
        return 0

    @property
    def buffer(self):
        buf = JSArrayBuffer(0)
        buf._bytes = bytearray(self._data.tobytes())
        return buf

    # -- Python container protocol (the interpreter's index/member
    #    fallbacks route JS element access here) -------------------------
    def __getitem__(self, i):
        if isinstance(i, slice):
            return list(self._data[i])
        return self._data[i]  # IndexError -> interpreter's UNDEFINED

    def __setitem__(self, i, v):
        if isinstance(i, slice):
            self._data[i] = array.array(
                self._typecode, [self._convert(x) for x in v])
            return
        i = int(i)
        if 0 <= i < len(self._data):
            self._data[i] = self._convert(v)
        # Out-of-range writes are silently dropped, as in JS.

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __array__(self, dtype=None, copy=None):
        import numpy as np
        if dtype is None:
            return np.asarray(self._data)
        return np.asarray(self._data, dtype=dtype)

    # -- JS methods --------------------------------------------------------
    def set(self, src, offset=0):
        offset = int(_to_number(offset))
        values = list(src._data) if isinstance(src, JSTypedArray) \
            else list(src)
        for j, v in enumerate(values):
            i = offset + j
            if 0 <= i < len(self._data):
                self._data[i] = self._convert(v)

    def subarray(self, begin=0, end=None):
        n = len(self._data)
        b = _ta_index(begin, n)
        e = n if end is None else _ta_index(end, n)
        out = type(self)()
        out._data = array.array(self._typecode, self._data[b:e])
        return out

    def slice(self, begin=0, end=None):
        return self.subarray(begin, end)

    def fill(self, value, start=0, end=None):
        n = len(self._data)
        b = _ta_index(start, n)
        e = n if end is None else _ta_index(end, n)
        v = self._convert(value)
        for i in range(b, e):
            self._data[i] = v
        return self

    def forEach(self, cb):
        for i, v in enumerate(self._data):
            cb(v, float(i), self)

    def map(self, cb):
        out = type(self)()
        out._data = array.array(self._typecode, [
            self._convert(cb(v, float(i), self))
            for i, v in enumerate(self._data)])
        return out

    def filter(self, cb):
        kept = [v for i, v in enumerate(self._data)
                if cb(v, float(i), self)]
        out = type(self)()
        out._data = array.array(
            self._typecode, [self._convert(v) for v in kept])
        return out

    def indexOf(self, v, from_index=0):
        n = len(self._data)
        target = self._convert(v)
        for i in range(_ta_index(from_index, n), n):
            if self._data[i] == target:
                return float(i)
        return -1.0

    def includes(self, v, from_index=0):
        return self.indexOf(v, from_index) >= 0

    def reverse(self):
        self._data.reverse()
        return self

    def sort(self, cmp=None):
        if cmp is None:
            self._data = array.array(self._typecode, sorted(self._data))
        else:
            from functools import cmp_to_key
            self._data = array.array(self._typecode, sorted(
                self._data,
                key=cmp_to_key(lambda a, b: _to_number(cmp(a, b)))))
        return self

    def join(self, sep=","):
        return str(sep).join(_to_str(v) for v in self._data)

    def toString(self):
        return self.join(",")


def _typed_array_from(cls, src, map_fn=None):
    values = list(src._data) if isinstance(src, JSTypedArray) else list(src)
    if map_fn is not None:
        values = [map_fn(v, float(i)) for i, v in enumerate(values)]
    return cls(values)


def _typed_array_of(cls, *values):
    return cls(list(values))


setattr(JSTypedArray, "from", classmethod(_typed_array_from))
setattr(JSTypedArray, "of", classmethod(_typed_array_of))


class Float32Array(JSTypedArray):
    _typecode, _bpe, _conv_kind = "f", 4, "f32"
    BYTES_PER_ELEMENT = 4


class Float64Array(JSTypedArray):
    _typecode, _bpe, _conv_kind = "d", 8, "f64"
    BYTES_PER_ELEMENT = 8


class Float16Array(JSTypedArray):
    # Stored as float32 with float16 rounding on write (stdlib array
    # has no half typecode); BYTES_PER_ELEMENT reports the JS value.
    _typecode, _bpe, _conv_kind = "f", 2, "f16"
    BYTES_PER_ELEMENT = 2


class Int8Array(JSTypedArray):
    _typecode, _bpe, _conv_kind, _bits, _signed = "b", 1, "int", 8, True
    BYTES_PER_ELEMENT = 1


class Uint8Array(JSTypedArray):
    _typecode, _bpe, _conv_kind, _bits, _signed = "B", 1, "int", 8, False
    BYTES_PER_ELEMENT = 1


class Uint8ClampedArray(JSTypedArray):
    _typecode, _bpe, _conv_kind, _bits, _signed = "B", 1, "clamp", 8, False
    BYTES_PER_ELEMENT = 1


class Int16Array(JSTypedArray):
    _typecode, _bpe, _conv_kind, _bits, _signed = "h", 2, "int", 16, True
    BYTES_PER_ELEMENT = 2


class Uint16Array(JSTypedArray):
    _typecode, _bpe, _conv_kind, _bits, _signed = "H", 2, "int", 16, False
    BYTES_PER_ELEMENT = 2


class Int32Array(JSTypedArray):
    _typecode, _bpe, _conv_kind, _bits, _signed = "i", 4, "int", 32, True
    BYTES_PER_ELEMENT = 4


class Uint32Array(JSTypedArray):
    _typecode, _bpe, _conv_kind, _bits, _signed = "I", 4, "int", 32, False
    BYTES_PER_ELEMENT = 4


def _ndarray_accessor(arr, name):
    """BufferAttribute-style component accessors for a numpy array
    standing in for an attribute (see the ndarray branch of
    JSInterpreter._get_member).  Shaped (N, itemSize) arrays index
    directly; flat arrays assume itemSize 3 when the size divides."""

    def item_size():
        if arr.ndim == 2:
            return arr.shape[1]
        return 3 if arr.size % 3 == 0 else 1

    def get(i, c):
        if arr.ndim == 2:
            return float(arr[int(i), c])
        return float(arr[int(i) * item_size() + c])

    def set_(i, c, v):
        if arr.ndim == 2:
            arr[int(i), c] = v
        else:
            arr[int(i) * item_size() + c] = v

    if name == "getX":
        return lambda i: get(i, 0)
    if name == "getY":
        return lambda i: get(i, 1)
    if name == "getZ":
        return lambda i: get(i, 2)
    if name == "getComponent":
        return lambda i, c: get(i, int(c))
    if name == "setX":
        def _set_x(i, v):
            set_(i, 0, v)
            return arr
        return _set_x
    if name == "setY":
        def _set_y(i, v):
            set_(i, 1, v)
            return arr
        return _set_y
    if name == "setZ":
        def _set_z(i, v):
            set_(i, 2, v)
            return arr
        return _set_z
    if name == "setXYZ":
        def _set_xyz(i, x, y, z):
            set_(i, 0, x)
            set_(i, 1, y)
            set_(i, 2, z)
            return arr
        return _set_xyz
    if name == "copyAt":
        def _copy_at(index1, attribute, index2):
            i1 = int(index1)
            i2 = int(index2)
            sz = item_size()
            if isinstance(attribute, np.ndarray):
                if attribute.ndim == 2:
                    for k in range(min(sz, attribute.shape[1])):
                        set_(i1, k, float(attribute[i2, k]))
                else:
                    for k in range(sz):
                        set_(i1, k, float(attribute[i2 * sz + k]))
            else:
                src = attribute.array
                ssz = int(getattr(attribute, "itemSize", sz))
                for k in range(min(sz, ssz)):
                    set_(i1, k, float(src[i2 * ssz + k]))
            return arr
        return _copy_at

    def _set_component(i, c, v):
        set_(i, int(c), v)
        return arr
    return _set_component


def _parse_date_string(s):
    """Best-effort ISO-8601 parse for the Date constructor / Date.parse;
    NaN when the string is not a recognisable date."""
    txt = s.strip()
    try:
        dt = datetime.datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except ValueError:
        return float("nan")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp() * 1000.0


def _performance_now():
    return time.monotonic() * 1000.0


class JSDate:
    """Minimal JS Date: a millisecond timestamp with the accessors
    sample code actually uses (Date.now(), new Date().getTime(),
    getMilliseconds() and friends).  Getters use UTC so results do
    not depend on the host timezone; getTimezoneOffset reports 0 to
    match.  Registered as a plain Python class so `new Date(...)`
    works through the interpreter's normal constructor path."""

    def __init__(self, *args):
        if not args:
            self._ms = time.time() * 1000.0
        elif len(args) == 1:
            a = args[0]
            if isinstance(a, JSDate):
                self._ms = a._ms
            elif isinstance(a, str):
                self._ms = _parse_date_string(a)
            elif isinstance(a, (int, float)) and not isinstance(a, bool):
                self._ms = float(a)
            else:
                self._ms = time.time() * 1000.0
        else:
            parts = [int(_to_number(a)) for a in args] + [0] * 7
            year, month, day, hh, mm, ss, ms = parts[:7]
            try:
                dt = datetime.datetime(year, month + 1, day or 1, hh, mm,
                                       ss, tzinfo=datetime.timezone.utc)
                self._ms = dt.timestamp() * 1000.0 + ms
            except (ValueError, OverflowError, OSError):
                self._ms = float("nan")

    @staticmethod
    def now():
        return time.time() * 1000.0

    @staticmethod
    def parse(s=None):
        return _parse_date_string(s) if isinstance(s, str) else float("nan")

    def _dt(self):
        return datetime.datetime.fromtimestamp(
            self._ms / 1000.0, tz=datetime.timezone.utc)

    def getTime(self):
        return self._ms

    def valueOf(self):
        return self._ms

    def getMilliseconds(self):
        return float(int(self._ms) % 1000)

    def getSeconds(self):
        return float(self._dt().second)

    def getMinutes(self):
        return float(self._dt().minute)

    def getHours(self):
        return float(self._dt().hour)

    def getDate(self):
        return float(self._dt().day)

    def getDay(self):
        # JS: Sunday = 0 .. Saturday = 6; datetime: Monday = 0.
        return float((self._dt().weekday() + 1) % 7)

    def getMonth(self):
        return float(self._dt().month - 1)

    def getFullYear(self):
        return float(self._dt().year)

    def getYear(self):
        return float(self._dt().year - 1900)

    def getTimezoneOffset(self):
        return 0.0

    def setTime(self, ms):
        self._ms = _to_number(ms)
        return self._ms

    def toISOString(self):
        dt = self._dt()
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + \
            f"{int(self._ms) % 1000:03d}Z"

    def toString(self):
        return self.toISOString()

    def toDateString(self):
        return self._dt().strftime("%a %b %d %Y")


class _StubGradient:
    """Gradient object for the stub 2D context: stops are accepted
    and forgotten (nothing is rasterised through this context)."""

    def addColorStop(self, *a):
        pass


class _StubContext2D:
    """No-op 2D context for stub elements (the getElementById path —
    elements obtained via createElement('canvas') are real
    CanvasElements with a working context).  Drawing through this
    context is lost; it exists so canvas-probing sample code runs."""

    def __init__(self):
        self.fillStyle = "#000000"
        self.strokeStyle = "#000000"
        self.font = "10px sans-serif"
        self.globalAlpha = 1.0
        self.lineWidth = 1.0

    def _noop(self, *a):
        pass

    fillRect = _noop
    clearRect = _noop
    strokeRect = _noop
    drawImage = _noop
    beginPath = _noop
    moveTo = _noop
    lineTo = _noop
    closePath = _noop
    arc = _noop
    fill = _noop
    stroke = _noop
    fillText = _noop
    strokeText = _noop
    putImageData = _noop
    save = _noop
    restore = _noop
    translate = _noop
    rotate = _noop
    scale = _noop
    setTransform = _noop

    def createLinearGradient(self, *a):
        return _StubGradient()

    def createRadialGradient(self, *a):
        return _StubGradient()

    def measureText(self, text=""):
        return JSObject({"width": float(len(str(text))) * 6.0})

    def getImageData(self, x=0, y=0, w=0, h=0):
        w = max(0, int(w))
        h = max(0, int(h))
        return JSObject({
            "width": float(w), "height": float(h),
            "data": Uint8ClampedArray(w * h * 4),
        })


class _SplitInstance:
    """Absorption stub for a split.js instance: the documented
    methods exist as no-ops and any other attribute resolves to a
    no-op callable, so layout calls in sample code run harmlessly."""

    def setSizes(self, *a):
        pass

    def getSizes(self):
        return []

    def collapse(self, *a):
        pass

    def destroy(self, *a):
        pass

    def __getattr__(self, name):
        if name in ("_props", "_class", "_interp", "_base") or (
            name.startswith("__") and name.endswith("__")
        ):
            raise AttributeError(name)
        return lambda *a, **k: None


class _StubElement:
    """document / DOM stub element: every method is a harmless no-op."""

    def __init__(self, tag="div"):
        self.tagName = tag.upper()
        self.children = []
        self.style = JSObject()
        self._context = None
        # Layout is not modelled; every element reports the host
        # viewport size so size/visibility checks in sample code
        # (updateSize, offscreen culling) take the "visible" path.
        self.clientWidth = 256.0
        self.clientHeight = 192.0
        self.offsetWidth = 256.0
        self.offsetHeight = 192.0

    def getBoundingClientRect(self):
        return JSObject({
            "left": 0.0, "top": 0.0,
            "right": self.clientWidth, "bottom": self.clientHeight,
            "width": self.clientWidth, "height": self.clientHeight,
            "x": 0.0, "y": 0.0,
        })

    def getContext(self, kind="2d", *args):
        # Some pages fetch a <canvas> by id (webgl_materials_texture_
        # canvas's drawing canvas); the id path cannot know the tag,
        # so a stub element offers a no-op 2D context.
        if self._context is None:
            self._context = _StubContext2D()
        return self._context

    def appendChild(self, child):
        self.children.append(child)
        return child

    def removeChild(self, child):
        return child

    def setAttribute(self, *a):
        pass

    def getAttribute(self, *a):
        return None

    def addEventListener(self, *a):
        pass

    def querySelector(self, sel=None):
        return _StubElement()

    def querySelectorAll(self, sel=None):
        return []

    def play(self):
        return None

    def pause(self):
        return None


class JSInterpreter:
    def __init__(self, three_module, host=None, html_elements=None):
        self.three_module = three_module
        self.host = host
        self._html_elements = dict(html_elements or {})
        self.raf_queue = []
        self._raf_time = 0.0
        # setTimeout / setInterval registrations: id -> (callback,
        # args, is_interval).  Delays are not modelled (the host runs
        # on frame boundaries); timers fire at task boundaries.
        self._timers = {}
        self._timer_seq = 0
        self.console_log = []
        self._listeners = []  # (target, type, callback) event listeners
        # Class machinery: the home class of the method currently
        # executing (for super), and constructor frames
        # {cls, inst, fields_done} for super() / field-init ordering.
        self._home_stack = []
        self._ctor_stack = []
        self.global_env = Env()
        self._install_globals()

    # -- module resolution -----------------------------------------
    # Addon modules resolved by the interpreter itself: module path ->
    # {exported name: attribute name on the three module}.  Deliberately
    # a small explicit registry, not a general module system.
    _ADDON_MODULES = {
        "three/addons/controls/OrbitControls.js": {
            "OrbitControls": "OrbitControls",
        },
        "three/examples/jsm/controls/OrbitControls.js": {
            "OrbitControls": "OrbitControls",
        },
        # Addon receptacles implemented on the three
        # module (stubs / utils).  The "__default__" entry names the
        # module attribute bound for a default import
        # (import Stats from ... / import GUI from 'lil-gui'); it is
        # not itself an export and is excluded from namespace
        # objects.
        "three/addons/libs/stats.module.js": {
            "Stats": "Stats",
            "__default__": "Stats",
        },
        "three/examples/jsm/libs/stats.module.js": {
            "Stats": "Stats",
            "__default__": "Stats",
        },
        "lil-gui": {
            "GUI": "GUI",
            "__default__": "GUI",
        },
        "three/addons/libs/lil-gui.module.min.js": {
            "GUI": "GUI",
            "__default__": "GUI",
        },
        "three/examples/jsm/libs/lil-gui.module.min.js": {
            "GUI": "GUI",
            "__default__": "GUI",
        },
        "three/addons/utils/BufferGeometryUtils.js": {
            "mergeGeometries": "mergeGeometries",
            "mergeBufferGeometries": "mergeBufferGeometries",
            "mergeAttributes": "mergeAttributes",
            "mergeVertices": "mergeVertices",
        },
        "three/examples/jsm/utils/BufferGeometryUtils.js": {
            "mergeGeometries": "mergeGeometries",
            "mergeBufferGeometries": "mergeBufferGeometries",
            "mergeAttributes": "mergeAttributes",
            "mergeVertices": "mergeVertices",
        },
        # Loaders / controls the manual imports by relative paths
        # into its bundled three tree; the basename fallback below
        # resolves those paths to these entries.
        "three/addons/loaders/GLTFLoader.js": {
            "GLTFLoader": "GLTFLoader",
        },
        "three/addons/loaders/OBJLoader.js": {
            "OBJLoader": "OBJLoader",
        },
        "three/addons/loaders/MTLLoader.js": {
            "MTLLoader": "MTLLoader",
        },
        "three/addons/loaders/HDRLoader.js": {
            "HDRLoader": "HDRLoader",
        },
        "three/examples/jsm/loaders/HDRLoader.js": {
            "HDRLoader": "HDRLoader",
        },
        "three/addons/loaders/RGBELoader.js": {
            "RGBELoader": "RGBELoader",
        },
        "three/examples/jsm/loaders/RGBELoader.js": {
            "RGBELoader": "RGBELoader",
        },
        "three/addons/loaders/PLYLoader.js": {
            "PLYLoader": "PLYLoader",
        },
        "three/examples/jsm/loaders/PLYLoader.js": {
            "PLYLoader": "PLYLoader",
        },
        "three/addons/geometries/ParametricGeometry.js": {
            "ParametricGeometry": "ParametricGeometry",
        },
        "three/examples/jsm/geometries/ParametricGeometry.js": {
            "ParametricGeometry": "ParametricGeometry",
        },
        "three/addons/utils/SkeletonUtils.js": {
            "SkeletonUtils": "SkeletonUtils",
            "clone": "cloneSkeleton",
        },
        "three/examples/jsm/utils/SkeletonUtils.js": {
            "SkeletonUtils": "SkeletonUtils",
            "clone": "cloneSkeleton",
        },
        "three/addons/lights/RectAreaLightUniformsLib.js": {
            "RectAreaLightUniformsLib": "RectAreaLightUniformsLib",
        },
        "three/examples/jsm/lights/RectAreaLightUniformsLib.js": {
            "RectAreaLightUniformsLib": "RectAreaLightUniformsLib",
        },
        "three/addons/controls/TrackballControls.js": {
            "TrackballControls": "TrackballControls",
        },
        "three/addons/webxr/VRButton.js": {
            "VRButton": "VRButton",
        },
        "three/addons/postprocessing/UnrealBloomPass.js": {
            "UnrealBloomPass": "UnrealBloomPass",
        },
        "three/addons/postprocessing/BloomPass.js": {
            "BloomPass": "BloomPass",
        },
        "three/addons/postprocessing/FilmPass.js": {
            "FilmPass": "FilmPass",
        },
        # The minimal post-processing chain (three.py
        # implements EffectComposer / RenderPass / ShaderPass /
        # OutputPass / Pass and the CopyShader dict).
        "three/addons/postprocessing/EffectComposer.js": {
            "EffectComposer": "EffectComposer",
        },
        "three/examples/jsm/postprocessing/EffectComposer.js": {
            "EffectComposer": "EffectComposer",
        },
        "three/addons/postprocessing/RenderPass.js": {
            "RenderPass": "RenderPass",
        },
        "three/examples/jsm/postprocessing/RenderPass.js": {
            "RenderPass": "RenderPass",
        },
        "three/addons/postprocessing/ShaderPass.js": {
            "ShaderPass": "ShaderPass",
        },
        "three/examples/jsm/postprocessing/ShaderPass.js": {
            "ShaderPass": "ShaderPass",
        },
        "three/addons/postprocessing/Pass.js": {
            "Pass": "Pass",
        },
        "three/examples/jsm/postprocessing/Pass.js": {
            "Pass": "Pass",
        },
        "three/addons/postprocessing/OutputPass.js": {
            "OutputPass": "OutputPass",
        },
        "three/examples/jsm/postprocessing/OutputPass.js": {
            "OutputPass": "OutputPass",
        },
        "three/addons/shaders/CopyShader.js": {
            "CopyShader": "CopyShader",
        },
        "three/examples/jsm/shaders/CopyShader.js": {
            "CopyShader": "CopyShader",
        },
        "three/addons/environments/RoomEnvironment.js": {
            "RoomEnvironment": "RoomEnvironment",
        },
        "three/examples/jsm/environments/RoomEnvironment.js": {
            "RoomEnvironment": "RoomEnvironment",
        },
        "three/addons/math/ImprovedNoise.js": {
            "ImprovedNoise": "ImprovedNoise",
        },
        "three/examples/jsm/math/ImprovedNoise.js": {
            "ImprovedNoise": "ImprovedNoise",
        },
        "three/addons/objects/Sky.js": {
            "Sky": "Sky",
        },
        "three/examples/jsm/objects/Sky.js": {
            "Sky": "Sky",
        },
    }

    def _three_export(self, name, source):
        module = self.three_module
        value = getattr(module, name, None) if module is not None else None
        if value is None:
            raise JSError(
                f"module {source!r}: THREE has no export {name!r} "
                "in this port")
        return value

    def _exec_import(self, bindings, source, env):
        if not bindings:
            return
        if source == "three":
            three_ns = None
            for kind, imported, local in bindings:
                if kind in ("namespace", "default"):
                    if three_ns is None:
                        three_ns = self.global_env.lookup("THREE")
                    env.declare(local, three_ns)
                else:
                    env.declare(local, self._three_export(imported, source))
            return
        registry = self._ADDON_MODULES.get(source)
        if registry is None and isinstance(source, str):
            # Basename fallback: page code importing an addon by a
            # relative path into its own bundled three copy (the
            # manual's r132 tree, a CodePen mirror, ...) still names
            # the same addon file.
            base = source.replace("\\", "/").rsplit("/", 1)[-1]
            for key, reg in self._ADDON_MODULES.items():
                if key.replace("\\", "/").rsplit("/", 1)[-1] == base:
                    registry = reg
                    break
        if registry is not None:
            for kind, imported, local in bindings:
                if kind == "named":
                    attr = registry.get(imported)
                    if attr is None:
                        raise JSError(
                            f"module {source!r} has no export {imported!r}")
                    env.declare(local, self._three_export(attr, source))
                elif kind == "default" and "__default__" in registry:
                    env.declare(
                        local,
                        self._three_export(registry["__default__"], source))
                else:
                    ns = JSObject({
                        name: self._three_export(attr, source)
                        for name, attr in registry.items()
                        if not name.startswith("__")
                    })
                    env.declare(local, ns)
            return
        # Unknown module: keep the historical lenient behaviour (the
        # import line is skipped).  As a courtesy, named imports that
        # match an attribute of the three module (e.g. a loader another
        # part of the port registered there) are still bound.
        for kind, imported, local in bindings:
            if kind == "named" and self.three_module is not None:
                value = getattr(self.three_module, imported, None)
                if value is not None:
                    env.declare(local, value)

    # -- event listeners -------------------------------------------------------
    def _add_event_listener(self, target, type_, callback):
        if callback is not None:
            self._listeners.append((target, str(type_), callback))

    def _remove_event_listener(self, target, type_, callback):
        self._listeners = [
            entry for entry in self._listeners
            if not (entry[0] is target and entry[1] == str(type_)
                    and entry[2] is callback)
        ]

    def dispatch_event(self, target, event):
        """Deliver ``event`` (a JSObject with a ``type`` property) to
        the listeners registered for ``target`` -- via addEventListener
        on the window/document stubs, or via a Python-side ``_listeners``
        dict on the target itself (renderer domElement).  Called by the
        host (main.py) to bridge Pyxel pointer input into JS.  Returns
        the number of callbacks invoked."""
        etype = None
        if isinstance(event, JSObject):
            etype = object.__getattribute__(event, "_props").get("type")
        callbacks = [
            cb for tgt, t, cb in list(self._listeners)
            if tgt is target and t == etype
        ]
        py_listeners = getattr(target, "_listeners", None)
        if isinstance(py_listeners, dict):
            callbacks.extend(py_listeners.get(etype, []))
        for cb in callbacks:
            self._call_value(cb, [event], this_value=target)
        return len(callbacks)

    # -- host globals ---------------------------------------------------------
    def _host_value(self, name, default=None):
        host = self.host
        if host is None:
            return default
        if isinstance(host, dict):
            return host.get(name, default)
        return getattr(host, name, default)

    def _install_globals(self):
        g = self.global_env
        three_ns = self._host_value("THREE", None)
        if three_ns is None:
            three_ns = _make_namespace(self.three_module)
        g.declare("THREE", three_ns)

        def request_animation_frame(cb):
            self.raf_queue.append(cb)
            return float(len(self.raf_queue))

        g.declare("requestAnimationFrame", request_animation_frame)

        body = _StubElement("body")
        body.addEventListener = (
            lambda type_=None, cb=None, *a:
                self._add_event_listener(body, type_, cb))
        body.removeEventListener = (
            lambda type_=None, cb=None, *a:
                self._remove_event_listener(body, type_, cb))
        def _create_element(tag="div"):
            if str(tag).lower() == "canvas":
                return self.three_module.CanvasElement()
            return _StubElement(tag)

        # The document's canvas: samples overwhelmingly fetch it with
        # document.querySelector('#c') (179 of the 183 manual lessons)
        # and hand it to WebGLRenderer, so the canvas selectors must
        # yield a real CanvasElement.  One shared element per
        # interpreter, like the one canvas in a real page.  Every
        # other selector is absorbed by the stub element / empty list
        # in the usual stub style (labels, loading overlays, view
        # divs: clientWidth & friends read as the host viewport).
        queried_canvas = []

        def _query_canvas():
            if not queried_canvas:
                queried_canvas.append(self.three_module.CanvasElement())
            return queried_canvas[0]

        def _query_selector(sel=None):
            s = str(sel).strip().lower() if sel is not None else ""
            if s in ("#c", "canvas"):
                return _query_canvas()
            return _StubElement()

        def _query_selector_all(sel=None):
            s = str(sel).strip().lower() if sel is not None else ""
            if s in ("#c", "canvas"):
                return [_query_canvas()]
            return []

        # The page's own URL: samples read query flags off
        # window.location / document.location (WebVR opt-in).  The
        # host page has no query string, so search is empty.
        location = JSObject({
            "href": "https://localhost/",
            "protocol": "https:",
            "host": "localhost",
            "hostname": "localhost",
            "port": "",
            "pathname": "/",
            "search": "",
            "hash": "",
            "origin": "https://localhost",
        })

        def _get_element_by_id(_id=None):
            el = _StubElement()
            if _id is not None:
                key = str(_id)
                el.id = key
                if key in self._html_elements:
                    txt = self._html_elements[key]
                    el.textContent = txt
                    el.innerText = txt
                    el.innerHTML = txt
                    el.value = txt
            return el

        document = JSObject({
            "body": body,
            "createElement": _create_element,
            "getElementById": _get_element_by_id,
            "querySelector": _query_selector,
            "querySelectorAll": _query_selector_all,
        })
        doc_props = object.__getattribute__(document, "_props")
        doc_props["addEventListener"] = (
            lambda type_=None, cb=None, *a:
                self._add_event_listener(document, type_, cb))
        doc_props["removeEventListener"] = (
            lambda type_=None, cb=None, *a:
                self._remove_event_listener(document, type_, cb))
        doc_props["location"] = location
        g.declare("document", self._host_value("document", document))

        window = JSObject({
            "innerWidth": 256,
            "innerHeight": 192,
            "requestAnimationFrame": request_animation_frame,
            "document": document,
        })
        win_props = object.__getattribute__(window, "_props")
        win_props["addEventListener"] = (
            lambda type_=None, cb=None, *a:
                self._add_event_listener(window, type_, cb))
        win_props["removeEventListener"] = (
            lambda type_=None, cb=None, *a:
                self._remove_event_listener(window, type_, cb))
        win_props["location"] = location
        win_props["URL"] = _JSURL
        window.devicePixelRatio = 1.0
        g.declare("window", self._host_value("window", window))
        g.declare("self", window)
        g.declare("location", location)
        g.declare("URL", _JSURL)
        g.declare("Map", JSMap)
        g.declare("Set", JSSet)
        g.declare("Promise", JSPromise)
        g.declare("RegExp", JSRegExp)
        g.declare("parseInt", _js_parse_int)
        g.declare("parseFloat", _js_parse_float)
        g.declare("encodeURIComponent",
                  lambda s="", *a: _encode_uri(s, frozenset()))
        g.declare("encodeURI",
                  lambda s="", *a: _encode_uri(s, _URI_RESERVED))
        g.declare("decodeURIComponent",
                  lambda s="", *a: _decode_uri(s, keep_reserved=False))
        g.declare("decodeURI",
                  lambda s="", *a: _decode_uri(s, keep_reserved=True))

        def _set_timer(cb, delay=0, *args, interval=False):
            self._timer_seq += 1
            tid = self._timer_seq
            self._timers[tid] = (cb, list(args), interval)
            return float(tid)

        def _clear_timer(tid=None):
            try:
                self._timers.pop(int(tid), None)
            except Exception:
                pass
            return None

        g.declare("setTimeout",
                  lambda cb=None, delay=0, *a:
                      _set_timer(cb, delay, *a, interval=False))
        g.declare("setInterval",
                  lambda cb=None, delay=0, *a:
                      _set_timer(cb, delay, *a, interval=True))
        g.declare("clearTimeout", _clear_timer)
        g.declare("clearInterval", _clear_timer)
        win_props["setTimeout"] = g.lookup("setTimeout")
        win_props["setInterval"] = g.lookup("setInterval")
        win_props["clearTimeout"] = _clear_timer
        win_props["clearInterval"] = _clear_timer

        def _log(*args):
            self.console_log.append(" ".join(_to_str(a) for a in args))

        console = JSObject({"log": _log, "warn": _log, "error": _log,
                            "info": _log})
        g.declare("console", console)

        # split.js (page-side layout library, loaded by a classic
        # script tag in the source pages): the manual's responsive
        # editor calls the global Split(selectors, options) once to
        # arrange its panes.  Layout does not exist in this host, so
        # the call is absorbed -- the returned instance's methods are
        # harmless no-ops, in the Stats / lil-gui stub style.
        g.declare("Split", lambda *a: _SplitInstance())

        math_ns = JSObject({
            "PI": math.pi, "E": math.e,
            "abs": abs, "floor": math.floor, "ceil": math.ceil,
            "round": lambda x: math.floor(x + 0.5),
            "sqrt": math.sqrt, "pow": math.pow,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "atan2": math.atan2, "exp": math.exp, "log": math.log,
            "max": lambda *a: max(a) if a else float("-inf"),
            "min": lambda *a: min(a) if a else float("inf"),
            "random": _js_random,
        })
        g.declare("Math", math_ns)
        g.declare("undefined", None)
        g.declare("NaN", float("nan"))
        g.declare("Infinity", float("inf"))

        # Error constructors: throw new Error("...") is the idiomatic
        # throw in sample code; the result is a plain object with
        # name/message, which is also what _wrap_error produces for
        # interpreter-raised errors.
        for err_name in ("Error", "TypeError", "RangeError",
                         "ReferenceError", "SyntaxError"):
            def _error_factory(message=None, _n=err_name):
                return JSObject({
                    "name": _n,
                    "message": "" if message is None else _to_str(message),
                })
            g.declare(err_name, _error_factory)

        # Small extras in the same spirit (frequent blockers in real
        # sample code): Array.isArray, Object.keys/values/entries/
        # assign, JSON.parse/stringify.
        g.declare("Array", JSObject({
            "isArray": JSBuiltinMethod(
                lambda v=None: isinstance(v, list)),
        }))
        g.declare("Object", JSObject({
            "keys": JSBuiltinMethod(lambda o=None: _object_keys(o)),
            "values": JSBuiltinMethod(
                lambda o=None: [_get_member_any(o, k)
                                for k in _object_keys(o)]),
            "entries": JSBuiltinMethod(
                lambda o=None: [[k, _get_member_any(o, k)]
                                for k in _object_keys(o)]),
            "assign": JSBuiltinMethod(_object_assign),
        }))
        g.declare("JSON", JSObject({
            "parse": JSBuiltinMethod(_json_parse),
            "stringify": JSBuiltinMethod(_json_stringify),
        }))

        # Date and performance: the two clock sources sample code
        # reaches for (Date.now() * 0.001 is the classic animate-loop
        # clock; performance.now() is the modern one).
        g.declare("Date", JSDate)
        performance = JSObject({"now": _performance_now})
        g.declare("performance", performance)
        win_props["performance"] = performance

        # Typed arrays + ArrayBuffer: geometry/texture setup in
        # sample code is built on these (new Float32Array(count * 3)
        # is the single most common construction in the examples).
        for _ta_name, _ta_cls in (
                ("Float32Array", Float32Array),
                ("Float64Array", Float64Array),
                ("Float16Array", Float16Array),
                ("Int8Array", Int8Array),
                ("Uint8Array", Uint8Array),
                ("Uint8ClampedArray", Uint8ClampedArray),
                ("Int16Array", Int16Array),
                ("Uint16Array", Uint16Array),
                ("Int32Array", Int32Array),
                ("Uint32Array", Uint32Array),
                ("ArrayBuffer", JSArrayBuffer)):
            g.declare(_ta_name, _ta_cls)

    # -- public API -----------------------------------------------------------
    def run(self, source):
        if not isinstance(source, str):
            raise JSError("run() expects a source string")
        try:
            program = Parser(_lex(source)).parse_program()
        except JSError:
            raise
        except Exception as exc:  # defensive: never leak non-JS errors
            raise JSError(f"parse error: {exc}")
        try:
            self._hoist_vars(program[1], self.global_env)
            self._exec_block(program[1], self.global_env, this_value=None)
        except _Return:
            raise JSError("return outside function")
        except (_Break, _Continue):
            raise JSError("break/continue outside loop")
        except _ThrowJS as thrown:
            raise JSError(f"uncaught throw: {_to_str(thrown.value)}")
        except JSError:
            raise
        except Exception as exc:
            raise JSError(f"runtime error: {type(exc).__name__}: {exc}")
        self._drain_deferred()
        return None

    def _drain_deferred(self):
        """Task-boundary work, in browser order: deferred
        LoadingManager.onLoad notifications (three.py queues them;
        in a browser they fire when the last async load settles,
        which for a synchronous port is 'after the current script')
        and due timers."""
        three_module = self.three_module
        drain = getattr(three_module, "drain_loading_managers", None)
        if drain is not None:
            drain()
        self._drain_timers()

    def _drain_timers(self):
        if not self._timers:
            return
        due = sorted(self._timers.items())
        for tid, (cb, args, is_interval) in due:
            if tid not in self._timers:
                continue  # cleared by an earlier timer in this drain
            if not is_interval:
                del self._timers[tid]
            if cb is not None and (callable(cb)
                                   or isinstance(cb, JSFunction)):
                self._call_value(cb, list(args), this_value=None)

    def call_raf(self):
        """Run each currently-registered RAF callback once."""
        self._drain_deferred()
        queue = self.raf_queue
        self.raf_queue = []
        self._raf_time += 16.7
        for cb in queue:
            self._call_value(cb, [self._raf_time], this_value=None)

    # -- statements -------------------------------------------------------------
    def _exec_block(self, stmts, env, this_value):
        # Hoisting (JS semantics): function declarations in this block
        # are instantiated before any statement runs, so a function
        # may be called above its declaration (the `init();` style
        # used throughout the official examples).  When a name is
        # declared twice the later declaration wins.  Blocks share the
        # enclosing env in this interpreter, so a block's declarations
        # land in that env when the block is entered.
        self._hoist_funcdecls(stmts, env)
        for stmt in stmts:
            if stmt is not None:
                self._exec_stmt(stmt, env, this_value)

    def _hoist_funcdecls(self, stmts, env):
        for stmt in stmts:
            if stmt is not None and stmt[0] == "funcdecl":
                _, name, params, body = stmt
                env.declare(name, JSFunction(self, name, params, body,
                                             env))

    def _hoist_vars(self, stmts, env):
        """Instantiate the var bindings of a program/function body:
        every name declared with var anywhere in the body (outside
        nested functions and classes) exists from scope entry with
        value undefined; initializers still run at their original
        position.  Names already bound in this env (parameters, the
        function's own name, arguments) are left untouched."""
        names = []
        self._collect_var_names(stmts, names)
        for name in names:
            if name not in env.vars:
                env.declare(name, UNDEFINED)

    def _collect_var_names(self, stmts, out):
        for stmt in stmts:
            if stmt is not None:
                self._collect_var_names_stmt(stmt, out)

    def _collect_var_names_stmt(self, node, out):
        tag = node[0]
        if tag == "vardecl":
            if node[1] == "var":
                for pattern, _init in node[2]:
                    self._pattern_names(pattern, out)
        elif tag in ("program", "block"):
            self._collect_var_names(node[1], out)
        elif tag == "if":
            self._collect_var_names_stmt(node[2], out)
            if node[3] is not None:
                self._collect_var_names_stmt(node[3], out)
        elif tag == "while":
            self._collect_var_names_stmt(node[2], out)
        elif tag == "dowhile":
            self._collect_var_names_stmt(node[1], out)
        elif tag == "for":
            if node[1] is not None:
                self._collect_var_names_stmt(node[1], out)
            self._collect_var_names_stmt(node[4], out)
        elif tag in ("forof", "forin"):
            if node[1] == "var":
                self._pattern_names(node[2], out)
            self._collect_var_names_stmt(node[4], out)
        elif tag == "try":
            self._collect_var_names_stmt(node[1], out)
            if node[3] is not None:
                self._collect_var_names_stmt(node[3], out)
            if node[4] is not None:
                self._collect_var_names_stmt(node[4], out)
        # funcdecl / classdecl bodies are separate scopes: not entered.
        # Expressions are not entered either (function expressions and
        # arrows inside them open their own scope).

    def _pattern_names(self, pattern, out):
        tag = pattern[0]
        if tag == "pident":
            out.append(pattern[1])
        elif tag == "parr":
            for el in pattern[1]:
                if el is None:
                    continue
                if el[0] == "prest":
                    out.append(el[1])
                else:
                    self._pattern_names(el[0], out)
        elif tag == "pobj":
            for _key, sub, _default in pattern[1]:
                self._pattern_names(sub, out)
            if pattern[2] is not None:
                out.append(pattern[2])
        # "pmember" targets are assignments, not declarations.

    def _exec_stmt(self, node, env, this_value):
        tag = node[0]
        if tag == "program" or tag == "block":
            # Blocks share the enclosing scope (var-style hoisting is
            # approximated; sample code does not rely on block scoping).
            self._exec_block(node[1], env, this_value)
            return
        if tag == "import":
            self._exec_import(node[1], node[2], env)
            return
        if tag == "vardecl":
            for pattern, init in node[2]:
                if init is None:
                    if node[1] == "var":
                        # The binding was already instantiated by
                        # hoisting; a bare var declaration assigns
                        # nothing (it must not clobber a parameter
                        # or a value assigned earlier).
                        continue
                    value = UNDEFINED
                else:
                    value = self._eval(init, env, this_value)
                self._bind_pattern(pattern, value, env, this_value,
                                   declare=True)
            return
        if tag == "funcdecl":
            # Function declarations bind at scope entry (see
            # _exec_block's hoisting scan), never at their statement
            # position: executing the statement itself is a no-op, so
            # an intervening assignment (var both = 5) is not undone
            # when the declaration line is reached.  Only a
            # declaration in a bare single-statement position (an
            # if/loop body without braces -- not valid strict-mode
            # JS, but accepted by the parser) can arrive here
            # unhoisted; bind it if the name is still unknown.
            try:
                env.lookup(node[1])
            except JSError:
                _, name, params, body = node
                env.declare(name, JSFunction(self, name, params, body,
                                             env))
            return
        if tag == "classdecl":
            cls = self._make_class(node[1], env)
            if node[1][1] is not None:
                env.declare(node[1][1], cls)
            return
        if tag == "throw":
            raise _ThrowJS(self._eval(node[1], env, this_value))
        if tag == "try":
            _, body, catch_param, catch_body, finally_body = node
            try:
                try:
                    self._exec_block(body[1], env, this_value)
                except _ThrowJS as thrown:
                    if catch_body is None:
                        raise
                    self._exec_catch(catch_body, catch_param,
                                     thrown.value, env, this_value)
                except JSError as exc:
                    if catch_body is None:
                        raise
                    self._exec_catch(catch_body, catch_param,
                                     _wrap_error(exc), env, this_value)
            finally:
                if finally_body is not None:
                    self._exec_block(finally_body[1], env, this_value)
            return
        if tag == "exprstmt":
            self._eval(node[1], env, this_value)
            return
        if tag == "if":
            _, cond, then, otherwise = node
            if _truthy(self._eval(cond, env, this_value)):
                self._exec_stmt(then, env, this_value)
            elif otherwise is not None:
                self._exec_stmt(otherwise, env, this_value)
            return
        if tag == "return":
            value = UNDEFINED
            if node[1] is not None:
                value = self._eval(node[1], env, this_value)
            raise _Return(value)
        if tag == "break":
            raise _Break()
        if tag == "switch":
            _, disc_expr, clauses = node
            disc_val = self._eval(disc_expr, env, this_value)
            start_idx = None
            default_idx = None
            for idx, (test_expr, _) in enumerate(clauses):
                if test_expr is None:
                    if default_idx is None:
                        default_idx = idx
                else:
                    case_val = self._eval(test_expr, env, this_value)
                    if _strict_eq(disc_val, case_val):
                        start_idx = idx
                        break
            if start_idx is None:
                start_idx = default_idx
            if start_idx is not None:
                try:
                    for idx in range(start_idx, len(clauses)):
                        for stmt in clauses[idx][1]:
                            self._exec_stmt(stmt, env, this_value)
                except _Break:
                    pass
            return
        if tag == "continue":
            raise _Continue()
        if tag == "while":
            _, cond, body = node
            while _truthy(self._eval(cond, env, this_value)):
                try:
                    self._exec_stmt(body, env, this_value)
                except _Break:
                    break
                except _Continue:
                    continue
            return
        if tag == "dowhile":
            _, body, cond = node
            while True:
                try:
                    self._exec_stmt(body, env, this_value)
                except _Break:
                    break
                except _Continue:
                    pass
                if not _truthy(self._eval(cond, env, this_value)):
                    break
            return
        if tag == "for":
            _, init, cond, update, body = node
            if init is not None:
                self._exec_stmt(init, env, this_value)
            while True:
                if cond is not None and not _truthy(self._eval(cond, env, this_value)):
                    break
                try:
                    self._exec_stmt(body, env, this_value)
                except _Break:
                    break
                except _Continue:
                    pass
                if update is not None:
                    self._eval(update, env, this_value)
            return
        if tag in ("forof", "forin"):
            _, _kind, pattern, iterable_node, body = node
            iterable = self._eval(iterable_node, env, this_value)
            if tag == "forof":
                items = _iterable_items(iterable)
            else:
                items = _iterable_keys(iterable)
            for item in items:
                self._bind_pattern(pattern, item, env, this_value,
                                   declare=True)
                try:
                    self._exec_stmt(body, env, this_value)
                except _Break:
                    break
                except _Continue:
                    continue
            return
        raise JSError(f"unknown statement {tag!r}")

    def _exec_catch(self, catch_body, catch_param, value, env, this_value):
        cenv = Env(env)
        if catch_param is not None:
            self._bind_pattern(catch_param, value, cenv, this_value,
                               declare=True)
        self._exec_block(catch_body[1], cenv, this_value)

    # -- classes ---------------------------------------------------------------
    def _make_class(self, node, env):
        _, name, super_node, members, fields = node
        superclass = None
        if super_node is not None:
            superclass = self._eval(super_node, env, None)
            if not isinstance(superclass, (JSClass, type)):
                raise JSError(
                    f"class {name or '(anonymous)'}: superclass is not "
                    "a class")
        ctor = None
        methods, statics, getters, setters = {}, {}, {}, {}
        static_getters, static_setters = {}, {}
        for kind, mname, params, body, is_static in members:
            fn = JSFunction(self, mname, params, body, env,
                            is_constructor=(kind == "constructor"))
            if kind == "constructor":
                if ctor is not None:
                    raise JSError(
                        f"class {name or '(anonymous)'}: duplicate "
                        "constructor")
                ctor = fn
            elif kind == "method":
                (statics if is_static else methods)[mname] = fn
            elif kind == "get":
                (static_getters if is_static else getters)[mname] = fn
            elif kind == "set":
                (static_setters if is_static else setters)[mname] = fn
        inst_fields = [(fname, init) for fname, init, is_static in fields
                       if not is_static]
        cls = JSClass(self, name, superclass, ctor, methods, statics,
                      getters, setters, inst_fields, {})
        cls.static_getters = static_getters
        cls.static_setters = static_setters
        cls.def_env = env
        for fn in list(methods.values()) + list(statics.values()) + \
                list(getters.values()) + list(setters.values()) + \
                list(static_getters.values()) + \
                list(static_setters.values()) + \
                ([ctor] if ctor is not None else []):
            fn.home_class = cls
        # Static field initializers run at class definition time.
        for fname, init, is_static in fields:
            if is_static:
                value = UNDEFINED
                if init is not None:
                    value = self._eval(init, env, cls)
                cls.static_fields[fname] = value
        return cls

    def _run_field_inits(self, cls, inst):
        env = cls.def_env
        for fname, init in cls.fields:
            value = UNDEFINED
            if init is not None:
                value = self._eval(init, env, inst)
            self._set_member(inst, fname, value)

    def _init_py_base(self, inst, pytype, args):
        if object.__getattribute__(inst, "_base") is not None:
            return
        try:
            base = pytype(*[_to_python(a) for a in args])
        except Exception as exc:
            raise JSError(
                f"superclass {pytype.__name__} constructor failed: "
                f"{type(exc).__name__}: {exc}")
        object.__setattr__(inst, "_base", base)

    def _construct_class(self, cls, args):
        inst = JSInstance(cls, self)
        result = self._run_ctor(cls, inst, args)
        if isinstance(result, (JSObject, JSInstance, dict)) or \
                (result is not None and not isinstance(
                    result, (int, float, str, bool))):
            return result
        return inst

    def _run_ctor(self, cls, inst, args):
        """Run cls's constructor (explicit or default) on inst;
        returns the constructor's return value."""
        ctor = cls.ctor
        if ctor is None:
            sup = cls.superclass
            if isinstance(sup, JSClass):
                self._run_ctor(sup, inst, args)
            elif isinstance(sup, type):
                self._init_py_base(inst, sup, args)
            self._run_field_inits(cls, inst)
            return UNDEFINED
        frame = {"cls": cls, "inst": inst, "fields_done": False}
        if cls.superclass is None:
            self._run_field_inits(cls, inst)
            frame["fields_done"] = True
        self._ctor_stack.append(frame)
        try:
            result = self._invoke(ctor, args, this_value=inst)
        finally:
            self._ctor_stack.pop()
        if not frame["fields_done"]:
            # A derived constructor that never called super(): real JS
            # throws; stay lenient -- initialise the Python base (if
            # any) and the fields so the instance is usable.
            sup = cls.superclass
            if isinstance(sup, type):
                self._init_py_base(inst, sup, [])
            self._run_field_inits(cls, inst)
        return result

    def _call_super_ctor(self, args, this_value):
        if not self._ctor_stack:
            raise JSError("super() called outside a class constructor")
        frame = self._ctor_stack[-1]
        cls = frame["cls"]
        inst = frame["inst"]
        sup = cls.superclass
        if sup is None:
            raise JSError("super() in a class with no superclass")
        if isinstance(sup, JSClass):
            self._run_ctor(sup, inst, args)
        elif isinstance(sup, type):
            self._init_py_base(inst, sup, args)
        if not frame["fields_done"]:
            self._run_field_inits(cls, inst)
            frame["fields_done"] = True
        return UNDEFINED

    def _eval_super_member(self, name, this_value):
        home = self._home_stack[-1] if self._home_stack else None
        if home is None:
            raise JSError("'super' used outside a class method")
        sup = home.superclass
        if sup is None:
            raise JSError("'super' in a class with no superclass")
        if isinstance(sup, JSClass):
            getter = sup.lookup("getters", name)
            if getter is not None:
                return self._invoke(getter, [], this_value=this_value)
            method = sup.lookup("methods", name)
            if method is not None:
                if isinstance(this_value, JSInstance):
                    return method.bind(this_value)
                return method
            return UNDEFINED
        # Python superclass: resolve on the wrapped base object.
        if isinstance(this_value, JSInstance):
            base = object.__getattribute__(this_value, "_base")
            if base is not None:
                try:
                    return getattr(base, name)
                except AttributeError:
                    return UNDEFINED
        raise JSError(f"super property {name!r} is not available")

    # -- destructuring ---------------------------------------------------------
    def _bind_pattern(self, pattern, value, env, this_value, declare):
        tag = pattern[0]
        if tag == "pident":
            if declare:
                env.declare(pattern[1], value)
            else:
                env.assign(pattern[1], value)
            return
        if tag == "pmember":
            self._assign_target(pattern[1], value, env, this_value)
            return
        if tag == "parr":
            items = _iterable_items(value)
            elems = pattern[1]
            for i, el in enumerate(elems):
                if el is None:
                    continue
                if el[0] == "prest":
                    env.declare(el[1], list(items[i:]))
                    return
                sub, default = el
                item = items[i] if i < len(items) else UNDEFINED
                if item is UNDEFINED and default is not None:
                    item = self._eval(default, env, this_value)
                self._bind_pattern(sub, item, env, this_value, declare)
            return
        if tag == "pobj":
            if value is None:
                raise JSError("cannot destructure undefined")
            _, props, rest = pattern
            used = set()
            for key, sub, default in props:
                used.add(key)
                item = self._get_member(value, key)
                if item is UNDEFINED and default is not None:
                    item = self._eval(default, env, this_value)
                self._bind_pattern(sub, item, env, this_value, declare)
            if rest is not None:
                rest_obj = JSObject()
                rest_props = object.__getattribute__(rest_obj, "_props")
                for key in _object_keys(value):
                    if key not in used:
                        rest_props[key] = self._get_member(value, key)
                env.declare(rest, rest_obj)
            return
        raise JSError(f"unknown pattern {tag!r}")

    # -- expressions ------------------------------------------------------------
    def _eval(self, node, env, this_value):
        tag = node[0]
        if tag == "num" or tag == "str" or tag == "bool":
            return node[1]
        if tag == "regex":
            return JSRegExp(node[1], node[2])
        if tag == "undef":
            return UNDEFINED
        if tag == "tmpl":
            return "".join(_to_str(self._eval(p, env, this_value))
                           for p in node[1])
        if tag == "this":
            return this_value
        if tag == "ident":
            return env.lookup(node[1])
        if tag == "super":
            raise JSError("'super' is only valid as super(...) or "
                          "super.property")
        if tag == "classexpr":
            return self._make_class(node[1], env)
        if tag == "array":
            out = []
            for e in node[1]:
                if e[0] == "spread":
                    out.extend(_iterable_items(
                        self._eval(e[1], env, this_value)))
                else:
                    out.append(self._eval(e, env, this_value))
            return out
        if tag == "object":
            obj = JSObject()
            props = object.__getattribute__(obj, "_props")
            for key, val_node in node[1]:
                if key is None:  # {...spread}
                    src = self._eval(val_node[1], env, this_value)
                    for k in _object_keys(src):
                        props[k] = self._get_member(src, k)
                else:
                    props[key] = self._eval(val_node, env, this_value)
            return obj
        if tag == "destruct":
            value = self._eval(node[2], env, this_value)
            self._bind_pattern(node[1], value, env, this_value,
                               declare=False)
            return value
        if tag == "funcexpr":
            _, name, params, body = node
            fn = JSFunction(self, name, params, body, env)
            return fn
        if tag == "arrowexpr":
            _, params, expr = node
            return JSFunction(self, None, params, None, env, expr_body=expr)
        if tag == "member":
            if node[1][0] == "super":
                return self._eval_super_member(node[2], this_value)
            obj = self._eval(node[1], env, this_value)
            return self._get_member(obj, node[2])
        if tag == "index":
            obj = self._eval(node[1], env, this_value)
            idx = self._eval(node[2], env, this_value)
            return self._get_index(obj, idx)
        if tag == "call":
            return self._eval_call(node, env, this_value)
        if tag == "new":
            callee = self._eval(node[1], env, this_value)
            args = self._eval_args(node[2], env, this_value)
            return self._construct(callee, args)
        if tag == "unary":
            return self._eval_unary(node, env, this_value)
        if tag == "await":
            val = self._eval(node[1], env, this_value)
            if hasattr(val, "_state") and hasattr(val, "_value"):
                if val._state == "rejected":
                    err = val._value
                    if isinstance(err, Exception):
                        raise err
                    raise JSError(str(err))
                return val._value
            return val
        if tag == "update":
            return self._eval_update(node, env, this_value)
        if tag == "binary":
            return self._eval_binary(node, env, this_value)
        if tag == "cond":
            _, cond, yes, no = node
            branch = yes if _truthy(self._eval(cond, env, this_value)) else no
            return self._eval(branch, env, this_value)
        if tag == "assign":
            return self._eval_assign(node, env, this_value)
        raise JSError(f"unknown expression {tag!r}")

    def _eval_unary(self, node, env, this_value):
        _, op, operand = node
        if op == "typeof":
            if operand[0] == "ident":
                try:
                    v = env.lookup(operand[1])
                except JSError:
                    return "undefined"
            else:
                v = self._eval(operand, env, this_value)
            return _typeof(v)
        v = self._eval(operand, env, this_value)
        if op == "-":
            return -_to_number(v)
        if op == "+":
            return _to_number(v)
        if op == "~":
            return float(~_to_int32(v))
        if op == "!":
            return not _truthy(v)
        if op == "void":
            return UNDEFINED
        if op == "delete":
            return True
        raise JSError(f"unsupported unary operator {op!r}")

    def _eval_update(self, node, env, this_value):
        _, op, target, prefix = node
        old = self._eval(target, env, this_value)
        new = _to_number(old) + (1.0 if op == "++" else -1.0)
        self._assign_target(target, new, env, this_value)
        return new if prefix else old

    def _eval_binary(self, node, env, this_value):
        _, op, left_node, right_node = node
        if op == "&&":
            left = self._eval(left_node, env, this_value)
            return self._eval(right_node, env, this_value) if _truthy(left) else left
        if op == "||":
            left = self._eval(left_node, env, this_value)
            return left if _truthy(left) else self._eval(right_node, env, this_value)
        left = self._eval(left_node, env, this_value)
        right = self._eval(right_node, env, this_value)
        if op == "+":
            if isinstance(left, str) or isinstance(right, str):
                return _to_str(left) + _to_str(right)
            return _to_number(left) + _to_number(right)
        if op == "-":
            return _to_number(left) - _to_number(right)
        if op == "*":
            return _to_number(left) * _to_number(right)
        if op == "/":
            r = _to_number(right)
            if r == 0:
                l = _to_number(left)
                return float("nan") if l == 0 else math.copysign(float("inf"), l)
            return _to_number(left) / r
        if op == "%":
            r = _to_number(right)
            if r == 0:
                return float("nan")
            return math.fmod(_to_number(left), r)
        if op == "**":
            return _to_number(left) ** _to_number(right)
        if op in ("==", "==="):
            return _loose_eq(left, right)
        if op in ("!=", "!=="):
            return not _loose_eq(left, right)
        if op in ("<", ">", "<=", ">="):
            if isinstance(left, str) and isinstance(right, str):
                a, b = left, right
            else:
                a, b = _to_number(left), _to_number(right)
            if op == "<":
                return a < b
            if op == ">":
                return a > b
            if op == "<=":
                return a <= b
            return a >= b
        if op == "instanceof":
            return _instanceof(left, right)
        if op in ("&", "|", "^", "<<", ">>", ">>>"):
            return _bitwise_binary(op, left, right)
        raise JSError(f"unsupported operator {op!r}")

    def _eval_assign(self, node, env, this_value):
        _, op, target, value_node = node
        if op == "=":
            value = self._eval(value_node, env, this_value)
        else:
            current = self._eval(target, env, this_value)
            rhs = self._eval(value_node, env, this_value)
            base = op[0]
            if base == "+":
                if isinstance(current, str) or isinstance(rhs, str):
                    value = _to_str(current) + _to_str(rhs)
                else:
                    value = _to_number(current) + _to_number(rhs)
            elif base == "-":
                value = _to_number(current) - _to_number(rhs)
            elif base == "*":
                value = _to_number(current) * _to_number(rhs)
            elif base == "/":
                r = _to_number(rhs)
                value = float("nan") if r == 0 else _to_number(current) / r
            elif base == "%":
                r = _to_number(rhs)
                value = float("nan") if r == 0 else math.fmod(_to_number(current), r)
            elif op == "**=":
                value = _to_number(current) ** _to_number(rhs)
            elif op in ("&=", "|=", "^=", "<<=", ">>=", ">>>="):
                value = _bitwise_binary(op[:-1], current, rhs)
            else:
                raise JSError(f"unsupported compound assignment {op!r}")
        self._assign_target(target, value, env, this_value)
        return value

    def _assign_target(self, target, value, env, this_value):
        tag = target[0]
        if tag == "ident":
            env.assign(target[1], value)
            return
        if tag == "member":
            obj = self._eval(target[1], env, this_value)
            self._set_member(obj, target[2], value)
            return
        if tag == "index":
            obj = self._eval(target[1], env, this_value)
            idx = self._eval(target[2], env, this_value)
            self._set_index(obj, idx, value)
            return
        raise JSError("invalid assignment target")

    def _eval_args(self, arg_nodes, env, this_value):
        args = []
        for a in arg_nodes:
            if a[0] == "spread":
                args.extend(_iterable_items(
                    self._eval(a[1], env, this_value)))
            else:
                args.append(self._eval(a, env, this_value))
        return args

    def _eval_call(self, node, env, this_value):
        _, callee_node, arg_nodes = node
        if callee_node[0] == "super":
            args = self._eval_args(arg_nodes, env, this_value)
            return self._call_super_ctor(args, this_value)
        if callee_node[0] == "member" and callee_node[1][0] == "super":
            args = self._eval_args(arg_nodes, env, this_value)
            fn = self._eval_super_member(callee_node[2], this_value)
            return self._call_value(fn, args, this_value=this_value)
        args = self._eval_args(arg_nodes, env, this_value)
        if callee_node[0] == "member":
            obj = self._eval(callee_node[1], env, this_value)
            fn = self._get_member(obj, callee_node[2])
            return self._call_value(fn, args, this_value=obj)
        if callee_node[0] == "index":
            obj = self._eval(callee_node[1], env, this_value)
            idx = self._eval(callee_node[2], env, this_value)
            fn = self._get_index(obj, idx)
            return self._call_value(fn, args, this_value=obj)
        fn = self._eval(callee_node, env, this_value)
        return self._call_value(fn, args, this_value=this_value)

    # -- calls / construction ---------------------------------------------------
    def _call_value(self, fn, args, this_value=None):
        if isinstance(fn, JSFunction):
            return self._invoke(fn, args, this_value)
        if isinstance(fn, JSBuiltinMethod):
            return fn.fn(*args)
        if isinstance(fn, JSClass):
            raise JSError(
                f"class constructor {fn.name or '(anonymous)'} cannot "
                "be invoked without 'new'")
        if callable(fn):
            args = [_to_python(a) for a in args]
            try:
                return fn(*args)
            except JSError:
                raise
            except TypeError as exc:
                raise JSError(f"call failed: {exc}")
            except Exception as exc:
                raise JSError(f"call failed: {type(exc).__name__}: {exc}")
        raise JSError(f"value is not callable: {fn!r}")

    def _invoke(self, fn, args, this_value):
        if fn.bound_this is not None and \
                (this_value is None or this_value is UNDEFINED):
            this_value = fn.bound_this
        local = Env(fn.closure)
        if fn.name:
            local.declare(fn.name, fn)
        i = 0
        for pattern, default in fn.params:
            if pattern[0] == "prest":
                local.declare(pattern[1], list(args[i:]))
                i = len(args)
                break
            value = args[i] if i < len(args) else UNDEFINED
            i += 1
            if value is UNDEFINED and default is not None:
                value = self._eval(default, local, this_value)
            self._bind_pattern(pattern, value, local, this_value,
                               declare=True)
        local.declare("arguments", list(args))
        if fn.body is not None:
            self._hoist_vars(fn.body, local)
        self._home_stack.append(fn.home_class)
        try:
            if fn.expr_body is not None:
                return self._eval(fn.expr_body, local, this_value)
            self._exec_block(fn.body, local, this_value)
        except _Return as ret:
            return ret.value
        finally:
            self._home_stack.pop()
        return UNDEFINED

    def _construct(self, callee, args):
        if isinstance(callee, JSClass):
            return self._construct_class(callee, args)
        if isinstance(callee, JSFunction):
            obj = JSObject()
            result = self._invoke(callee, args, this_value=obj)
            if isinstance(result, (JSObject, dict)) or \
                    (result is not None and not isinstance(
                        result, (int, float, str, bool))):
                return result
            return obj
        if isinstance(callee, type):
            args = [_to_python(a) for a in args]
            try:
                return callee(*args)
            except Exception as exc:
                raise JSError(
                    f"constructor {callee.__name__} failed: "
                    f"{type(exc).__name__}: {exc}")
        if callable(callee):
            # Some host bindings expose factory callables instead of classes.
            try:
                return callee(*args)
            except Exception as exc:
                raise JSError(f"constructor call failed: {exc}")
        raise JSError(f"value is not a constructor: {callee!r}")

    # -- member / index access ----------------------------------------------------
    def _get_member(self, obj, name):
        if obj is None:
            raise JSError(f"cannot read property {name!r} of undefined")
        if isinstance(obj, JSObject):
            props = object.__getattribute__(obj, "_props")
            if name in props:
                return props[name]
            return UNDEFINED
        if isinstance(obj, JSClass):
            if name in obj.static_props:
                return obj.static_props[name]
            getter = obj.lookup("static_getters", name) \
                if hasattr(obj, "static_getters") else None
            if getter is not None:
                return self._invoke(getter, [], this_value=obj)
            method = obj.lookup("statics", name)
            if method is not None:
                return method.bind(obj)
            if name in obj.static_fields:
                return obj.static_fields[name]
            if name == "name":
                return obj.name
            return UNDEFINED
        if isinstance(obj, dict):
            return obj.get(name, UNDEFINED)
        if isinstance(obj, list):
            if name == "length":
                return float(len(obj))
            if name.isdigit():
                i = int(name)
                return obj[i] if i < len(obj) else UNDEFINED
            method = self._array_method(obj, name)
            if method is not None:
                return method
            return UNDEFINED
        if isinstance(obj, str):
            if name == "length":
                return float(len(obj))
            method = self._string_method(obj, name)
            if method is not None:
                return method
            return UNDEFINED
        if isinstance(obj, bool):
            if name == "toString":
                return JSBuiltinMethod(
                    lambda radix=None: "true" if obj else "false")
            if name == "valueOf":
                return JSBuiltinMethod(lambda: obj)
            return UNDEFINED
        if isinstance(obj, (int, float)):
            method = self._number_method(obj, name)
            if method is not None:
                return method
            return UNDEFINED
        if type(obj).__name__ == "ndarray" and \
                type(obj).__module__ == "numpy":
            # numpy arrays play two JS-facing roles in this port:
            # the .array of a BufferAttribute (flat, typed-array-like)
            # and the values in BufferGeometry.attributes (shaped
            # (N, itemSize), BufferAttribute-like).  Expose the union
            # of the two surfaces JS code reads.
            if name == "length":
                return float(obj.size)
            if name == "count":
                return float(obj.shape[0]) if obj.ndim >= 1 else 0.0
            if name == "itemSize":
                return float(obj.shape[1]) if obj.ndim == 2 else 1.0
            if name == "array":
                return obj.reshape(-1) if obj.ndim > 1 else obj
            if name in ("getX", "getY", "getZ", "getComponent",
                        "setX", "setY", "setZ", "setXYZ", "setComponent",
                        "copyAt"):
                return _ndarray_accessor(obj, name)
            return UNDEFINED
        # Python objects (THREE classes/instances, modules, stubs) and
        # JSInstance (whose __getattr__ implements the class lookup).
        try:
            return getattr(obj, name)
        except AttributeError:
            return UNDEFINED

    def _set_member(self, obj, name, value):
        if obj is None:
            raise JSError(f"cannot set property {name!r} of undefined")
        if isinstance(obj, JSObject):
            object.__getattribute__(obj, "_props")[name] = value
            return
        if isinstance(obj, JSClass):
            setter = obj.lookup("static_setters", name) \
                if hasattr(obj, "static_setters") else None
            if setter is not None:
                self._invoke(setter, [value], this_value=obj)
                return
            obj.static_props[name] = value
            return
        if isinstance(obj, dict):
            obj[name] = value
            return
        if isinstance(obj, list) and name == "length":
            new_len = max(0, int(_to_number(value)))
            if new_len < len(obj):
                del obj[new_len:]
            elif new_len > len(obj):
                obj.extend([UNDEFINED] * (new_len - len(obj)))
            return
        if type(obj).__name__ == "ndarray" and \
                type(obj).__module__ == "numpy":
            # numpy arrays stand in for BufferAttributes in this
            # port; JS expandos on them (needsUpdate above all) are
            # no-ops for a renderer that re-reads the array every
            # frame, and numpy forbids attribute assignment anyway.
            return
        try:
            setattr(obj, name, value)
        except Exception as exc:
            raise JSError(f"cannot set property {name!r}: {exc}")

    def _get_index(self, obj, idx):
        if obj is None:
            raise JSError("cannot index undefined")
        if isinstance(idx, str):
            return self._get_member(obj, idx)
        if isinstance(obj, (list, str)):
            i = int(_to_number(idx))
            if 0 <= i < len(obj):
                return obj[i]
            return UNDEFINED
        if isinstance(obj, dict):
            return obj.get(idx, UNDEFINED)
        if isinstance(obj, JSObject):
            return self._get_member(obj, _num_to_str(_to_number(idx)))
        try:
            return obj[int(_to_number(idx))]
        except Exception:
            return UNDEFINED

    def _set_index(self, obj, idx, value):
        if isinstance(idx, str) and not isinstance(obj, (list, str)):
            self._set_member(obj, idx, value)
            return
        if isinstance(obj, list):
            i = int(_to_number(idx))
            while len(obj) <= i:
                obj.append(UNDEFINED)
            obj[i] = value
            return
        if isinstance(obj, dict):
            obj[idx] = value
            return
        if isinstance(obj, JSObject):
            self._set_member(obj, _num_to_str(_to_number(idx)), value)
            return
        try:
            obj[int(_to_number(idx))] = value
        except Exception as exc:
            raise JSError(f"cannot set index {idx!r}: {exc}")


    # -- Array / String prototype surface ---------------------------------------
    def _require_callable(self, fn, what):
        if not (isinstance(fn, (JSFunction, JSBuiltinMethod)) or
                callable(fn)):
            raise JSError(f"{what}: callback is not a function")

    def _array_method(self, arr, name):
        if name == "push":
            return JSBuiltinMethod(
                lambda *a: (arr.extend(a), float(len(arr)))[1])
        if name == "pop":
            return JSBuiltinMethod(
                lambda: arr.pop() if arr else UNDEFINED)
        if name == "shift":
            return JSBuiltinMethod(
                lambda: arr.pop(0) if arr else UNDEFINED)
        if name == "unshift":
            def _unshift(*a):
                arr[:0] = list(a)
                return float(len(arr))
            return JSBuiltinMethod(_unshift)
        if name == "splice":
            def _splice(start=0, delete_count=None, *items):
                n = len(arr)
                idx = int(_to_number(start))
                if idx < 0:
                    idx = max(n + idx, 0)
                else:
                    idx = min(idx, n)
                if delete_count is None or delete_count is UNDEFINED:
                    dc = n - idx
                else:
                    dc = max(0, min(int(_to_number(delete_count)), n - idx))
                removed = arr[idx : idx + dc]
                arr[idx : idx + dc] = list(items)
                return removed
            return JSBuiltinMethod(_splice)
        if name == "map":
            def _map(cb=None, this_arg=UNDEFINED):
                self._require_callable(cb, "Array.map")
                return [self._call_value(cb, [v, float(i), arr],
                                         this_value=this_arg)
                        for i, v in enumerate(arr)]
            return JSBuiltinMethod(_map)
        if name == "filter":
            def _filter(cb=None, this_arg=UNDEFINED):
                self._require_callable(cb, "Array.filter")
                return [v for i, v in enumerate(arr)
                        if _truthy(self._call_value(
                            cb, [v, float(i), arr], this_value=this_arg))]
            return JSBuiltinMethod(_filter)
        if name == "forEach":
            def _for_each(cb=None, this_arg=UNDEFINED):
                self._require_callable(cb, "Array.forEach")
                for i, v in enumerate(arr):
                    self._call_value(cb, [v, float(i), arr],
                                     this_value=this_arg)
                return UNDEFINED
            return JSBuiltinMethod(_for_each)
        if name == "reduce":
            def _reduce(cb=None, *init):
                self._require_callable(cb, "Array.reduce")
                if init:
                    acc, start = init[0], 0
                else:
                    if not arr:
                        raise JSError(
                            "reduce of empty array with no initial value")
                    acc, start = arr[0], 1
                for i in range(start, len(arr)):
                    acc = self._call_value(
                        cb, [acc, arr[i], float(i), arr],
                        this_value=UNDEFINED)
                return acc
            return JSBuiltinMethod(_reduce)
        if name == "find":
            def _find(cb=None, this_arg=UNDEFINED):
                self._require_callable(cb, "Array.find")
                for i, v in enumerate(arr):
                    if _truthy(self._call_value(
                            cb, [v, float(i), arr], this_value=this_arg)):
                        return v
                return UNDEFINED
            return JSBuiltinMethod(_find)
        if name == "findIndex":
            def _find_index(cb=None, this_arg=UNDEFINED):
                self._require_callable(cb, "Array.findIndex")
                for i, v in enumerate(arr):
                    if _truthy(self._call_value(
                            cb, [v, float(i), arr], this_value=this_arg)):
                        return float(i)
                return -1.0
            return JSBuiltinMethod(_find_index)
        if name == "includes":
            def _includes(x=None, from_index=0):
                i = int(_to_number(from_index))
                if i < 0:
                    i = max(0, len(arr) + i)
                return any(_strict_eq(v, x) for v in arr[i:])
            return JSBuiltinMethod(_includes)
        if name == "indexOf":
            def _index_of(x=None, from_index=0):
                i = int(_to_number(from_index))
                if i < 0:
                    i = max(0, len(arr) + i)
                for j in range(i, len(arr)):
                    if _strict_eq(arr[j], x):
                        return float(j)
                return -1.0
            return JSBuiltinMethod(_index_of)
        if name == "lastIndexOf":
            def _last_index_of(x=None, from_index=None):
                i = len(arr) - 1 if from_index is None \
                    else int(_to_number(from_index))
                if i < 0:
                    i = len(arr) + i
                for j in range(min(i, len(arr) - 1), -1, -1):
                    if _strict_eq(arr[j], x):
                        return float(j)
                return -1.0
            return JSBuiltinMethod(_last_index_of)
        if name == "slice":
            def _slice(a=None, b=None):
                lo = None if a is None else int(_to_number(a))
                hi = None if b is None else int(_to_number(b))
                return list(arr[lo:hi])
            return JSBuiltinMethod(_slice)
        if name == "join":
            def _join(sep=","):
                return _to_str(sep).join(
                    "" if v is None else _to_str(v) for v in arr)
            return JSBuiltinMethod(_join)
        if name == "concat":
            def _concat(*others):
                out = list(arr)
                for o in others:
                    if isinstance(o, list):
                        out.extend(o)
                    else:
                        out.append(o)
                return out
            return JSBuiltinMethod(_concat)
        if name == "flat":
            def _flat(depth=1):
                d = _to_number(depth)
                out = []

                def rec(a, dd):
                    for v in a:
                        if isinstance(v, list) and dd > 0:
                            rec(v, dd - 1)
                        else:
                            out.append(v)
                rec(arr, d)
                return out
            return JSBuiltinMethod(_flat)
        if name == "flatMap":
            def _flat_map(cb=None, this_arg=UNDEFINED):
                self._require_callable(cb, "Array.flatMap")
                out = []
                for i, v in enumerate(arr):
                    r = self._call_value(cb, [v, float(i), arr],
                                         this_value=this_arg)
                    if isinstance(r, list):
                        out.extend(r)
                    else:
                        out.append(r)
                return out
            return JSBuiltinMethod(_flat_map)
        if name == "reverse":
            def _reverse():
                arr.reverse()
                return arr
            return JSBuiltinMethod(_reverse)
        if name == "sort":
            def _sort(cb=None):
                if cb is None:
                    arr.sort(key=_to_str)
                else:
                    self._require_callable(cb, "Array.sort")
                    import functools

                    def cmp(a, b):
                        r = self._call_value(cb, [a, b],
                                             this_value=UNDEFINED)
                        r = _to_number(r)
                        return -1 if r < 0 else (1 if r > 0 else 0)
                    arr.sort(key=functools.cmp_to_key(cmp))
                return arr
            return JSBuiltinMethod(_sort)
        return None

    def _string_method(self, s, name):
        if name == "split":
            def _split(sep=UNDEFINED, limit=None):
                if sep is UNDEFINED:
                    return [s]
                if isinstance(sep, JSRegExp):
                    parts = sep._re.split(s)
                else:
                    sep = _to_str(sep)
                    if sep == "":
                        parts = list(s)
                    else:
                        parts = s.split(sep)
                if limit is not None:
                    lim = int(_to_number(limit))
                    if lim == 0:
                        return []
                    if lim > 0:
                        parts = parts[:lim]
                return parts
            return JSBuiltinMethod(_split)
        if name == "includes":
            return JSBuiltinMethod(
                lambda sub="", pos=0: _to_str(sub) in s[int(_to_number(pos)):])
        if name == "startsWith":
            return JSBuiltinMethod(
                lambda sub="", pos=0:
                    s[int(_to_number(pos)):].startswith(_to_str(sub)))
        if name == "endsWith":
            def _ends(sub="", pos=None):
                end = len(s) if pos is None else int(_to_number(pos))
                return s[:end].endswith(_to_str(sub))
            return JSBuiltinMethod(_ends)
        if name == "indexOf":
            return JSBuiltinMethod(
                lambda sub="", pos=0: float(
                    s.find(_to_str(sub), int(_to_number(pos)))))
        if name == "lastIndexOf":
            def _last(sub="", pos=None):
                if pos is None:
                    return float(s.rfind(_to_str(sub)))
                return float(s.rfind(_to_str(sub), 0,
                                     int(_to_number(pos)) + len(_to_str(sub))))
            return JSBuiltinMethod(_last)
        if name == "slice":
            def _slice(a=None, b=None):
                lo = None if a is None else int(_to_number(a))
                hi = None if b is None else int(_to_number(b))
                return s[lo:hi]
            return JSBuiltinMethod(_slice)
        if name == "substring":
            def _substring(a=0, b=None):
                lo = max(0, int(_to_number(a)))
                hi = len(s) if b is None else max(0, int(_to_number(b)))
                lo = min(lo, len(s))
                hi = min(hi, len(s))
                if lo > hi:
                    lo, hi = hi, lo
                return s[lo:hi]
            return JSBuiltinMethod(_substring)
        if name == "charAt":
            def _char_at(i=0):
                i = int(_to_number(i))
                return s[i] if 0 <= i < len(s) else ""
            return JSBuiltinMethod(_char_at)
        if name == "charCodeAt":
            def _char_code(i=0):
                i = int(_to_number(i))
                return float(ord(s[i])) if 0 <= i < len(s) \
                    else float("nan")
            return JSBuiltinMethod(_char_code)
        if name == "toUpperCase":
            return JSBuiltinMethod(lambda: s.upper())
        if name == "toLowerCase":
            return JSBuiltinMethod(lambda: s.lower())
        if name == "trim":
            return JSBuiltinMethod(lambda: s.strip())
        if name == "trimStart":
            return JSBuiltinMethod(lambda: s.lstrip())
        if name == "trimEnd":
            return JSBuiltinMethod(lambda: s.rstrip())
        if name == "replace" or name == "replaceAll":
            def _replace(search="", replacement=""):
                if isinstance(search, JSRegExp):
                    cnt = 0 if ("g" in search.flags or name == "replaceAll") else 1
                    if callable(replacement):
                        return search._re.sub(
                            lambda m: _to_str(replacement(m.group(0))),
                            s, count=cnt)
                    rep = _to_str(replacement).replace("$&", r"\g<0>")
                    return search._re.sub(rep, s, count=cnt)
                search = _to_str(search)
                replacement = _to_str(replacement)
                count = -1 if name == "replaceAll" else 1
                if search == "":
                    if count == 1:
                        return replacement + s
                    return replacement.join([""] + list(s) + [""]) \
                        if s else replacement
                out = []
                rest = s
                done = 0
                while count < 0 or done < count:
                    idx = rest.find(search)
                    if idx < 0:
                        break
                    rep = replacement.replace("$$", "\x00") \
                        .replace("$&", search) \
                        .replace("$`", s[:len(s) - len(rest) + idx]) \
                        .replace("$'", rest[idx + len(search):]) \
                        .replace("\x00", "$")
                    out.append(rest[:idx])
                    out.append(rep)
                    rest = rest[idx + len(search):]
                    done += 1
                out.append(rest)
                return "".join(out)
            return JSBuiltinMethod(_replace)
        if name == "repeat":
            def _repeat(n=0):
                n = int(_to_number(n))
                if n < 0:
                    raise JSError("repeat count must be non-negative")
                return s * n
            return JSBuiltinMethod(_repeat)
        if name == "padStart" or name == "padEnd":
            def _pad(target=0, pad=" "):
                target = int(_to_number(target))
                pad = _to_str(pad)
                if len(s) >= target or pad == "":
                    return s
                fill = (pad * (target - len(s)))[:target - len(s)]
                return fill + s if name == "padStart" else s + fill
            return JSBuiltinMethod(_pad)
        if name == "concat":
            return JSBuiltinMethod(
                lambda *parts: s + "".join(_to_str(p) for p in parts))
        if name == "toString" or name == "valueOf":
            return JSBuiltinMethod(lambda: s)
        return None

    def _number_method(self, num, name):
        """Number.prototype surface for plain Python numbers (the
        interpreter's JS numbers): toString (with radix), valueOf,
        toFixed / toPrecision / toExponential."""
        if name == "toString":
            def _to_string(radix=None):
                v = float(num)
                if radix is None:
                    return _num_to_str(v)
                r = int(_to_number(radix))
                if r == 10:
                    return _num_to_str(v)
                if math.isnan(v):
                    return "NaN"
                if math.isinf(v):
                    return "Infinity" if v > 0 else "-Infinity"
                if not 2 <= r <= 36:
                    raise JSError(
                        "Number.prototype.toString: radix out of range")
                digits = "0123456789abcdefghijklmnopqrstuvwxyz"
                sign = "-" if v < 0 else ""
                v = abs(v)
                whole = int(v)
                frac = v - whole
                if whole == 0:
                    out = "0"
                else:
                    out = ""
                    while whole:
                        out = digits[whole % r] + out
                        whole //= r
                if frac > 0.0:
                    tail = ""
                    for _ in range(40):
                        frac *= r
                        d = int(frac)
                        tail += digits[d]
                        frac -= d
                        if frac <= 0.0:
                            break
                    out += "." + tail
                return sign + out
            return JSBuiltinMethod(_to_string)
        if name == "valueOf":
            return JSBuiltinMethod(lambda: float(num))
        if name == "toFixed":
            def _fixed(digits=0):
                d = int(_to_number(digits))
                if not 0 <= d <= 100:
                    raise JSError("toFixed: digits out of range")
                return f"{float(num):.{d}f}"
            return JSBuiltinMethod(_fixed)
        if name == "toPrecision":
            def _precision(p=None):
                if p is None:
                    return _num_to_str(float(num))
                return format(float(num), f".{int(_to_number(p))}g")
            return JSBuiltinMethod(_precision)
        if name == "toExponential":
            def _exponential(digits=None):
                v = float(num)
                if digits is None:
                    return _num_to_str(v) if v == int(v) and \
                        abs(v) < 1e21 else repr(v)
                return format(v, f".{int(_to_number(digits))}e")
            return JSBuiltinMethod(_exponential)
        return None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _to_python(v):
    """Convert interpreter values crossing into host (Python) code:
    JSObject -> dict (recursively), arrays stay lists of converted items."""
    if isinstance(v, JSObject):
        props = object.__getattribute__(v, "_props")
        return {k: _to_python(x) for k, x in props.items()}
    if isinstance(v, list):
        return [_to_python(x) for x in v]
    return v


def _js_random():
    import random
    return random.random()


def _typeof(v):
    if v is None:
        return "undefined"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, JSFunction) or callable(v):
        return "function"
    return "object"


def _strict_eq(a, b):
    """JS === (used by Array indexOf/includes): no type coercion,
    except that int/float are both 'number'."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return a is b


def _instanceof(left, right):
    if isinstance(right, JSClass):
        if isinstance(left, JSInstance):
            for cls in object.__getattribute__(left, "_class").chain():
                if cls is right:
                    return True
            # Not in the class chain: false.  (The Python-base check
            # below only applies when the right side is a Python type;
            # a JSClass is not one, so isinstance() must not see it.)
            return False
        return False
    if isinstance(right, type):
        if isinstance(left, JSInstance):
            base = object.__getattribute__(left, "_base")
            if base is not None and isinstance(base, right):
                return True
            cls = object.__getattribute__(left, "_class")
            while isinstance(cls, JSClass):
                cls = cls.superclass
            return cls is right
        return isinstance(left, right)
    return False


def _wrap_error(exc):
    """Wrap a Python-side JSError as a JS-visible Error object so a
    JS catch block can read .name / .message."""
    obj = JSObject({"name": "Error", "message": str(exc)})
    return obj


def _object_keys(v):
    if isinstance(v, JSObject):
        return list(object.__getattribute__(v, "_props").keys())
    if isinstance(v, dict):
        return [k if isinstance(k, str) else _to_str(k) for k in v.keys()]
    if isinstance(v, (list, str)):
        return [_num_to_str(float(i)) for i in range(len(v))]
    if isinstance(v, JSInstance):
        return list(object.__getattribute__(v, "_props").keys())
    return []


def _get_member_any(v, key):
    if isinstance(v, JSObject):
        return object.__getattribute__(v, "_props").get(key, UNDEFINED)
    if isinstance(v, dict):
        return v.get(key, UNDEFINED)
    if isinstance(v, (list, str)) and key.isdigit():
        i = int(key)
        return v[i] if i < len(v) else UNDEFINED
    try:
        return getattr(v, key)
    except AttributeError:
        return UNDEFINED


def _object_assign(target=None, *sources):
    if target is None:
        raise JSError("Object.assign: target is undefined")
    for src in sources:
        if src is None:
            continue
        for key in _object_keys(src):
            value = _get_member_any(src, key)
            if isinstance(target, JSObject):
                object.__getattribute__(target, "_props")[key] = value
            elif isinstance(target, dict):
                target[key] = value
            else:
                try:
                    setattr(target, key, value)
                except Exception:
                    pass
    return target


def _json_parse(text=None):
    if not isinstance(text, str):
        text = _to_str(text)

    def to_js(v):
        if isinstance(v, dict):
            return JSObject({k: to_js(x) for k, x in v.items()})
        if isinstance(v, list):
            return [to_js(x) for x in v]
        if isinstance(v, bool) or v is None or isinstance(v, str):
            return v
        if isinstance(v, (int, float)):
            return float(v)
        return v

    try:
        return to_js(json.loads(text))
    except Exception as exc:
        raise JSError(f"JSON.parse: {exc}")


def _json_stringify(value=None):
    def sanitize(v):
        if isinstance(v, JSObject):
            props = object.__getattribute__(v, "_props")
            return {k: sanitize(x) for k, x in props.items()}
        if isinstance(v, JSInstance):
            props = object.__getattribute__(v, "_props")
            return {k: sanitize(x) for k, x in props.items()}
        if isinstance(v, dict):
            return {k: sanitize(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sanitize(x) for x in v]
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            if v == int(v) and abs(v) < 1e21:
                return int(v)
            return v
        if isinstance(v, (str, bool)) or v is None:
            return v
        return _to_str(v)

    try:
        return json.dumps(sanitize(value), ensure_ascii=False,
                          separators=(",", ":"))
    except Exception as exc:
        raise JSError(f"JSON.stringify: {exc}")


def _loose_eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        if isinstance(a, bool) and isinstance(b, bool):
            return a == b
        return _to_number(a) == _to_number(b)
    if isinstance(a, (int, float)) and isinstance(b, str):
        return float(a) == _to_number(b)
    if isinstance(a, str) and isinstance(b, (int, float)):
        return _to_number(a) == float(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b or a is b


def _iterable_items(v):
    if v is None:
        raise JSError("value is not iterable")
    if isinstance(v, (list, str)):
        return list(v)
    if isinstance(v, dict):
        return list(v.values())
    try:
        return list(v)
    except TypeError:
        raise JSError(f"value is not iterable: {v!r}")


def _iterable_keys(v):
    if isinstance(v, dict):
        return list(v.keys())
    if isinstance(v, JSObject):
        return list(object.__getattribute__(v, "_props").keys())
    if isinstance(v, (list, str)):
        return [_num_to_str(float(i)) for i in range(len(v))]
    return []


class _Namespace:
    """Attribute view over a module or a dict (used for THREE)."""

    def __init__(self, source):
        object.__setattr__(self, "_source", source)

    def __getattr__(self, name):
        src = object.__getattribute__(self, "_source")
        if isinstance(src, dict):
            if name in src:
                return src[name]
            raise AttributeError(name)
        return getattr(src, name)

    def __setattr__(self, name, value):
        src = object.__getattribute__(self, "_source")
        if isinstance(src, dict):
            src[name] = value
        else:
            setattr(src, name, value)


def _make_namespace(three_module):
    if three_module is None:
        return JSObject()
    if isinstance(three_module, (JSObject, _Namespace)):
        return three_module
    return _Namespace(three_module)

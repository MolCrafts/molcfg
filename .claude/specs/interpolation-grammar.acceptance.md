---
slug: interpolation-grammar
criteria:
  - id: ac-001
    summary: env reference resolves from environ mapping
    type: code
    pass_when: |
      `interpolate({"home": "${env:MY_HOME}"}, environ={"MY_HOME": "/Users/alice"})`
      returns `{"home": "/Users/alice"}`.
  - id: ac-002
    summary: absolute dotted reference resolves from root
    type: code
    pass_when: |
      `interpolate({"a": {"b": "hello"}, "c": "${a.b} world"})` returns
      `{"a": {"b": "hello"}, "c": "hello world"}`.
  - id: ac-003
    summary: relative sibling reference resolves within current mapping
    type: code
    pass_when: |
      `interpolate({"db": {"host": "localhost", "url": "${.host}:5432"}})`
      yields `result["db"]["url"] == "localhost:5432"`.
  - id: ac-004
    summary: relative parent reference climbs one level per extra dot
    type: code
    pass_when: |
      `interpolate({"top": "T", "child": {"v": "${..top}"}})` yields
      `result["child"]["v"] == "T"`.
  - id: ac-005
    summary: env default fires when variable is absent
    type: code
    pass_when: |
      `interpolate({"x": "${env:NOPE, fallback}"}, environ={})` returns
      `{"x": "fallback"}` and does not raise.
  - id: ac-006
    summary: absolute-reference default fires when path is missing
    type: code
    pass_when: |
      `interpolate({"x": "${missing.path, fallback}"})` returns
      `{"x": "fallback"}`.
  - id: ac-007
    summary: relative sibling default fires when sibling missing
    type: code
    pass_when: |
      `interpolate({"db": {"url": "${.host, localhost}"}})` yields
      `result["db"]["url"] == "localhost"`.
  - id: ac-008
    summary: relative parent default fires when ancestor missing
    type: code
    pass_when: |
      `interpolate({"child": {"v": "${..top, T}"}})` yields
      `result["child"]["v"] == "T"`.
  - id: ac-009
    summary: missing reference without default raises InterpretError
    type: code
    pass_when: |
      `interpolate({"a": "${nonexistent}"})` raises `InterpretError`
      whose message contains the path `'nonexistent'`.
  - id: ac-010
    summary: relative reference escaping root raises InterpretError
    type: code
    pass_when: |
      `interpolate({"x": "${..foo}"})` raises `InterpretError` whose
      message contains the substring `escaped root`.
  - id: ac-011
    summary: circular reference raises InterpretError mentioning circular
    type: code
    pass_when: |
      `interpolate({"a": "${b}", "b": "${a}"})` raises `InterpretError`
      whose message contains the substring `circular`.
  - id: ac-012
    summary: malformed placeholder raises ParserError with raw token
    type: code
    pass_when: |
      `interpolate({"a": "${ , foo}"})` raises `ParserError` whose
      message contains the original token `${ , foo}`; same behavior
      for inputs `"${}"` and `"${env:}"`.
  - id: ac-013
    summary: list leaves are recursively interpolated
    type: code
    pass_when: |
      `interpolate({"base": "/opt", "paths": ["${base}/a", "${base}/b", 7]})`
      yields `result["paths"] == ["/opt/a", "/opt/b", 7]` (non-string leaf 7
      preserved unchanged).
  - id: ac-014
    summary: tuple leaves are recursively interpolated and stay tuples
    type: code
    pass_when: |
      `interpolate({"base": "/opt", "paths": ("${base}/a", "${base}/b")})`
      yields `result["paths"] == ("/opt/a", "/opt/b")` and
      `isinstance(result["paths"], tuple)` is true.
  - id: ac-015
    summary: public API exposes interpolate and the two new error classes
    type: code
    pass_when: |
      `from molcfg import interpolate, ParserError, InterpretError`
      succeeds; `"interpolate"`, `"ParserError"`, and `"InterpretError"`
      are all members of `molcfg.__all__`; `from molcfg import
      CircularReferenceError` raises `ImportError`; `from molcfg.errors
      import CircularReferenceError` raises `ImportError`.
  - id: ac-016
    summary: full lint + format + test gate is green
    type: runtime
    pass_when: |
      Running `ruff check . && ruff format --check . && pytest -q` from
      the repo root exits 0 with no failing tests and no lint or format
      diagnostics.
---

# Acceptance criteria

- **ac-001 to ac-004** lock the four happy-path ref shapes (env, absolute, relative sibling, relative parent).
- **ac-005 to ac-008** lock the orthogonal `default` semantics across all four ref shapes — this is the spec's biggest behavioral promise and must not regress.
- **ac-009 to ac-011** lock the `InterpretError` paths (missing without default, escaping root via `..`, circular reference). ac-011 binds the literal substring `circular` so the new message replaces the deleted `CircularReferenceError` cleanly.
- **ac-012** locks the `ParserError` path. The criterion lists three malformed tokens; all three must fail through the parser layer.
- **ac-013 / ac-014** lock container recursion across `list` and `tuple`, including the type-preservation contract for tuples and the no-touch contract for non-string leaves.
- **ac-015** is the public-API contract: positive imports succeed and `__all__` is correct; the deleted `CircularReferenceError` is genuinely unimportable from both `molcfg` and `molcfg.errors`.
- **ac-016** is the runtime gate matching `mol_project.ci.local`.

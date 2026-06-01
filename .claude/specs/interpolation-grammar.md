---
slug: interpolation-grammar
title: Rewrite ${...} 插值引擎并迁出至 interpolation 模块
status: approved
created: 2026-05-07
language: zh
---

# Rewrite ${...} 插值引擎并迁出至 interpolation 模块

## Summary

把 `molcfg` 现有 `${...}` 字符串插值引擎从 `concurrency.py` 中拆出，搬到全新的 `src/molcfg/interpolation.py` 模块，并以 vscode + omegaconf 风格的语法重写。新语法支持 `${env:VAR}`、`${foo.bar}` 绝对引用、`${.foo}` / `${..foo}` 相对引用，以及对四类引用一律生效的 `${ref, default}` 兜底；容器递归从仅支持 `dict` 扩展到 `dict | list | tuple` 的字符串叶子。错误模型同时被替换为新的两层划分 —— `ParserError` 表示 token 语法错误、`InterpretError` 表示找不到值或循环引用 —— 旧的 `CircularReferenceError` 与"无法解析时保留原字符串"的 graceful-fallback 全部移除。本次改动是一次显式的 public-API break，不保留向后兼容。

## Design

### 模块定位

新模块 `src/molcfg/interpolation.py` 位于 `utility` 层（与 `registry` 同列）。允许的内部 `molcfg.*` 依赖只有 `from molcfg.errors import ParserError, InterpretError`；`Config` / `concurrency` 一概不导入。`concurrency.py` 在删除插值代码后回归"并发原语"单一职责：模块 docstring 第一行从 `"""Thread-safe config wrapper, file locking, and interpolation engine."""` 改为 `"""Thread-safe config wrapper and file locking."""`，并删除 `from molcfg.errors import CircularReferenceError`。

### 错误模型

`src/molcfg/errors.py` 中新增两个类，均直接继承 `ConfigError`：

- `ParserError(ConfigError)` — `${...}` token 自身语法不合法。错误消息保留原 token 全文，例如 `ParserError("malformed placeholder: '${}'")`、`ParserError("malformed placeholder: '${ , foo}'")`、`ParserError("malformed placeholder: '${env:}'")`。
- `InterpretError(ConfigError)` — token 解析成功但运行期解释失败。错误消息含 dotted-path 或 env var 名 + 失败原因，例如 `InterpretError("missing config reference: 'foo.bar'")`、`InterpretError("missing env var: 'MY_HOME'")`、`InterpretError("relative reference '..foo' escaped root")`、`InterpretError("circular reference at 'a'")`。

`CircularReferenceError` 类被整段删除；任何 `from molcfg.errors import CircularReferenceError` 也一并删除。

### 语法 / 语义

每一个 `${...}` 占位符在被 parser 抓出后，按如下规则逐个解释：

1. **拆 default**：以**第一个**逗号 `,` split 一次。逗号前后两侧分别 `strip()`；逗号前为 `ref`、之后为 `default`（literal 文本）。无逗号即无 default。default 不再二次插值，不支持嵌套 `${...}`，不支持转义 `,` 或 `}`。
2. **路由 ref 头部**：
   - 以 `env:` 开头 → env 引用。`env:` 之后必须有非空 var 名，否则 `ParserError`。
   - 以 `.` 开头（一个或多个连续点）→ 相对引用。N 个前导点意味着相对当前 mapping 上爬 (N-1) 层；剩余部分按 `.` 切分作为下钻路径。爬过 root → `InterpretError`。
   - 其他情况 → 绝对引用。剩余部分按 `.` 切分，从 root 起下钻。
   - 完全空 ref（含只剩空白）→ `ParserError`。
3. **解析失败回退**：找不到值时如果有 default，用 default literal 文本替换；没有 default 则 `InterpretError`。
4. **拼接**：`${a.b}/data` 之类的部分子串替换照旧支持 —— `re.sub` 替换骨架不变，每个 `${...}` 各自独立解释。
5. **容器递归**：遍历 `dict | list | tuple` 的所有叶子，仅对 `str` 类型的叶子触发解析；`int / bool / None / float` 等非字符串叶子原样保留。`tuple` 输出仍是 `tuple`，`list` 输出仍是 `list`。
6. **顶层 `${.foo}` 的语义**：当当前 mapping 即 root 时，`${.foo}` 等价于绝对引用 `${foo}`（相当于"climb 0 层"）；`${..foo}` 在 root 直接 `InterpretError("relative reference '..foo' escaped root")`。这条规则在 spec 中明确以避免后续争议。
7. **没有转义**：config 中无法表达 literal `${...}`。任何形如 `${...}` 的子串都会被 parser 抓住并尝试解释。

### 内部分解

私有 helper 命名沿袭 `registry.py` 风格（顶层小函数 + snake_case + leading underscore）：

- `interpolate(data, environ=None)` — 唯一公开入口。`data: dict[str, Any]`、`environ: Mapping[str, str] | None`，返回新对象（immutable 风格，永不就地修改入参）。
- `_resolve_node(node, root, environ, prefix_stack, resolving)` — 遍历 `dict | list | tuple`；`prefix_stack: tuple[str, ...]` 记录"从 root 走到当前 mapping 的路径栈"，相对引用据此回溯。
- `_resolve_string(template, root, environ, prefix_stack, resolving)` — 对单个字符串叶子做 `re.sub`；负责循环检测（进入前 `add`，离开 `discard`，重入即 `InterpretError("circular reference at ...")`），其中循环 key 用进入时的 dotted path 表示。
- `_parse_placeholder(token)` — 接受 `${...}` 内的原始 ref body（不含 `${ }`），返回小型 NamedTuple `Placeholder(kind: Literal["env", "absolute", "relative"], path: tuple[str, ...], levels_up: int, default: str | None, raw: str)`；非法直接 `ParserError(raw)`。
- `_resolve_env(name, default)` — env 路由。
- `_resolve_absolute(root, parts, default)` / `_resolve_relative(root, prefix_stack, levels_up, parts, default)` — 引用解析；找不到 + 无 default → `InterpretError`，找不到 + 有 default → 返回 default literal。

模块顶部使用 `from __future__ import annotations`、PEP 604 unions、`collections.abc.Callable / Mapping`。`_INTERP_RE` 仍保留 `\$\{([^}]+)\}` 的整体边界形状，但内部解析下沉给 `_parse_placeholder`。

### Public API 重布线

`src/molcfg/__init__.py` 调整：

- `from molcfg.concurrency import FileLock, ThreadSafeConfig, interpolate` 拆成两行：
  - `from molcfg.concurrency import FileLock, ThreadSafeConfig`
  - `from molcfg.interpolation import interpolate`
- `from molcfg.errors import (...)` 删除 `CircularReferenceError`，新增 `InterpretError`、`ParserError`。
- `__all__` 中删除 `"CircularReferenceError"`，新增 `"ParserError"`、`"InterpretError"`。`interpolate` 仍归在已有的 `# Concurrency` 分组内，**保留原有分组顺序与注释** —— 这是一个有意决定：`interpolate` 已是 user-facing 名字，单独建一个 `# Interpolation` 分组会改变 `__init__.py` 的视觉布局并扩散 diff，价值不抵噪声；模块物理位置已经迁移，分组注释属于内部记号，不影响 import path。

### 决议记录

- 顶层 `${.foo}` 视为绝对引用而非报错（见上文规则 6），理由是与"N dots = climb (N-1) levels"规则保持闭环 —— 1 个点等于上爬 0 层，恰好就是绝对解析。
- `interpolate` 留在 `# Concurrency` 分组（见上文 Public API 重布线说明）。

## Files to create or modify

- `src/molcfg/interpolation.py` (new) — 新插值引擎模块；公开 `interpolate`，私有 `_resolve_node` / `_resolve_string` / `_resolve_absolute` / `_resolve_relative` / `_resolve_env` / `_parse_placeholder` / `_INTERP_RE`。
- `tests/test_interpolation.py` (new) — 完整 grammar / resolver 测试矩阵。
- `src/molcfg/errors.py` (edit) — 新增 `ParserError`、`InterpretError`；删除 `CircularReferenceError`。
- `src/molcfg/concurrency.py` (edit) — 删除 `_INTERP_RE`、`interpolate`、`_interpolate_dict`、`_resolve_string`、`_get_nested`；删除 `from molcfg.errors import CircularReferenceError`；模块 docstring 改为 `"""Thread-safe config wrapper and file locking."""`。
- `src/molcfg/__init__.py` (edit) — 拆 `interpolate` 的 import 来源；errors import 替换；`__all__` 更新。
- `tests/test_concurrency.py` (edit) — 删除 `class TestInterpolation` 整段（7 个 `test_*`）；删除模块顶部 `from molcfg.errors import CircularReferenceError`；模块 docstring 顺手收紧为 `"""Tests for thread safety and file locking."""`。

## Tasks

- [ ] Write failing tests for `_parse_placeholder` and the parser layer in `tests/test_interpolation.py` — 覆盖 well-formed token 解析为 `Placeholder` 元组、以及 `${}`、`${ , foo}`、`${env:}`、`${,}`、纯空白 ref 等 malformed token 触发 `ParserError`.
- [ ] Write failing tests for the resolver layer in `tests/test_interpolation.py` — env / 绝对 / 相对 (`.` / `..`) 各形态 happy path、四类 default fallback、missing without default 抛 `InterpretError`、`..` 爬过 root 抛 `InterpretError`、循环引用抛 `InterpretError("circular reference at ...")`、`dict / list / tuple` 容器递归（含 tuple 输出仍为 tuple）、字符串拼接 `${a.b}/data`、非字符串叶子原样保留、顶层 `${.foo}` 视作绝对引用.
- [ ] Add `ParserError(ConfigError)` and `InterpretError(ConfigError)` to `src/molcfg/errors.py`; delete `CircularReferenceError`.
- [ ] Implement `interpolate` and the private parser + resolver helpers in `src/molcfg/interpolation.py` so the failing tests above turn green; ensure zero internal `molcfg.*` imports beyond `molcfg.errors`.
- [ ] Refactor `src/molcfg/concurrency.py` — delete `_INTERP_RE`, `interpolate`, `_interpolate_dict`, `_resolve_string`, `_get_nested`, and the `CircularReferenceError` import; rewrite module docstring to `"""Thread-safe config wrapper and file locking."""`.
- [ ] Wire `src/molcfg/__init__.py` — split the `interpolate` import out of `molcfg.concurrency` and route it from `molcfg.interpolation`; replace `CircularReferenceError` with `InterpretError` and `ParserError` in both the errors import and `__all__`.
- [ ] Delete `class TestInterpolation` and the `from molcfg.errors import CircularReferenceError` line from `tests/test_concurrency.py`; tighten the module docstring to drop "interpolation".
- [ ] Add Google-style docstrings (with units / examples where applicable) on `interpolate`, `ParserError`, and `InterpretError`.
- [ ] Run full check + test suite: `ruff check . && ruff format --check . && pytest -q`.

## Testing strategy

Happy paths（`tests/test_interpolation.py`）：

- env 引用 `${env:HOME}` 命中 `environ` 返回字符串值。
- 绝对引用 `${a.b.c}` 从 root 下钻到字符串叶子。
- 相对 sibling `${.foo}` 在嵌套 mapping 中解析到同级 key。
- 相对父级 `${..foo}` 上爬一层后下钻。
- 拼接：`${env:HOST}:${port}` 与 `${a.b}/data` 在同一字符串中混用。
- 容器递归：`list[str]`、`tuple[str, ...]`、嵌套 `dict[str, list[dict[str, str]]]` 全部触达字符串叶子；非字符串叶子（`int`、`bool`、`None`）原样保留；tuple 输入返回 tuple。
- 链式引用：`x → y → z` 的多跳解析仍正确。

Default fallback（四条等价矩阵，每种 ref 形态 1 条）：

- `${env:NOPE, fallback}` → `"fallback"`（且不污染 `os.environ`）。
- `${missing.path, fallback}` → `"fallback"`。
- `${.missing, fallback}` → `"fallback"`。
- `${..missing, fallback}` → `"fallback"`。
- 逗号前后空白 strip：`${a.b ,  fallback }` 与 `${a.b, fallback}` 行为等价。
- default 不二次插值：`${missing, ${other}}` 以 literal `"${other}"` 落地（spec 不支持嵌套）。

Edge cases / 错误：

- `${}`、`${ , foo}`、`${env:}`、`${,}`、纯空白 ref → `ParserError`，错误消息含原 token。
- `${nonexistent}`（无 default）→ `InterpretError("missing config reference: 'nonexistent'")`。
- `${env:NOPE}`（无 default）→ `InterpretError("missing env var: 'NOPE'")`。
- 在 root 层使用 `${..foo}` → `InterpretError`，消息含 `escaped root`。
- 循环引用 `{"a": "${b}", "b": "${a}"}` → `InterpretError`，消息含 `"circular"` 字样。
- 顶层 `${.foo}` → 视作绝对引用（不抛错），有显式测试钉死该决定。

Public API / 集成：

- `from molcfg import interpolate, ParserError, InterpretError` 三者全部成功；三者均出现在 `molcfg.__all__`。
- `from molcfg import CircularReferenceError` 抛 `ImportError`；`from molcfg.errors import CircularReferenceError` 同样抛 `ImportError`。
- `tests/test_concurrency.py` 在删除 `TestInterpolation` 后剩余测试（`TestThreadSafeConfig` + `TestFileLock`）依旧全绿。
- 全仓 gate：`ruff check . && ruff format --check . && pytest -q` 一次性通过。

## Out of scope

- **转义机制**：spec 明确"没有转义"；config 中无法表达 literal `${...}`，亦不支持反斜杠或重复符号转义。
- **default 内的二次插值**：`${ref, ${other}}` 不递归解释，default 永远是 literal 文本。
- **default 内对 `,` / `}` 的转义**：`split-on-first-comma` 一刀切；如果 default 文本本身需要含 `,` 或 `}`，请改用配置外部填充。
- **list / tuple 中非字符串叶子的插值**：`int / bool / None / float` 等原样保留；只有 `str` 叶子触发解析。
- **JSON / TOML / YAML loader 隐式 `interpolate`**：source loader 维持纯解析；`interpolate` 仍是显式调用，绝不在 `Config.load_*` 内部隐式触发。
- **`docs/concurrency.md` 的内容迁移**：当前 `docs/concurrency.md` 中可能包含插值章节；本 spec 不强制在实施期同步重写文档，作为 follow-up 由 `/mol:note` 之后的文档迭代处理（实施时如顺手可一并改，但不计入 acceptance）。
- **蓝图 / `CLAUDE.md` 同步刷新**：实施完成后由 user 手动跑 `/mol:map` 刷新蓝图（`concurrency` 与 `errors` 两条蓝图条目的 layer-role / public-surface 描述会过期）；不在本 spec 的 task list 内。

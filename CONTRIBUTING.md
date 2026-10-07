# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you want to work on the documentation site:

```bash
pip install -e ".[docs]"
```

## Local Checks

Run lint and tests before opening a pull request:

```bash
ruff check .
pytest -q
```

If documentation dependencies are installed, preview docs locally:

```bash
zensical serve
```

## CI

One workflow per kind of work. A *feature* ref is any branch other than
`dev`/`master`/`main`; an *integration* ref is one of those, or a pull request
into one. A pull request from a branch of this repository does not re-run
what its push already ran: lint and docs never, the full test tier only when
the head is a feature branch (its push ran the fast tier).

| workflow | feature branch (fork or MolCrafts) | integration ref (fork or MolCrafts) | MolCrafts only |
|---|---|---|---|
| `lint.yml` | `lint / hooks` (pre-commit stage, all files) | same | — |
| `test.yml` | `test / py3.12 (ubuntu-latest)` | `test / py{3.12,3.14} ({ubuntu,macos,windows}-latest)` | — |
| `docs.yml` | `docs / build` (`zensical build --strict`) | same | deploy: Cloudflare Pages, outside Actions |
| `release.yml` | — | — | `v*` tag: lint + test + `release / build` + `release / pypi`; `workflow_dispatch` = dry run (no upload) |

The `protect-master` ruleset on `master` requires a pull request, blocks force
pushes and deletion, and requires the integration-tier `lint /`, `test /` and `docs /` checks.

## Change Expectations

- Keep the runtime package dependency free.
- Add tests for behavior changes and bug fixes.
- Preserve backwards compatibility for public APIs unless the change is explicitly scheduled for a major release.
- Update `README.md` and `docs/` when behavior or public interfaces change.
  Release history lives in git tags / GitHub Releases (no `CHANGELOG.md`).

## Pull Requests

- Keep PRs scoped to one logical change.
- Describe user-visible impact clearly.
- Include migration notes when behavior changes.

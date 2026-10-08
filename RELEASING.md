# Releasing

This repository uses a static version in `pyproject.toml`.

## PyPI Trusted Publisher

This repository is set up to publish through PyPI trusted publishing from GitHub Actions.
No PyPI API token should be stored in GitHub secrets for the normal release flow.

Configure the trusted publisher in PyPI with:

- Project name: `molcrafts-molcfg`
- Owner: `MolCrafts`
- Repository: `molcfg`
- Workflow: `release.yml`
- Environment: `pypi`

The GitHub repository must also have an environment named `pypi`.

## Release Checklist

1. Ensure `pyproject.toml` has the intended version.
2. Run the full test suite:

   ```bash
   pytest -q
   ```

3. Build distributions:

   ```bash
   python -m build
   ```

4. Verify artifacts:

   ```bash
   python -m twine check dist/*
   ```

5. Tag the release:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

6. Wait for the `release` workflow: it re-runs lint and the tests on the tag,
   checks the tag against `pyproject.toml`, builds, and publishes to PyPI via
   trusted publishing. Run it by hand (`workflow_dispatch`) for a dry run that
   builds without uploading.
7. Publish a GitHub release for the tag; draft notes from `git log` since the
   previous tag. History lives in git and the GitHub release — this repo
   keeps neither a `CHANGELOG.md` nor a release-notes page.

## Documentation Release

If documentation dependencies are installed:

```bash
zensical build
```

Deploy the generated `site/` directory with Cloudflare Pages.

Recommended Cloudflare Pages settings:

- Framework preset: `None`
- Build command: `uv run zensical build`
- Build output directory: `site`

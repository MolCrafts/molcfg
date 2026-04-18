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
2. Update `CHANGELOG.md` with the release date and notable changes.
3. Run the full test suite:

   ```bash
   pytest -q
   ```

4. Build distributions:

   ```bash
   python -m build
   ```

5. Verify artifacts:

   ```bash
   python -m twine check dist/*
   ```

6. Tag the release:

   ```bash
   git tag v1.3.0
   git push origin v1.3.0
   ```

7. Wait for the `Release` GitHub Actions workflow to publish the artifacts to PyPI via trusted publishing.
8. Publish a GitHub release for the tag and paste the matching changelog entry into the release notes.

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

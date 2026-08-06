# DATADOC 0.4.0 release checklist

This checklist is for maintainers publishing a new open-source release. The repository must be pushed and GitHub Actions must pass before uploading to PyPI.

## Local verification

```bash
python -m pytest -q
python -m ruff check datadoc tests
python -m ruff format --check datadoc tests
python -m compileall -q datadoc
uv lock --check
uv build
```

From `web/`:

```bash
npm ci
npm run build
```

Confirm that `dist/` contains exactly the intended `datadoc_cli-0.4.0.tar.gz` and `datadoc_cli-0.4.0-py3-none-any.whl` artifacts. Inspect the archives to ensure generated datasets, secrets, local environments, and scratch files are absent.

## GitHub release flow

1. Review the diff and stage only source, tests, docs, lockfile, and packaging changes.
2. Commit with a message such as `release: prepare datadoc 0.4.0`.
3. Push the branch and open a pull request.
4. Wait for the Python 3.10–3.12, operating-system, package, and frontend jobs to pass.
5. Merge the pull request into `main`.
6. Create the matching Git tag:

```bash
git tag -a v0.4.0 -m "DATADOC 0.4.0"
git push origin v0.4.0
```

## PyPI upload

Use a trusted publishing workflow or a short-lived API token. Do not store the token in the repository or shell history.

```bash
python -m pip install --upgrade twine
python -m twine check dist/*
python -m twine upload dist/datadoc_cli-0.4.0*
```

After publishing, verify installation in a clean environment:

```bash
python -m venv /tmp/datadoc-release-check
/tmp/datadoc-release-check/bin/python -m pip install datadoc-cli==0.4.0
/tmp/datadoc-release-check/bin/datadoc --help
```

The release is complete only when the PyPI page, GitHub tag, README installation instructions, documentation site, and package metadata all show the same version.

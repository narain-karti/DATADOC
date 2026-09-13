# Release Checklist (PyPI)

Use for every `datadoc-cli` release. Current release: **0.6.0**.

## 1. Pre-flight (repo must be green)

```bash
ruff check datadoc/ tests/
ruff format --check datadoc/ tests/
python -m pytest tests/ -q
uv lock --check
```

CI (`.github/workflows/ci.yml`, Python 3.10–3.12) must pass on `main`:
lint, format check, full test suite, CLI smoke.

## 2. Version sweep (single source: `pyproject.toml`)

1. Bump `version` in `pyproject.toml` **and** `__version__` in `datadoc/__init__.py` (keep identical).
2. Add a `CHANGELOG.md` entry (Added / Changed / Fixed).
3. Docs badges/footers under `docs/*.html` must show the new version.
4. `uv lock` to re-sync `uv.lock`, then `uv lock --check`.

## 3. Dashboard bundle (required for the `ui` extra)

The PyPI wheel ships the built frontend. Rebuild from a clean tree:

```bash
cd web
npm ci
npm run build
cd ..
```

Verify `web/dist/index.html` + `web/dist/assets/` are fresh. The wheel
force-includes `web/dist` as `datadoc/_webui`; `datadoc/cli/ui_server.py`
serves the installed bundle first and falls back to the checkout path.

## 4. Build and verify artifacts locally

```bash
uv build
python -m twine check dist/*
```

Install the wheel in a **clean** virtualenv and prove the entry points work:

```bash
python -m venv /tmp/dd-verify && /tmp/dd-verify/bin/python -m pip install "dist/datadoc_cli-<ver>-py3-none-any.whl" "dist/datadoc_cli-<ver>.tar.gz"
datadoc version
datadoc plugins list
datadoc profile demo.csv --target churn --explain
```

With extras: `pip install "dist/...whl[ml,ui]"`, then run one
`datadoc evaluate … --target …` and boot `datadoc ui demo.csv --no-browser`
and hit `/api/dataset/metadata` (expect 200).

## 5. Tag and publish

```bash
git tag v0.6.0 && git push origin main v0.6.0
python -m twine upload dist/datadoc_cli-0.6.0*
```

Never commit a PyPI token. Prefer trusted publishing; otherwise use a
short-lived token from the environment.

## 6. Post-publish

- Confirm the PyPI page renders (`README.md` is the long description).
- Open the docs site and spot-check version badges.
- Keep `pipeline.json` artifact compatibility in mind: `DataDocPipeline.load`
  accepts v1 and v2; bump `ARTIFACT_VERSION` only with a migration note.

<!--
===================================================================
REDDIT R/OPENSOURCE SUBMISSION FILE
===================================================================

COMMUNITY: r/opensource (~130k members, friendly to OSS authors asking
for feedback; no hard karma gate for well-written text posts)
TYPE: Text Post

TITLE TO COPY:
I open-sourced a CLI that freezes train-only preprocessing state into an
auditable pipeline.json (Polars-based, MIT) — feedback wanted on the format

Everything below the divider is the POST BODY to copy & paste.
===================================================================
-->

Hi r/opensource,

I've been working on **DATADOC**, an MIT-licensed CLI + Python library for tabular ML dataset preparation, and I'd love feedback from people who think about open-source design more than I do.

**The problem it solves:** in most ML projects, preprocessing lives in ad-hoc pandas notebooks. The model in the notebook scores 88%, then loses several points in production — usually because imputation medians, categorical vocabularies and scaling stats were computed over the *whole* dataset instead of the training split (silent data leakage), or because the state lives in a fragile pickle file.

**The core design decision I want feedback on:** everything the pipeline learns is serialized into a plain, inspectable **`pipeline.json`**:

```json
{
  "version": "0.6.0",
  "target": "churn",
  "steps": [
    { "plugin": "MissingValuePlugin", "priority": 10,
      "state": { "age__median": 34.5, "city__mode": "Bangalore" } },
    { "plugin": "OutlierPlugin", "priority": 20,
      "state": { "fare__lower": 0.0, "fare__upper": 65.6 } }
  ]
}
```

The rationale: JSON is diffable in git (a reviewer sees exactly which median changed between model versions), parseable from non-Python services, and carries zero arbitrary-code-execution risk compared to pickle. The trade-off is no NumPy/pickle object fidelity — every transform must be expressible as data, which constrained how plugins are written (a priority-ordered registry of 9 built-ins, extendable via entry points).

Repo: https://github.com/narain-karti/DATADOC
Docs: https://narain-karti.github.io/DATADOC/
Install: `pip install datadoc-cli`

Quick taste of the workflow:

```bash
datadoc health train.csv --target churn      # 0-100 quality score
datadoc fit train.csv --target churn --preset balanced
datadoc transform test.csv --pipeline artifacts/pipeline.json --validate
```

It's built on Polars (Apache Arrow), fully offline, zero telemetry, 110 automated tests.

What would make `pipeline.json` a format you could trust in production? Anything missing — versioning/compatibility fields, checksums, plugin provenance? Would you diff it differently? Any licensing/structure mistakes you see for an OSS project trying to grow contributors? Brutal feedback welcome.

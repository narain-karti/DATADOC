<!--
===================================================================
REDDIT R/SIDEPROJECT SUBMISSION FILE
===================================================================

COMMUNITY: r/SideProject (700k+ members, built for showing projects
you made — most promo-tolerant large sub, no hard karma gate)
TYPE: Text Post (attach assets/benchmark_comparison.jpg if posting manually)

TITLE TO COPY:
I built an open-source CLI that stops silent data leakage in ML preprocessing

Everything below the divider is the POST BODY to copy & paste.
===================================================================
-->

Every ML project I've shipped had the same ghost: validation scores looked great in the notebook, then dropped the moment the model saw real data. The culprit was almost always my own preprocessing — imputing missing values and scaling features *before* splitting train/test, which quietly leaks test statistics into training.

So I built **DATADOC**, an open-source CLI + Python library that makes leakage-safe preprocessing automatic:

- `datadoc health data.csv` → instant 0–100 data quality score + letter grade
- `datadoc fit train.csv --target churn` → learns medians, categorical vocab and outlier bounds **only from the training split**
- `datadoc transform test.csv --pipeline pipeline.json` → applies the exact frozen state to unseen data, with schema validation
- `datadoc report data.csv` → a standalone interactive HTML audit report you can share

A few choices I made that people here might find interesting:

- It runs on **Polars** (Apache Arrow), so it stays fast on multi-million-row CSVs.
- All learned state goes into a human-readable **`pipeline.json`** instead of a pickle blob — you can diff it in git, load it in any language, and there's no arbitrary code execution risk.
- There's an optional AI advisor (OpenAI/Gemini/Ollama) that suggests feature hypotheses, but it's **strictly forbidden from executing code** — the deterministic engine does all the real work.

On Titanic with 5-fold CV, replacing naive pandas prep with the strict train-only pipeline gave +1.01% LogReg / +0.67% RF accuracy. Small numbers, but the real win is that your offline validation score finally matches production.

- Repo: https://github.com/narain-karti/DATADOC
- Docs: https://narain-karti.github.io/DATADOC/
- Install: `pip install datadoc-cli` (MIT, 110 tests, fully offline)

Would genuinely appreciate feedback from this community — especially on the `pipeline.json` artifact format and the CLI UX. Roast it 🙂

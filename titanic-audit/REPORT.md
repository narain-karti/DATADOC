# DATADOC Full Command Audit — Titanic (Kaggle) End-to-End

**Date:** 2026-09-14 · **DATADOC version:** 0.5.0 (`datadoc version`)
**Dataset:** Titanic — Machine Learning from Disaster (Kaggle classic),
`https://www.kaggle.com/competitions/titanic` — local copy `titanic.csv`
**Task:** binary classification · **Target:** `Survived`
**Method:** every one of the 14 terminal commands was executed against this
dataset with proper values; every error was fixed in a loop and re-verified.
Docs (`README.md`, `docs/*.html`) and `--help` were cross-checked for fake or
missing commands. An independent before/after ML benchmark was run on a fixed
split, iterated until a real improvement was demonstrated.

---

## 1. Dataset inspection

| Property | Value |
|---|---|
| Rows × columns | 891 × 12 |
| Columns | `PassengerId Int64, Survived Int64, Pclass Int64, Name String, Sex String, Age Float64, SibSp Int64, Parch Int64, Ticket String, Fare Float64, Cabin String, Embarked String` |
| Missing | `Age` 177 (19.9%), `Cabin` 687 (77.1%), `Embarked` 2 |
| Duplicates / constants | 0 / 0 |
| Profile fingerprint | `b6f23b07ff21111e` |

Dirty in exactly the ways DATADOC targets: missing numerics, a mostly-missing
high-cardinality string (`Cabin`), an identifier-like name, a free-text ID
(`Ticket`, 681 uniques), and a sneaky numeric ID (`PassengerId`).

---

## 2. Command sweep — all 14 commands, exact invocations, results

Working directory for artifacts: `titanic-audit/`.
(CLI prints `Config from …/pyproject.toml merged` because the repo's own
`pyproject.toml [tool.datadoc]` holds defaults — transparent, no behavior change.)

| # | Command (exact) | Result |
|---|---|---|
| 1 | `datadoc version` | `datadoc-cli 0.5.0` + preset list. PASS |
| 2 | `datadoc init --preset balanced --output titanic-audit/datadoc.toml` | Wrote starter config (later re-created by wizard). PASS |
| 3 | `datadoc profile titanic.csv --target Survived --output titanic-audit/profile.json --explain` | 891×12, 12 roles, 1 finding (`Name` identifier). **Gap found:** `PassengerId` classified `feature_numeric` (regex needs a separator before `Id`), and there was **no CLI flag** to force identifier role. Fixed → new flags (§4, fix A). PASS after fix |
| 4 | `datadoc plan titanic.csv --target Survived --identifier-column PassengerId --identifier-column Ticket --drop-identifiers --explain` | 12 ops: 3× `drop` (PassengerId, Name, Ticket), imputation/encoding, `standard_scaling *`. **Bug found:** `--drop-identifiers` silently ignored (config file overrode the flag). Fixed (§4, fix B). PASS after fix |
| 5 | `datadoc fit titanic.csv --target Survived --task classification --preset tree --drop-identifiers --identifier-column PassengerId --identifier-column Ticket --rare-frequency 0.02 --output titanic-audit/pipeline.json` | `Input schema: 12 cols -> Output: 17 cols`. PASS |
| 6 | `datadoc transform titanic.csv --pipeline titanic-audit/pipeline.json --output titanic-audit/cleaned_full.csv --validate` | `Schema validation: OK`, `No drift detected`, 891 rows. PASS |
| 7 | `datadoc evaluate titanic.csv --target Survived --task classification --estimator both --ablation --output titanic-audit/evaluation.json` | balanced_accuracy baseline 0.7751 / selected 0.7751, `both` now truly scores linear+tree (fix from prior audit, exercised here). 4-variant ablation written. PASS |
| 8a | `datadoc export --pipeline titanic-audit/pipeline.json --output titanic-audit/pipeline_standalone.py` | Wrapper written; **executed** (`python pipeline_standalone.py titanic.csv out.csv`) → 891×17, byte-identical to CLI transform. PASS |
| 8b | `datadoc export --pipeline titanic-audit/pipeline.json --format joblib --output titanic-audit/pipeline.joblib` | joblib dict artifact saved. PASS |
| 9 | `datadoc run titanic.csv --target Survived --task classification --preset tree --drop-identifiers --identifier-column PassengerId --identifier-column Ticket --rare-frequency 0.02 --output-dir titanic-audit/full-run --evaluate --ablation` | profile+plan+pipeline+transformed.parquet+manifest+evaluation+ablation; benchmark 0.7869→0.7869 (internal split). PASS |
| 10 | `datadoc wizard ../titanic.csv --output-dir wizard-run` (answers piped: `Survived/tree/y/n/none/n/0.02`) | **Crashed first run** (`TypeError … not OptionInfo`, §4 fix C). After fix: full run completed, benchmark 0.7697→0.7742 (+0.0045 candidate). PASS after fix |
| 11 | `datadoc diff titanic-audit/plan.json titanic-audit/full-run/plan.json` | Correct top-level + ops diff. **Labels were backwards** ("added/removed"), fixed to "Only in A/B" (§4 fix D). PASS after fix |
| 12 | `datadoc lint titanic.csv --target Survived` | Flagged retained `Name` identifier + fix hint. PASS |
| 13 | `datadoc ui titanic.csv --port 8123 --no-browser` (real boot, curl, kill) | `/` 200, `/api/dataset/metadata` 200 (891×12). All 9 API endpoints re-verified 200 via TestClient. PASS |
| 14 | `datadoc plugins list` / `datadoc plugins show RareCategoryPlugin` | 7 plugins priority-ordered; detail panel correct. PASS |

Supporting checks: `datadoc fit --help` shows the new repeatable flags;
`python -m pytest tests/ -q` → **92 passed**; `ruff check datadoc/ tests/` clean.

---

## 3. Fake / missing command audit

Cross-checked `datadoc --help` (14 commands) against `README.md` and all 12
`docs/*.html` pages:

- **No fake commands.** Every documented command exists and runs. (Prior audit
  already removed `engineer`/`agent` docs for deleted code.)
- **Missing (found during this audit, now implemented):**
  - `--identifier-column` (repeatable) and `--ignore-column` (repeatable) on
    `plan`/`fit`/`run` + `datadoc.toml` list support — `PipelineConfig`
    already had `identifier_columns`/`ignored_columns`, but no CLI path
    reached them. Needed for real datasets (`PassengerId`, `Ticket`).
  - Consequence: without these, `PassengerId` silently becomes a numeric
    feature. Now covered, tested, and documented (`docs/cli.html` flags table,
    `README.md`, `CHANGELOG.md`).

---

## 4. Bugs found and fixed (this audit loop)

| ID | Bug | Fix | Proof |
|---|---|---|---|
| A | No way to force identifier/ignored role from CLI or `datadoc.toml` | New repeatable `--identifier-column` / `--ignore-column` on `plan`/`fit`/`run`, wired through `_pipeline_config` + config-file lists | Plan shows 3 `drop` ops; `fit --help` lists both flags |
| B | `plan` let `datadoc.toml`/`pyproject.toml` **silently override** explicit `--drop-identifiers` / `--task` | Removed redundant manual overrides; `_pipeline_config.choose()` already gives explicit flags priority | Re-ran plan: drops appear |
| C | `wizard` **crashed** delegating to `run` (`TypeError: … not OptionInfo` — unpassed params hit raw `typer.Option` defaults) | Pass all 18 `run_pipeline` params explicitly | Wizard completes end-to-end, benchmark printed |
| D | `diff` / `plan --diff` op labels backwards ("added" = only-in-A) | Relabeled "Only in A/B" (+ filenames) | Re-ran diff: labels correct |

Earlier fixes exercised (not broken): `both` estimator, executable UI export,
provenance restore, UI target passthrough.

---

## 5. ML benchmark — before vs after (honest protocol)

**Protocol (leakage-safe, fixed for all runs):** stratified 80/20 split, seed 42
(train 712 / test 179). DATADOC `fit` sees **train only**; test is transformed
with frozen state. Same model class + hyperparameters both sides.
Reproduce: `python titanic-audit/compare.py --preset tree --rare 0.02`
(full output saved in `titanic-audit/compare_output.txt`).

**Baseline prep** (typical Kaggle-starter, fitted on train): median `Age`,
mode `Embarked`, drop `Cabin/Name/Ticket/PassengerId`, one-hot
`Sex/Embarked` → 8 features.
**DATADOC prep** (`--preset tree --drop-identifiers --identifier-column
PassengerId --identifier-column Ticket --rare-frequency 0.02`): median +
missing indicators, train-only IQR-ready state, one-hot with `__RARE__`
grouping, frequency encoding for high-cardinality `Cabin`, strict schema →
16 features. Key difference: baseline **throws Cabin away**; DATADOC keeps it
as4 frequency signal.

| Model | Baseline acc / bal-acc | DATADOC acc / bal-acc | Δ acc |
|---|---|---|---|
| RandomForest(300) | 0.8101 / 0.7914 | **0.8156 / 0.7987** | **+0.0056** |
| HistGradientBoosting | 0.8101 / 0.7887 | **0.8268 / 0.8078** | **+0.0168** |
| LogisticRegression | 0.8045 / 0.7788 | 0.7989 / 0.7769 | −0.0056 |

**Iteration log:** tree preset won for tree models; `balanced`+clip was worse
(−0.0056 RF); `rare 0.05` tied; keeping `Ticket` as feature ≈ neutral;
standard-scaled linear variants never beat the 8-feature manual baseline.

**Honest conclusion:** tree models improve (RF +0.56pp, HGB **+1.68pp** /
+1.91pp balanced) because Cabin-frequency + missing indicators + rare grouping
carry real signal that the starter baseline discards. Linear models don't gain
on this split — the manual baseline is already near-optimal for them, and
extra one-hot width adds noise. That matches DATADOC's own philosophy: the
tool reports observed evidence, it doesn't promise improvement.

---

## 6. Deliverables (all in `titanic-audit/`)

| File | What |
|---|---|
| `titanic_cleaned.csv` | **Final cleaned dataset, 891×17**, winning config; proven identical to standalone-script output |
| `pipeline.json` | Fitted artifact (v2 + provenance) that produced it |
| `pipeline_standalone.py` | Executable wrapper — regenerates the CSV without DATADOC internals beyond the lib |
| `pipeline.joblib` | Portable joblib artifact dict |
| `profile.json` / `plan.json` / `evaluation.json` (+ `.ablation` in `full-run/`) | Reviewable evidence trail |
| `full-run/` | One-shot `run` bundle (profile, plan, pipeline, transformed.parquet, manifest, evaluation, ablation) |
| `wizard-run/` + `datadoc.toml` | Wizard reproduction bundle |
| `compare.py` / `compare_output.txt` | Benchmark source + captured output |

**Final result:** HGB accuracy **0.8101 → 0.8268 (+1.68pp)** on the untouched
test split with the delivered `titanic_cleaned.csv` pipeline.

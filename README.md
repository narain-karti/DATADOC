# DATADOC: Master Planning Board

> **Tagline:** "The Open Source Operating System for Dataset Engineering."

Welcome to the DATADOC Master Planning Workspace. This document serves as the single source of truth for the entire project's architecture, design decisions, and implementation strategies before any code is written.

---

## 🎨 Board 1: Project Vision

DATADOC is an intelligent, modular Dataset Engineering Framework that orchestrates existing powerful libraries (Pandas, Polars, Scikit-learn, Featuretools) into a standardized, reusable workflow.

```mermaid
mindmap
  root((DATADOC))
    Core Philosophy
      Orchestrate, dont replace
      Modular and Extensible
      Explainable Rules
      Deterministic First
    Initial Release (v1.0)
      Python SDK Library
      Terminal CLI (Rich)
      Local Execution
      Rule Engine
    Long Term Ecosystem
      REST API
      Dashboard Web App
      Plugin Marketplace
      AI Orchestration Planner
      Enterprise Edition
      Cloud SaaS
    Ecosystem Integrations
      Pandas
      Polars
      Scikit-learn
      Feature-engine
```

---

## 🏛️ Board 2: System Architecture

The architecture is strictly separated into layers. The business logic lives exclusively in the **Core Engine**.

```mermaid
graph TD
    UserCLI["Terminal CLI (Typer / Rich)"]
    UserSDK["Python SDK (DATADOC)"]
    
    subgraph "DATADOC Core"
        Core["Core Engine"]
        RuleEng["Deterministic Rule Engine"]
        PluginMgr["Plugin Manager"]
        Config["Configuration & State"]
    end
    
    subgraph "Plugin Ecosystem"
        Plugins["Pre-processing Plugins"]
        Plugin1["Missing Values"]
        Plugin2["Encoders"]
        Plugin3["Outliers"]
        Plugin4["Feature Creation"]
    end
    
    subgraph "Base Libraries (The Muscle)"
        Pandas["Pandas / Polars"]
        Sklearn["Scikit-Learn"]
        FeatEng["Feature-engine"]
    end

    UserCLI --> UserSDK
    UserSDK --> Core
    Core --> RuleEng
    Core --> PluginMgr
    Core --> Config
    RuleEng --> PluginMgr
    PluginMgr --> Plugins
    Plugins --> Plugin1
    Plugins --> Plugin2
    Plugins --> Plugin3
    Plugins --> Plugin4
    Plugins --> Pandas
    Plugins --> Sklearn
    Plugins --> FeatEng
```

> [!TIP]
> **Design Decision:** The CLI interacts with the Python SDK, not the Core Engine directly. This ensures the SDK is robust and capable of everything the CLI can do.

---

## 📁 Board 3: Project Structure

An enterprise-grade repository structure designed for modularity and open-source contribution.

```mermaid
graph LR
    Root["datadoc/"]
    Root --> Pkg["datadoc/"]
    Root --> Tests["tests/"]
    Root --> Docs["docs/"]
    Root --> Examples["examples/"]
    Root --> Github[".github/"]
    
    Pkg --> Core["core/ (engine, rule_engine)"]
    Pkg --> Cli["cli/ (app, commands)"]
    Pkg --> Plugins["plugins/ (manager, base, standard/)"]
    Pkg --> Utils["utils/"]
    Pkg --> Config["config/"]
    Pkg --> Exceptions["exceptions/"]
```

---

## 🔌 Board 4: Plugin Architecture

Every preprocessing operation is an isolated plugin inheriting from a strict base class.

```mermaid
classDiagram
    class BasePlugin {
        <<Abstract>>
        +name: str
        +version: str
        +description: str
        +priority: int
        +supported_datatypes: list
        +dependencies: list
        +analyze(data) dict
        +recommend(analysis_results) list
        +apply(data, config) data
        +validate(data) bool
        +rollback(data) data
        +explain() str
        +estimate_runtime(data) float
    }

    class MissingValuePlugin {
        +strategies: list
        +analyze(data)
        +recommend(analysis_results)
        +apply(data, config)
    }
    
    class CategoricalEncoderPlugin {
        +analyze(data)
        +apply(data, config)
    }

    BasePlugin <|-- MissingValuePlugin
    BasePlugin <|-- CategoricalEncoderPlugin
```

> [!IMPORTANT]
> **Design Decision:** Plugins MUST implement `rollback()` and `explain()`. Explainability is a core pillar. We must be able to tell the user *why* an operation occurred and revert it if needed.

---

## ⚙️ Board 5: Feature Engineering (Rule Engine)

Version 1 uses a Deterministic Rule Engine to orchestrate plugins based on hardcoded best practices.

```mermaid
graph TD
    Start["Ingest Dataset"] --> Analyze["Run Plugin.analyze() across all Plugins"]
    Analyze --> RuleMissing{"Missing Values > 0?"}
    RuleMissing -- Yes --> MvRec["MissingValuePlugin.recommend()"]
    RuleMissing -- No --> RuleCard{"Categorical Columns?"}
    
    MvRec --> RuleCard
    
    RuleCard -- Yes --> EncRec["EncodingPlugin.recommend()"]
    RuleCard -- No --> RuleDate{"Datetime Columns?"}
    
    EncRec --> RuleDate
    
    RuleDate -- Yes --> DateRec["DatetimePlugin.recommend()"]
    RuleDate -- No --> RuleOutlier{"Outliers Detected?"}
    
    DateRec --> RuleOutlier
    
    RuleOutlier -- Yes --> OutRec["OutlierPlugin.recommend()"]
    RuleOutlier -- No --> BuildPipe["Construct Pipeline Plan"]
    
    OutRec --> BuildPipe
    BuildPipe --> Apply["Execute Pipeline"]
    Apply --> Validate["Validate Output Data"]
```

---

## 💻 Board 6: CLI Design

The CLI uses modern tools (e.g., Typer and Rich) for a beautiful, colorful, and highly readable terminal experience.

| Command | Description |
|---|---|
| `datadoc analyze` | Scans dataset, returns rich health report table |
| `datadoc recommend`| Outputs a list of suggested engineering steps |
| `datadoc engineer` | Automatically applies the best-practice pipeline |
| `datadoc compare` | Diff-like view between raw and engineered datasets |
| `datadoc pipeline` | Exports the generated pipeline as a standalone `.py` script |
| `datadoc report` | Generates a visual HTML/Markdown report |
| `datadoc plugin` | Lists, enables, or disables local plugins |

```mermaid
graph LR
    CLI["$ datadoc analyze train.csv"] --> Parser["CLI Parser (Typer)"]
    Parser --> Init["Initialize SDK"]
    Init --> Analyze["SDK.analyze()"]
    Analyze --> Format["Rich Table Formatter"]
    Format --> Output["Render Terminal Output"]
```

---

## 🐍 Board 7: Python SDK Interface

The SDK must feel elegant, chained, and highly intuitive.

```mermaid
sequenceDiagram
    participant User
    participant SDK as DATADOC
    participant Engine as CoreEngine
    participant PM as PluginManager

    User->>SDK: doc = DATADOC("train.csv")
    SDK->>Engine: load_data()
    User->>SDK: doc.diagnose()
    SDK->>Engine: execute_health_check()
    Engine->>PM: run_analyze()
    PM-->>SDK: health_metrics
    SDK-->>User: Return Health Score
    User->>SDK: doc.engineer()
    SDK->>Engine: trigger_rule_engine()
    Engine->>PM: apply_recommended_plugins()
    PM-->>SDK: transformed_dataset
    SDK-->>User: Return Clean Data
```

---

## 🔄 Board 8: Core Workflow

The end-to-end lifecycle of a dataset passing through DATADOC.

```mermaid
stateDiagram-v2
    [*] --> Ingestion
    Ingestion --> Profiling: Extract Metadata
    Profiling --> Diagnosis: Plugin Analysis
    Diagnosis --> Recommendation: Rule Engine triggers
    Recommendation --> UserReview: (Optional via CLI)
    UserReview --> Execution: Apply Plugins
    Recommendation --> Execution: (Auto Mode)
    Execution --> Validation: Post-Execution Checks
    Validation --> Export: Pipeline & Data
    Export --> [*]
```

---

## 🗺️ Board 9: Roadmap (Milestones)

| Milestone | Objectives | Estimated Time | Deliverables |
|---|---|---|---|
| **M1: Foundation** | Core Engine, Base Plugin Class, CLI skeleton | 2 Weeks | `datadoc` core package, CLI entrypoint |
| **M2: Profiler** | `analyze()`, `diagnose()` logic, Rich CLI tables | 2 Weeks | Health score, terminal analysis reports |
| **M3: Rule Engine** | Deterministic engine, Missing Value/Encoder plugins | 3 Weeks | First end-to-end automated `engineer()` workflow |
| **M4: Adv Features** | Outliers, Datetimes, Scaling, Pipeline Export | 3 Weeks | Exportable `.py` pipelines, expanded plugins |
| **M5: V1.0 Polish** | Docs, benchmarks, test coverage, release | 2 Weeks | PyPI Release, ReadTheDocs, GitHub Actions |

---

## 🐙 Board 10: GitHub & Open Source Strategy

- **Branching Strategy:** GitHub Flow (main is always deployable, feature branches for work).
- **Semantic Versioning:** Strict SemVer (`MAJOR.MINOR.PATCH`).
- **Issues & PRs:** Enforced templates (Bug Report, Feature Request, Plugin Proposal).
- **CI/CD Actions:**
  - `lint.yml` (Ruff, MyPy)
  - `test.yml` (Pytest matrix across Python 3.9 - 3.12, OS matrix)
  - `publish.yml` (Auto-publish to PyPI on GitHub Release)
- **Community:** 
  - `CONTRIBUTING.md` with explicit instructions on "How to build a Plugin".
  - Labels: `good first issue`, `plugin-idea`, `core-engine`.

---

## 🧪 Board 11: Testing Strategy

Every step is independently testable. AI should not be involved in the determinism of tests.

```mermaid
graph TD
    TestStrat["Testing Strategy"]
    TestStrat --> Unit["Unit Tests"]
    TestStrat --> Int["Integration Tests"]
    TestStrat --> E2E["End-to-End Tests"]
    
    Unit --> CoreT["Core Engine Logic"]
    Unit --> PluginT["Isolated Plugin Tests (Analyze/Apply)"]
    
    Int --> RuleT["Rule Engine Workflow"]
    Int --> DataT["Cross-Plugin Data Integrity"]
    
    E2E --> CLI["CLI Command Execution"]
    E2E --> SDK["Full SDK Pipeline Generation"]
```

---

## 🚀 Board 12: Future Roadmap (Ecosystem)

```mermaid
mindmap
  root((DATADOC v2.0+))
    Web UI
      Local Streamlit/React Dashboard
      Drag and drop pipeline editing
    REST API
      FastAPI Microservice wrapping Core
    AI Planner
      LLM replaces Rule Engine
      Dynamic strategy generation
    Enterprise
      Cloud Data Warehouse Integrations
      Role-based Access
      Dataset Registry
```

---

## 🛒 Board 13: Plugin Marketplace Architecture

In future versions, plugins are decoupled from the core repository.

```mermaid
graph TD
    CLI["CLI: datadoc plugin install imbalanced-learn"] --> Registry["Marketplace Registry (JSON/API)"]
    Registry --> Pypi["PyPI Package Download"]
    Pypi --> Local["Install to local Virtual Environment"]
    Local --> PluginMgr["Core Plugin Manager auto-discovers"]
```

> [!NOTE]
> **Design Decision:** The marketplace will act as a curated index pointing to PyPI packages prefixed with `datadoc-plugin-`. This leverages existing Python infrastructure (pip/uv).

---

## 🧠 Board 14: AI Planner (Future Vision)

When the project matures, the deterministic rule engine is swapped for an LLM planner, **but the plugins remain unchanged**.

```mermaid
graph TD
    Data["Dataset"] --> Meta["Extract Metadata (Schema, Stats)"]
    Meta --> Prompt["Compile Prompt with Available Plugins"]
    Prompt --> LLM["LLM (GPT/Claude/Gemini)"]
    LLM --> JSON["Output JSON Execution Plan"]
    JSON --> Core["Core Engine Executes Plan"]
    Core --> CleanData["Engineered Dataset"]
```

---

## 🪜 Board 15: Implementation Sequence

> **Crucial Rule:** Never build everything at once. Build sequentially, ensuring a working, testable product at each step.

1. **Step 1: The Skeleton** 
   - Set up `pyproject.toml`, Ruff, Pytest.
   - Create `DATADOC` base class (no logic, just loading data into Pandas).
2. **Step 2: Plugin Architecture Foundation**
   - Create `BasePlugin` abstract class.
   - Create `PluginManager` that registers a dummy plugin.
3. **Step 3: CLI Scaffolding**
   - Implement `Typer` app with a dummy `analyze` command that prints "Hello from DATADOC".
4. **Step 4: The Profiler & Health Metrics**
   - Implement basic `analyze()` in the Core. Calculate missing value %, data types.
   - Connect CLI to print Rich tables.
5. **Step 5: The First Real Plugin**
   - Implement `MissingValuePlugin` (Mean/Median imputation).
6. **Step 6: The Rule Engine V1**
   - Implement basic IF statements in Core to trigger `MissingValuePlugin` if missing values are found.
7. **Step 7: The "Fix" Workflow**
   - Wire SDK `doc.fix()` to execute the Rule Engine. 
   - Wire CLI `datadoc fix` to output the result.
8. **Step 8: Expanding the Arsenal (Parallelizable)**
   - Add `CategoricalEncoderPlugin`.
   - Add `OutlierPlugin`.
9. **Step 9: Pipeline Generation**
   - Implement logic to record plugin actions and export them to a `.py` file (`datadoc pipeline`).
10. **Step 10: Polish & Release v0.1.0**
    - Documentation, examples, and CI/CD pipelines.

---
*End of Master Planning Board.*

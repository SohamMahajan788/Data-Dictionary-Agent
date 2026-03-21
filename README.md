# Data Dictionary Agent

> Turn any CSV folder, SQLite database, or PostgreSQL/MySQL connection into a documented, profiled, relationship-mapped data dictionary with optional AI summaries and PII awareness.

## What It Does

Data teams routinely inherit databases with little or no documentation. Manual dictionary work does not scale, and static exports go stale the moment schemas change. This project is an **agentic pipeline** that ingests heterogeneous sources through a single orchestration path, reflects real schema and sample statistics, and emits machine- and human-readable artifacts for governance, onboarding, and hackathon demos.

The **Data Dictionary Agent** normalises inputs behind SQLAlchemy: CSV folders are materialised into an in-memory SQLite engine; `.sqlite`/`.db` files are attached directly; PostgreSQL and MySQL are reached via connection URIs. On top of that engine, a sequence of specialised **agents** extracts structure, infers relationships where FKs are missing, profiles completeness and distributions, flags likely PII, and (when configured) calls an LLM to produce business-oriented narratives. Everything merges into one **`DictionaryResult`** that serialises to JSON and drives an HTML report.

The same pipeline powers a **Streamlit** console for analysts and a **FastAPI** surface for automation. A **VersionStore** records lightweight snapshots per source so you can compare runs and surface schema drift, row-count shifts, and completeness changes—useful for CI-style checks or operational reviews.

The codebase is modular by design: connectors and agents are small, typed modules with consistent `structlog` instrumentation, making it straightforward to swap models, add connectors, or tighten PII rules without rewriting the orchestrator.

## Architecture

```
INPUT LAYER
CSV Folder / SQLite File / PostgreSQL / MySQL
         |
         v
CONNECTOR LAYER
csv_loader.py | sqlite_connector.py | postgres_connector.py
         |
         v
AGENT PIPELINE (orchestrator/pipeline.py)
    |
    |-- [1] Schema Extractor      --> tables, columns, PKs, FKs
    |-- [2] Relationship Mapper   --> FK graph, implicit detection
    |-- [3] Data Profiler         --> null rates, completeness, stats
    |-- [4] PII Detector          --> rule-based + Presidio AI scan
    |-- [5] LLM Summariser        --> Groq/Claude business descriptions
         |
         v
OUTPUT LAYER
HTML Report | JSON Result | PDF (Linux/Mac)
         |
         v
INTERFACES
Streamlit UI (port 8501) | FastAPI REST (port 8000)
         |
         v
STORAGE
VersionStore --> schema drift detection across runs
```

| Layer | Role |
|--------|------|
| **Input** | User or API supplies a folder path, SQLite file, or database URI. |
| **Connectors** | Load or attach data so downstream code always sees a SQLAlchemy `Engine` (in-memory SQLite for CSV bundles). |
| **Agent pipeline** | Ordered stages produce schema, relationships, profiles, PII report, and optional LLM summaries. |
| **Output** | Jinja2-rendered HTML, full JSON on disk, optional PDF where WeasyPrint native deps are available. |
| **Interfaces** | Streamlit for interactive runs; FastAPI for programmatic execution and read APIs. |
| **Storage** | `VersionStore` persists snapshots and drift history in JSON for a given `source_path`. |

## Tech Stack

| Layer | Technology | Purpose |
|--------|------------|---------|
| Data | **pandas**, **SQLAlchemy** | CSV ingestion, typing, and portable SQL access |
| Profiling | **pandas** (+ SQL `LIMIT` samples) | Per-column nulls, uniqueness, stats, table completeness |
| PII | **presidio-analyzer**, **spaCy** (`en_core_web_lg`) | Rule-based column naming + NER on sample values |
| LLM | **Groq** API (Llama 3.3), **python-dotenv** | Table/database business narratives; optional caching via settings |
| Relationships | **networkx** | Directed graph of explicit and inferred FK-style edges |
| Output | **Jinja2**, **WeasyPrint** (optional) | HTML report and PDF when platform libraries allow |
| API | **FastAPI**, **uvicorn**, **Pydantic** | REST endpoints for runs, snapshots, drift, tables, PII, reports |
| UI | **Streamlit** | Upload/path-driven runs, metrics, downloads |
| Infrastructure (planned / optional) | **Celery**, **Redis** (listed in planning) | Future async job execution |
| Logging | **structlog** | Structured, consistent observability across modules |

## Project Structure

```text
data-dictionary-agent/
├── PLANNING.md                 # Architecture notes and build order
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package metadata; enables editable install
├── .env                        # Local secrets (not committed)
├── .env.example                # Template for required/optional env vars
├── .gitignore                  # VCS ignore rules
│
├── config/
│   ├── __init__.py             # Package marker
│   └── settings.py             # Loads .env; shared defaults (paths, LLM flags)
│
├── connectors/
│   ├── __init__.py             # Package marker
│   ├── csv_loader.py           # Folder of CSVs → in-memory SQLite engine
│   ├── sqlite_connector.py     # Validates and opens .sqlite / .db files
│   ├── postgres_connector.py   # Placeholder for dedicated PG wiring (URIs used in pipeline today)
│   └── sql_loader.py           # Placeholder for future .sql dump → SQLite loader
│
├── agents/
│   ├── __init__.py             # Package marker
│   ├── schema_extractor.py     # SQLAlchemy inspect → DatabaseSchema dataclasses
│   ├── relationship_mapper.py  # Explicit FKs + heuristics → RelationshipMap + networkx
│   ├── data_profiler.py        # Sampled profiles → DatabaseProfile
│   ├── pii_detector.py         # Rules + Presidio → PIIReport
│   └── llm_summariser.py       # Groq chat completions → DatabaseSummary / TableSummary
│
├── orchestrator/
│   ├── __init__.py             # Package marker
│   └── pipeline.py             # run_pipeline, InputType, DictionaryResult, JSON export
│
├── output/
│   ├── __init__.py             # Package marker
│   ├── report_generator.py     # Jinja2 render + optional WeasyPrint PDF
│   ├── templates/
│   │   └── dictionary.html     # Data dictionary report template
│   ├── generated/             # Timestamped HTML, JSON, PDF from runs (artifacts)
│   └── chinook_summary.json   # Example LLM export from development
│
├── ui/
│   ├── __init__.py             # Package marker
│   └── app.py                  # Streamlit UI: run pipeline, tabs, downloads
│
├── api/
│   ├── __init__.py             # Package marker
│   └── routes.py               # FastAPI app: health, /run, snapshots, drift, tables, PII, report
│
├── storage/
│   ├── __init__.py             # Package marker
│   ├── versioning.py           # VersionStore, SchemaSnapshot, DriftReport
│   ├── version_store.json      # Persisted snapshots + drift (local state)
│   └── version_store_test.json # Optional test store file
│
├── tests/
│   └── __init__.py             # Test package marker (pytest)
│
└── sample_data/
    ├── olist/                  # 9 CSVs — Brazilian e-commerce (see dataset section)
    ├── bikestore/              # Multiple CSVs — retail bike store scenario
    └── chinook/
        └── Chinook_Sqlite.sqlite  # SQLite sample with explicit FKs
```

## Quick Start

### Prerequisites

- Python 3.10+
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd data-dictionary-agent
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**

   **Windows (PowerShell):**

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**

   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Install the spaCy language model** (required for Presidio’s NLP engine)

   ```bash
   python -m spacy download en_core_web_lg
   ```

6. **Install the package in editable mode**

   ```bash
   pip install -e .
   ```

7. **Configure environment variables**

   ```bash
   copy .env.example .env
   ```

   On macOS/Linux use `cp .env.example .env`. Edit `.env` and set at least **`GROQ_API_KEY`** for LLM summaries.

### Running the Streamlit UI

From the project root (with the virtual environment activated):

```bash
streamlit run ui/app.py
```

Open **http://localhost:8501** in your browser.

### Running the FastAPI

From the project root:

```bash
python api/routes.py
```

Or:

```bash
uvicorn api.routes:app --host 0.0.0.0 --port 8000
```

- **API base URL:** `http://localhost:8000`
- **Interactive docs:** `http://localhost:8000/docs` (Swagger UI)

## Testing With All 3 Datasets

### Dataset 1 — Olist Brazilian E-Commerce

| | |
|--|--|
| **Source** | Public Brazilian e-commerce dataset (often distributed via [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)) |
| **License** | **CC BY 4.0** (verify on the distribution you use) |
| **Format** | **9 CSV files** under `sample_data/olist/` (orders, customers, geolocation, products, sellers, reviews, etc.) |

**What it tests:** Tables loaded **without declared foreign keys** exercise **implicit relationship detection**, naming heuristics, and profiling at non-trivial row counts.

**How to run — UI:** Choose **CSV folder path**, enter the absolute path to `sample_data/olist`, then **Generate Dictionary**.

**How to run — pipeline (Python):**

```bash
python -c "from pathlib import Path; import sys; sys.path.insert(0, '.'); from orchestrator.pipeline import PipelineConfig, InputType, run_pipeline; p=Path('sample_data/olist').resolve(); run_pipeline(PipelineConfig(input_path=str(p), input_type=InputType.CSV_FOLDER, enable_llm=True, enable_pii=True))"
```

**Expected output summary:** Multiple tables, high relationship count from inferred edges, JSON under `output/generated/`, HTML report when `generate_report` is invoked (e.g. via UI or API `/run`).

### Dataset 2 — Bike Store

| | |
|--|--|
| **Source** | Common **SQL Server–style retail sample** (bike store); files provided under `sample_data/bikestore/` for local demos |
| **License** | Treat as **educational / sample data**; confirm terms on your upstream source if you redistribute |
| **Format** | **Multiple CSV files** (brands, categories, customers, orders, products, staffs, stocks, stores, …) |

**What it tests:** A **second CSV-folder scenario** in a different domain (retail inventory and orders), validating connector and profiler behaviour beyond Olist.

**How to run:** Same as Olist: point the UI or `PipelineConfig` `input_path` at `sample_data/bikestore` with `InputType.CSV_FOLDER`.

### Dataset 3 — Chinook Music Store

| | |
|--|--|
| **Source** | [Chinook database](https://github.com/lerocha/chinook-database) (SQLite build in repo) |
| **License** | Typically distributed under permissive terms (e.g. **Ms-PL** on upstream; confirm for your copy) |
| **Format** | Single **SQLite** file: `sample_data/chinook/Chinook_Sqlite.sqlite` |

**What it tests:** **Declared foreign keys**, smaller schema, and the **sqlite** connector path end-to-end—ideal for quick runs and LLM smoke tests.

**How to run — UI:** Select **SQLite / .db file**, upload or reference the Chinook file path (depending on your workflow).

**How to run — pipeline:**

```bash
python orchestrator/pipeline.py
```

(The module’s `__main__` block runs Chinook and Olist samples when executed from the project root with `PYTHONPATH` set appropriately, or use a one-liner with `PipelineConfig` and `InputType.SQLITE_FILE` as in the Olist example.)

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GROQ_API_KEY` | Authenticates Groq API calls for `llm_summariser` | *(none — required for LLM output)* |
| `ANTHROPIC_API_KEY` | Reserved / legacy; central `settings` reads it but summariser uses Groq | *(none)* |
| `SAMPLE_DATA_PATH` | Default root for sample data in tooling | `./sample_data` |
| `OUTPUT_PATH` | Default directory for generated artifacts | `./output/generated` |
| `LLM_MODEL` | Model name exposed via settings (summariser may use its own Groq model constant) | `claude-sonnet-4-6` |
| `MAX_SAMPLE_ROWS` | Upper bound hint for sampling configuration | `10000` |
| `LLM_CACHE_ENABLED` | When `true`, reuse cached table summaries within a process | `true` |

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness; returns version and snapshot count in `VersionStore` |
| `POST` | `/run` | Body: `RunPipelineRequest` — runs full pipeline, saves snapshot, renders HTML, returns `result_to_dict()` JSON |
| `GET` | `/snapshots` | Lists all snapshots (newest first): id, source, time, table count, schema hash |
| `GET` | `/snapshots/{snapshot_id}` | Full metadata for one snapshot |
| `GET` | `/drift/{source_path}` | Latest **DriftReport** for a source (404 if none); URL-encoded path segments |
| `GET` | `/tables/{source_path}/{table_name}` | Detailed table slice from latest saved pipeline JSON (schema, profile, rels, PII, LLM) |
| `GET` | `/tables/{source_path}` | List of `TableSummaryResponse` rows for the latest snapshot’s matching JSON |
| `GET` | `/pii/{source_path}` | PII report dict from the latest pipeline JSON for that source |
| `GET` | `/report/{source_path}` | `FileResponse` HTML download for the most recent report recorded for that source (after a `/run` that generated HTML) |

## Key Design Decisions

- **Universal connector normalisation** — Every input path ends as a SQLAlchemy `Engine`, so agents stay database-agnostic and testable on SQLite.
- **Bounded sampling** — Profiling uses per-table `LIMIT` samples (configurable `sample_size`) so large tables remain tractable without full scans.
- **Implicit FK detection** — CSV-style data gets relationship hints via column/table name heuristics and confidence scores, merged with explicit FKs from the RDBMS.
- **Dual PII detection** — Fast, explainable name rules run first; Presidio + spaCy validates cell samples when rules are inconclusive, with FK-aware suppression for obvious surrogate keys.
- **LLM provider flexibility** — Summaries are driven through a chat-completions style integration (Groq in code today); prompts and parsing are isolated in `llm_summariser.py` for swap or fallback behaviour.
- **Modular agents** — Each stage owns dataclasses and `*_to_dict` serializers; the orchestrator only sequences work and aggregates `DictionaryResult`.
- **Versioning as a separate concern** — `VersionStore` tracks structural and quality fingerprints over time without coupling drift logic to the core profiling agents.
- **Dual surfaces** — The same `DictionaryResult` feeds Streamlit, FastAPI, and on-disk JSON/HTML, avoiding duplicate business logic.

## What Makes It Enterprise-Grade

- **PII safety** — Explicit PII typing, rule transparency, and sample-based NLP with configurable thresholds help teams triage columns before wide publication.
- **Schema drift awareness** — Snapshots and drift reports highlight breaking removals and meaningful completeness or row-count shifts between runs.
- **Sampling for scale** — Default caps keep memory and runtime predictable while still reflecting real data distributions.
- **Modular, swappable agents** — New detectors, summarisers, or connectors can be added without rewriting the full pipeline contract.
- **Structured observability** — `structlog` events across connectors, agents, API middleware, and report generation support grep-friendly operations.
- **Downstream integration** — Canonical JSON output and REST endpoints let BI tools, catalogs, and CI jobs consume the same dictionary the UI displays.

## Sample Output

When you run `orchestrator/pipeline.py` (or equivalent), the CLI prints a summary similar to:

```text

=== Pipeline Summary ===
Input: C:\...\sample_data\chinook\Chinook_Sqlite.sqlite
Total tables: 11
Total columns: 67
Overall completeness score: 0.9997
Total relationships found: 12
Total PII columns found: 8
LLM summaries generated: True
Output file path: ...\output\generated\dictionary_result_sqlite_file_2026-03-21T05-34-45.140935+00-00.json
Pipeline duration: 42.18s
```

(Exact paths, counts, and duration depend on your machine, flags, and dataset.)

## License

MIT

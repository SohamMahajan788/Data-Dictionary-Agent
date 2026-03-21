# Data Dictionary Agent — Project Planning

## What this project does
An AI agent that accepts any database (CSV files, SQLite, PostgreSQL,
MySQL) and automatically generates a complete human-readable data
dictionary including:
- Schema documentation (tables, columns, types, keys)
- Relationship maps and ER structure
- Data quality metrics (nulls, completeness, freshness)
- AI-generated business context summaries via Claude API
- PII detection and flagging

## Architecture
INPUT → Universal Connector → Core Agent Pipeline → Output

### Connectors (connectors/)
- csv_loader.py — loads CSV files, infers schema, loads into SQLite
- sql_loader.py — executes .sql dump files into SQLite
- sqlite_connector.py — connects directly to .sqlite/.db files
- postgres_connector.py — connects to live PostgreSQL/MySQL databases
- All connectors return a SQLAlchemy engine pointing at SQLite

### Agents (agents/)
- schema_extractor.py — extracts tables, columns, types, PKs, FKs
- relationship_mapper.py — builds FK graph using networkx
- data_profiler.py — computes null rate, completeness, stats per column
- pii_detector.py — flags sensitive columns using presidio-analyzer
- llm_summariser.py — sends schema + samples to Claude API for summaries

### Orchestrator (orchestrator/)
- pipeline.py — runs all agents in order, merges into one output object

### Output (output/)
- report_generator.py — renders Jinja2 HTML template into PDF and HTML
- templates/dictionary.html — the report template

### API (api/)
- routes.py — FastAPI routes to query the dictionary programmatically

### UI (ui/)
- app.py — Streamlit app, the main user interface

### Storage (storage/)
- versioning.py — stores dictionary snapshots, diffs schema changes

### Config (config/)
- settings.py — loads .env variables, centralised config object

## Tech Stack
pandas, sqlalchemy, networkx, ydata-profiling, great-expectations,
presidio-analyzer, anthropic SDK, fastapi, streamlit,
jinja2, weasyprint, celery, redis, structlog, pytest

## Test Datasets
- sample_data/olist/       — 9 CSV files (Brazilian e-commerce)
- sample_data/bikestore/   — CSV files (retail bike store)
- sample_data/chinook/     — Chinook_Sqlite.sqlite file

## Build Order
1. connectors/csv_loader.py
2. connectors/sqlite_connector.py
3. agents/schema_extractor.py
4. agents/relationship_mapper.py
5. agents/data_profiler.py
6. agents/pii_detector.py
7. agents/llm_summariser.py
8. orchestrator/pipeline.py
9. output/report_generator.py + dictionary.html
10. ui/app.py
11. storage/versioning.py
12. api/routes.py
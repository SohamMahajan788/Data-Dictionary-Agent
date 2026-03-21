"""
Generate business-oriented table and database summaries via the Anthropic Claude API.

Consumes outputs from schema extraction, profiling, relationship mapping, and PII detection.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import structlog

from agents.data_profiler import ColumnProfile, DatabaseProfile, TableProfile
from agents.pii_detector import PIIFinding, PIIReport
from agents.relationship_mapper import Relationship, RelationshipMap
from agents.schema_extractor import ColumnInfo, DatabaseSchema, TableSchema
from config.settings import LLM_CACHE_ENABLED

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger = structlog.get_logger(__name__)

_UNAVAILABLE = "Auto-generated description unavailable"


@dataclass
class ColumnSummary:
    """LLM-generated business context for a single column."""

    column_name: str
    business_description: str
    suggested_business_name: str
    data_notes: str | None


@dataclass
class TableSummary:
    """LLM-generated summary for one table."""

    table_name: str
    business_description: str
    suggested_business_name: str
    business_purpose: str
    columns: list[ColumnSummary]
    data_quality_notes: str
    pii_warning: str | None


@dataclass
class DatabaseSummary:
    """Overall database narrative and per-table summaries."""

    database_description: str
    business_domain: str
    tables: list[TableSummary]
    generated_at: str
    model_used: str


_TABLE_SUMMARY_CACHE: dict[str, TableSummary] = {}


def _utc_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _get_table_profile(profile: DatabaseProfile, table_name: str) -> TableProfile | None:
    """Return :class:`TableProfile` for ``table_name`` if present."""
    for tp in profile.tables:
        if tp.table_name == table_name:
            return tp
    return None


def _relationships_for_table(
    rel_map: RelationshipMap,
    table_name: str,
) -> list[Relationship]:
    """Collect relationships where ``table_name`` is on either side of the edge."""
    out: list[Relationship] = []
    for r in rel_map.relationships:
        if r.from_table == table_name or r.to_table == table_name:
            out.append(r)
    return out


def _pii_for_table(pii_report: PIIReport, table_name: str) -> list[PIIFinding]:
    """Return PII findings scoped to ``table_name``."""
    return [f for f in pii_report.findings if f.table_name == table_name]


def _format_relationship_line(r: Relationship) -> str:
    """Single-line description of a relationship for prompts."""
    ex = "explicit" if r.is_explicit else "implicit"
    return (
        f"  - {r.from_table}.{r.from_column} -> {r.to_table}.{r.to_column} "
        f"({r.relationship_type}, {ex}, confidence={r.confidence:.2f})"
    )


def _format_pii_line(f: PIIFinding) -> str:
    """Single-line PII hint for prompts."""
    return (
        f"  - {f.column_name}: {f.pii_type} via {f.detection_method} "
        f"(confidence={f.confidence:.2f})"
    )


def build_table_prompt(
    table: TableSchema,
    profile: TableProfile,
    relationships: list[Relationship],
    pii_findings: list[PIIFinding],
) -> str:
    """
    Build the user prompt for summarising one table.

    Includes column names/types, row counts, completeness, relationships, and PII flags.
    Instructs the model to answer with **only** a JSON object of the required shape.

    Args:
        table: Reflected table schema.
        profile: Statistical profile for the same table.
        relationships: Edges touching this table.
        pii_findings: PII hits for this table.

    Returns:
        Prompt text for Claude.
    """
    col_lines = [
        f"  - {c.name}: {c.data_type} (nullable={c.is_nullable}, "
        f"PK={c.is_primary_key}, FK={c.is_foreign_key})"
        for c in table.columns
    ]
    prof_lines: list[str] = []
    for cp in profile.columns:
        prof_lines.append(
            f"  - {cp.column_name}: null_rate={cp.null_rate:.3f}, "
            f"uniqueness_rate={cp.uniqueness_rate:.3f}"
        )

    rel_block = (
        "\n".join(_format_relationship_line(r) for r in relationships)
        if relationships
        else "  (none detected)"
    )
    pii_block = (
        "\n".join(_format_pii_line(f) for f in pii_findings)
        if pii_findings
        else "  (none flagged)"
    )

    return f"""You are a data governance assistant. Analyse this database table and respond with **ONLY** valid JSON (no markdown fences, no commentary).

Table name: {table.table_name}
Approximate row count (from catalogue): {table.row_count}
Profiled sample rows: {profile.total_rows}
Table completeness score (1 - avg null rate): {profile.completeness_score:.4f}
Duplicate rows in sample: {profile.duplicate_row_count}
Has data-quality flag: {profile.has_data_quality_issues}

Columns (schema):
{chr(10).join(col_lines)}

Column statistics (sample):
{chr(10).join(prof_lines)}

Relationships involving this table:
{rel_block}

PII-related flags for this table:
{pii_block}

Return **only** a JSON object with exactly these keys:
- "business_description": string
- "suggested_business_name": string (human-friendly name)
- "business_purpose": string (what this table is for operationally)
- "data_quality_notes": string (reference nulls, duplicates, completeness briefly)
- "pii_warning": string or null (warn analysts about sensitive columns; null if none)
- "columns": array of objects, one per logical column, each with:
    - "column_name": string (must match an actual column name)
    - "business_description": string
    - "suggested_business_name": string
    - "data_notes": string or null (e.g. high null rate, constant value)

Use double quotes for all JSON strings."""


def build_database_prompt(
    schema: DatabaseSchema,
    summaries: list[TableSummary],
) -> str:
    """
    Build the prompt for the overall database narrative.

    Args:
        schema: Full :class:`DatabaseSchema` (for source path / scale).
        summaries: Completed per-table summaries.

    Returns:
        Prompt text requesting JSON with ``database_description`` and ``business_domain``.
    """
    compact = [
        {
            "table_name": s.table_name,
            "suggested_business_name": s.suggested_business_name,
            "business_purpose": s.business_purpose,
            "business_description": s.business_description,
        }
        for s in summaries
    ]
    ctx = json.dumps(compact, indent=2)
    return f"""You are a data governance assistant. Given per-table summaries, describe the whole database.

Source identifier: {schema.source_path}
Total tables (catalogue): {schema.total_tables}
Total columns (catalogue): {schema.total_columns}

Per-table summaries (JSON):
{ctx}

Respond with **ONLY** valid JSON (no markdown) with exactly:
{{
  "database_description": "<2-4 sentences on what this database represents>",
  "business_domain": "<short industry/domain label, e.g. Retail / Music retail / E-commerce>"
}}
Use double quotes for strings."""


def _extract_json_object(text: str) -> dict | None:
    """
    Parse the first JSON object from ``text``, stripping optional ``` fences.

    Args:
        text: Raw model output.

    Returns:
        Parsed dict or ``None`` if parsing fails.
    """
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n", s, re.IGNORECASE)
    if fence:
        s = s[fence.end() :]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _fallback_column_summary(col: ColumnInfo, cp: ColumnProfile | None) -> ColumnSummary:
    """Heuristic column summary when the LLM is unavailable."""
    notes: str | None = None
    if cp is not None and cp.null_rate > 0:
        notes = f"{cp.null_rate * 100:.0f}% null rate in sample — may be optional or sparse"
    return ColumnSummary(
        column_name=col.name,
        business_description=_UNAVAILABLE,
        suggested_business_name=col.name,
        data_notes=notes,
    )


def _fallback_table_summary(
    table: TableSchema,
    profile: TableProfile,
    pii_findings: list[PIIFinding],
) -> TableSummary:
    """Heuristic table summary when the LLM is unavailable."""
    pmap = {c.column_name: c for c in profile.columns}
    cols = [
        _fallback_column_summary(c, pmap.get(c.name)) for c in table.columns
    ]
    dq = (
        f"Completeness (sample avg): {profile.completeness_score:.3f}; "
        f"duplicate rows in sample: {profile.duplicate_row_count}; "
        f"quality flag: {profile.has_data_quality_issues}"
    )
    pii_warn: str | None
    if pii_findings:
        parts = [f"{f.column_name} ({f.pii_type})" for f in pii_findings]
        pii_warn = "Possible PII columns: " + ", ".join(parts)
    else:
        pii_warn = None
    return TableSummary(
        table_name=table.table_name,
        business_description=_UNAVAILABLE,
        suggested_business_name=table.table_name.replace("_", " ").title(),
        business_purpose=_UNAVAILABLE,
        columns=cols,
        data_quality_notes=dq,
        pii_warning=pii_warn,
    )


def _parse_column_summaries(raw_cols: object, table: TableSchema) -> list[ColumnSummary]:
    """Normalise the ``columns`` array from model JSON."""
    out: list[ColumnSummary] = []
    if not isinstance(raw_cols, list):
        return out
    known = {c.name for c in table.columns}
    for item in raw_cols:
        if not isinstance(item, dict):
            continue
        name = str(item.get("column_name", "")).strip()
        if name not in known:
            continue
        out.append(
            ColumnSummary(
                column_name=name,
                business_description=str(
                    item.get("business_description") or _UNAVAILABLE
                ),
                suggested_business_name=str(
                    item.get("suggested_business_name") or name
                ),
                data_notes=(
                    None
                    if item.get("data_notes") in (None, "")
                    else str(item.get("data_notes"))
                ),
            )
        )
    return out


def _merge_column_summaries(
    parsed: list[ColumnSummary],
    table: TableSchema,
    profile: TableProfile,
) -> list[ColumnSummary]:
    """Ensure every schema column appears at least once (fill gaps with fallbacks)."""
    by_name = {c.column_name: c for c in parsed}
    pmap = {c.column_name: c for c in profile.columns}
    merged: list[ColumnSummary] = []
    for col in table.columns:
        if col.name in by_name:
            merged.append(by_name[col.name])
        else:
            merged.append(_fallback_column_summary(col, pmap.get(col.name)))
    return merged


def _call_claude(
    prompt: str,
    *,
    model: str,
    max_tokens: int = 1000,
) -> tuple[str | None, float | None]:
    """
    Invoke Groq and return assistant text plus elapsed seconds.

    Args:
        prompt: User message body.
        model: Groq model id.
        max_tokens: Response token budget.

    Returns:
        ``(text, seconds)`` or ``(None, seconds)`` on failure.
    """
    if not GROQ_API_KEY:
        logger.warning("llm_call_skipped", reason="missing GROQ_API_KEY")
        return None, None
    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.error("llm_client_init_failed", error=repr(e))
        return None, None

    t0 = time.perf_counter()
    try:
        msg = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.perf_counter() - t0
        content = msg.choices[0].message.content if msg.choices else None
        text = (content or "").strip() or None
        return text, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(
            "llm_api_call_failed",
            error=repr(e),
            seconds=round(elapsed, 4),
        )
        return None, elapsed


def _summarise_one_table(
    table: TableSchema,
    profile: TableProfile,
    rel_map: RelationshipMap,
    pii_report: PIIReport,
    *,
    model: str,
    max_tokens: int,
) -> TableSummary:
    """
    Call Claude for a single table and parse :class:`TableSummary`.

    On API or JSON failure, apply structured fallbacks per product rules.
    """
    rels = _relationships_for_table(rel_map, table.table_name)
    pii = _pii_for_table(pii_report, table.table_name)
    prompt = build_table_prompt(table, profile, rels, pii)

    raw, elapsed = _call_claude(prompt, model=model, max_tokens=max_tokens)
    logger.info(
        "llm_table_call",
        table=table.table_name,
        model=model,
        seconds=None if elapsed is None else round(elapsed, 4),
        success=raw is not None,
    )

    if raw is None:
        return _fallback_table_summary(table, profile, pii)

    data = _extract_json_object(raw)
    if data is None:
        logger.error(
            "llm_table_json_parse_failed",
            table=table.table_name,
            preview=raw[:500],
        )
        fb = _fallback_table_summary(table, profile, pii)
        return TableSummary(
            table_name=table.table_name,
            business_description=raw.strip(),
            suggested_business_name=fb.suggested_business_name,
            business_purpose=_UNAVAILABLE,
            columns=fb.columns,
            data_quality_notes=fb.data_quality_notes,
            pii_warning=fb.pii_warning,
        )

    cols = _merge_column_summaries(
        _parse_column_summaries(data.get("columns"), table),
        table,
        profile,
    )
    pii_warn = data.get("pii_warning")
    return TableSummary(
        table_name=table.table_name,
        business_description=str(
            data.get("business_description") or _UNAVAILABLE
        ),
        suggested_business_name=str(
            data.get("suggested_business_name") or table.table_name
        ),
        business_purpose=str(data.get("business_purpose") or _UNAVAILABLE),
        columns=cols,
        data_quality_notes=str(
            data.get("data_quality_notes") or _UNAVAILABLE
        ),
        pii_warning=None if pii_warn in (None, "") else str(pii_warn),
    )


def summarise_database(
    schema: DatabaseSchema,
    profile: DatabaseProfile,
    rel_map: RelationshipMap,
    pii_report: PIIReport,
) -> DatabaseSummary:
    """
    Produce :class:`DatabaseSummary` by calling Claude per table, then once for the DB.

    Uses :data:`config.settings.ANTHROPIC_API_KEY` and :data:`config.settings.LLM_MODEL`.
    Table results may be reused from an in-memory cache when
    :data:`config.settings.LLM_CACHE_ENABLED` is true.

    Args:
        schema: Extracted schema.
        profile: Data profile with sample stats.
        rel_map: Relationship graph.
        pii_report: PII scan output.

    Returns:
        Fully populated :class:`DatabaseSummary`.
    """
    global _TABLE_SUMMARY_CACHE

    model = "llama-3.3-70b-versatile"
    max_tokens = 1000
    logger.info(
        "llm_summarise_start",
        tables=len(schema.tables),
        model=model,
        cache_enabled=LLM_CACHE_ENABLED,
    )

    table_summaries: list[TableSummary] = []
    for ts in schema.tables:
        tp = _get_table_profile(profile, ts.table_name)
        if tp is None:
            logger.warning(
                "llm_summarise_missing_profile",
                table=ts.table_name,
            )
            tp = TableProfile(
                table_name=ts.table_name,
                total_rows=0,
                total_columns=len(ts.columns),
                columns=[
                    ColumnProfile(
                        column_name=c.name,
                        data_type=c.data_type,
                        total_rows=0,
                        null_count=0,
                        null_rate=0.0,
                        unique_count=0,
                        uniqueness_rate=0.0,
                        min_value=None,
                        max_value=None,
                        mean_value=None,
                        sample_values=[],
                        is_constant=False,
                        is_unique_key=False,
                    )
                    for c in ts.columns
                ],
                completeness_score=1.0,
                duplicate_row_count=0,
                has_data_quality_issues=False,
            )

        if LLM_CACHE_ENABLED and ts.table_name in _TABLE_SUMMARY_CACHE:
            logger.info("llm_table_cache_hit", table=ts.table_name)
            table_summaries.append(_TABLE_SUMMARY_CACHE[ts.table_name])
            continue

        summary = _summarise_one_table(
            ts, tp, rel_map, pii_report, model=model, max_tokens=max_tokens
        )
        if LLM_CACHE_ENABLED:
            _TABLE_SUMMARY_CACHE[ts.table_name] = summary
        table_summaries.append(summary)

    db_prompt = build_database_prompt(schema, table_summaries)
    raw_db, elapsed_db = _call_claude(db_prompt, model=model, max_tokens=max_tokens)
    logger.info(
        "llm_database_call",
        model=model,
        seconds=None if elapsed_db is None else round(elapsed_db, 4),
        success=raw_db is not None,
    )

    db_desc = _UNAVAILABLE
    domain = "Unknown"
    if raw_db is not None:
        parsed = _extract_json_object(raw_db)
        if parsed is None:
            logger.error(
                "llm_database_json_parse_failed",
                preview=raw_db[:500],
            )
            db_desc = raw_db.strip()
        else:
            db_desc = str(parsed.get("database_description") or _UNAVAILABLE)
            domain = str(parsed.get("business_domain") or "Unknown")

    result = DatabaseSummary(
        database_description=db_desc,
        business_domain=domain,
        tables=table_summaries,
        generated_at=_utc_iso(),
        model_used=model,
    )
    logger.info(
        "llm_summarise_done",
        tables=len(table_summaries),
        model=model,
    )
    return result


def summary_to_dict(summary: DatabaseSummary) -> dict:
    """
    Convert :class:`DatabaseSummary` to nested dicts for JSON export.

    Args:
        summary: Output of :func:`summarise_database`.

    Returns:
        JSON-serialisable structure.
    """
    return asdict(summary)


if __name__ == "__main__":
    import sys

    _proj = Path(__file__).resolve().parent.parent
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    from connectors.sqlite_connector import load_sqlite

    from agents.data_profiler import profile_database
    from agents.pii_detector import scan_for_pii
    from agents.relationship_mapper import build_relationship_map
    from agents.schema_extractor import extract_schema

    _chinook = _proj / "sample_data" / "chinook" / "Chinook_Sqlite.sqlite"
    _out_dir = _proj / "output"
    _out_dir.mkdir(parents=True, exist_ok=True)
    _out_file = _out_dir / "chinook_summary.json"

    _engine = load_sqlite(str(_chinook))
    _source = str(_chinook.resolve())
    _schema = extract_schema(_engine, _source)
    _profile = profile_database(_engine, _schema, sample_size=3000)
    _rels = build_relationship_map(_schema)
    _pii = scan_for_pii(_engine, _schema, _profile)

    _summary = summarise_database(_schema, _profile, _rels, _pii)

    print("\nDatabase description:\n", _summary.database_description)
    print("\nBusiness domain:", _summary.business_domain)
    print("\n--- Tables ---")
    for t in _summary.tables:
        print(f"\n[{t.table_name}]")
        print("  business_description:", t.business_description[:200] + ("..." if len(t.business_description) > 200 else ""))
        print("  business_purpose:", t.business_purpose[:200] + ("..." if len(t.business_purpose) > 200 else ""))

    _out_file.write_text(
        json.dumps(summary_to_dict(_summary), indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {_out_file}")

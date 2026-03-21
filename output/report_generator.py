"""
Render a full DictionaryResult into polished HTML and optional PDF reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape

from orchestrator.pipeline import (
    DictionaryResult,
    InputType,
    PipelineConfig,
    run_pipeline,
)

logger = structlog.get_logger(__name__)


@dataclass
class ReportConfig:
    """Configuration for rendering dictionary reports."""

    output_dir: str
    report_title: str = "Data Dictionary Report"
    include_pii_section: bool = True
    include_quality_section: bool = True
    include_relationships_section: bool = True
    include_llm_summaries: bool = True


def _quality_label(score: float) -> str:
    """Return display bucket name for completeness score."""
    if score > 0.95:
        return "good"
    if score > 0.80:
        return "warn"
    return "bad"


def _quality_badge_class(null_rate: float) -> str:
    """Badge class from null-rate thresholds."""
    completeness = 1.0 - null_rate
    return _quality_label(completeness)


def _table_relationships(result: DictionaryResult, table_name: str) -> list[dict]:
    """Collect relationships touching one table into template-friendly dicts."""
    out: list[dict] = []
    for rel in result.relationships.relationships:
        if rel.from_table != table_name and rel.to_table != table_name:
            continue
        out.append(
            {
                "from_table": rel.from_table,
                "from_column": rel.from_column,
                "to_table": rel.to_table,
                "to_column": rel.to_column,
                "relationship_type": rel.relationship_type,
                "is_explicit": rel.is_explicit,
                "confidence": rel.confidence,
            }
        )
    return out


def _table_pii(result: DictionaryResult, table_name: str) -> list[dict]:
    """Collect PII findings for a table."""
    if result.pii_report is None:
        return []
    findings = []
    for f in result.pii_report.findings:
        if f.table_name != table_name:
            continue
        findings.append(
            {
                "column_name": f.column_name,
                "pii_type": f.pii_type,
                "detection_method": f.detection_method,
                "confidence": f.confidence,
                "sample_trigger": f.sample_trigger,
            }
        )
    return findings


def _table_summary_lookup(result: DictionaryResult) -> dict[str, object]:
    """Map table name -> TableSummary for quick joins."""
    if result.summary is None:
        return {}
    return {t.table_name: t for t in result.summary.tables}


def _mermaid_entity_name(table_name: str) -> str:
    """Return a Mermaid-safe entity identifier (quote if needed)."""
    if table_name.isidentifier():
        return table_name
    escaped = table_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _build_mermaid_er_diagram(result: DictionaryResult) -> str | None:
    """
    Build Mermaid ``erDiagram`` source from relationships and schema tables.

    Returns None when there are no relationships.
    """
    rel_map = result.relationships
    if not rel_map.relationships:
        return None

    lines: list[str] = ["erDiagram"]
    for rel in rel_map.relationships:
        if rel.relationship_type == "one_to_many":
            arrow = "||--o{"
        elif rel.relationship_type == "many_to_many":
            arrow = "}o--o{"
        elif rel.relationship_type == "one_to_one":
            arrow = "||--||"
        else:
            continue
        ft = _mermaid_entity_name(rel.from_table)
        tt = _mermaid_entity_name(rel.to_table)
        label = rel.from_column.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f"    {ft} {arrow} {tt} : \"{label}\"")

    for ts in result.schema.tables:
        ent = _mermaid_entity_name(ts.table_name)
        lines.append(f"    {ent} {{")
        lines.append("    }")

    return "\n".join(lines)


def prepare_template_context(result: DictionaryResult) -> dict:
    """
    Build the complete Jinja context from a DictionaryResult.

    Includes table-level schema/profile/relationship/PII data and top-level report stats.
    """
    summary_map = _table_summary_lookup(result)
    schema_map = {t.table_name: t for t in result.schema.tables}
    rel_total = result.relationships.total_relationships
    pii_total = 0 if result.pii_report is None else result.pii_report.total_pii_columns

    tables: list[dict] = []
    quality_issues: list[dict] = []

    for tp in result.profile.tables:
        ts = schema_map.get(tp.table_name)
        if ts is None:
            continue

        table_summary = summary_map.get(tp.table_name)
        column_summary_map = {}
        if table_summary is not None:
            column_summary_map = {c.column_name: c for c in table_summary.columns}

        pii_rows = _table_pii(result, tp.table_name)
        pii_by_column: dict[str, list[dict]] = {}
        for p in pii_rows:
            pii_by_column.setdefault(p["column_name"], []).append(p)

        columns: list[dict] = []
        for c in ts.columns:
            profile_col = next((x for x in tp.columns if x.column_name == c.name), None)
            summary_col = column_summary_map.get(c.name)
            null_rate = 0.0 if profile_col is None else profile_col.null_rate
            columns.append(
                {
                    "name": c.name,
                    "data_type": c.data_type,
                    "is_nullable": c.is_nullable,
                    "is_primary_key": c.is_primary_key,
                    "is_foreign_key": c.is_foreign_key,
                    "foreign_key_references": c.foreign_key_references,
                    "null_rate": null_rate,
                    "null_rate_pct": round(null_rate * 100, 2),
                    "quality_class": _quality_badge_class(null_rate),
                    "business_description": (
                        None if summary_col is None else summary_col.business_description
                    ),
                    "business_name": (
                        None if summary_col is None else summary_col.suggested_business_name
                    ),
                    "pii_findings": pii_by_column.get(c.name, []),
                }
            )

        table_dict = {
            "name": tp.table_name,
            "anchor": tp.table_name.replace(" ", "_"),
            "row_count": tp.total_rows,
            "column_count": tp.total_columns,
            "completeness_score": tp.completeness_score,
            "completeness_pct": round(tp.completeness_score * 100, 2),
            "completeness_class": _quality_label(tp.completeness_score),
            "duplicate_row_count": tp.duplicate_row_count,
            "has_data_quality_issues": tp.has_data_quality_issues,
            "business_description": (
                None if table_summary is None else table_summary.business_description
            ),
            "business_purpose": (
                None if table_summary is None else table_summary.business_purpose
            ),
            "data_quality_notes": (
                None if table_summary is None else table_summary.data_quality_notes
            ),
            "pii_warning": None if table_summary is None else table_summary.pii_warning,
            "columns": columns,
            "relationships": _table_relationships(result, tp.table_name),
            "pii_findings": pii_rows,
        }
        tables.append(table_dict)

        if tp.has_data_quality_issues:
            quality_issues.append(
                {
                    "table_name": tp.table_name,
                    "completeness_score": tp.completeness_score,
                    "duplicate_row_count": tp.duplicate_row_count,
                }
            )

    context = {
        "report_title": "Data Dictionary Report",
        "generated_at": result.generated_at,
        "pipeline_duration": result.pipeline_duration_seconds,
        "database_description": (
            None if result.summary is None else result.summary.database_description
        ),
        "business_domain": (None if result.summary is None else result.summary.business_domain),
        "tables": tables,
        "overall_stats": {
            "total_tables": result.schema.total_tables,
            "total_columns": result.schema.total_columns,
            "overall_completeness": result.profile.overall_completeness,
            "total_relationships": rel_total,
            "total_pii_columns": pii_total,
        },
        "quality_issues": quality_issues,
        "mermaid_er_diagram": _build_mermaid_er_diagram(result),
    }
    return context


def _build_template_env(base_dir: Path) -> Environment:
    """Create Jinja environment rooted at the output template folder."""
    templates_dir = base_dir / "templates"
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def generate_report(result: DictionaryResult, config: ReportConfig) -> dict:
    """
    Render dictionary report HTML and attempt PDF generation.

    Returns:
        dict with ``html_path``, ``pdf_path`` (or None), and ``success``.
    """
    t0 = datetime.now(timezone.utc)
    logger.info(
        "report_generate_start",
        output_dir=config.output_dir,
        report_title=config.report_title,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module_dir = Path(__file__).resolve().parent
    template_name = "dictionary.html"

    context = prepare_template_context(result)
    context["report_title"] = config.report_title
    context["include_pii_section"] = config.include_pii_section
    context["include_quality_section"] = config.include_quality_section
    context["include_relationships_section"] = config.include_relationships_section
    context["include_llm_summaries"] = config.include_llm_summaries

    env = _build_template_env(module_dir)
    template = env.get_template(template_name)
    html = template.render(**context)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    html_path = output_dir / f"data_dictionary_{stamp}.html"
    html_path.write_text(html, encoding="utf-8")
    logger.info("report_html_written", html_path=str(html_path.resolve()))

    pdf_path: Path | None = output_dir / f"data_dictionary_{stamp}.pdf"
    try:
        from weasyprint import HTML

        HTML(string=html, base_url=str(module_dir)).write_pdf(str(pdf_path))
        logger.info("report_pdf_written", pdf_path=str(pdf_path.resolve()))
    except Exception as e:
        logger.warning(
            "report_pdf_skipped",
            error=str(e),
            reason="weasyprint_failed_on_platform",
        )
        pdf_path = None

    delta = (datetime.now(timezone.utc) - t0).total_seconds()
    logger.info("report_generate_done", seconds=round(delta, 4))
    return {
        "html_path": str(html_path.resolve()),
        "pdf_path": None if pdf_path is None else str(pdf_path.resolve()),
        "success": True,
    }


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

    _chinook = _proj / "sample_data" / "chinook" / "Chinook_Sqlite.sqlite"
    _cfg = PipelineConfig(
        input_path=str(_chinook.resolve()),
        input_type=InputType.SQLITE_FILE,
        sample_size=10000,
        enable_llm=True,
        enable_pii=True,
    )
    _result = run_pipeline(_cfg)
    _report_cfg = ReportConfig(output_dir=str((_proj / "output" / "generated").resolve()))
    _out = generate_report(_result, _report_cfg)
    print("HTML report:", _out["html_path"])

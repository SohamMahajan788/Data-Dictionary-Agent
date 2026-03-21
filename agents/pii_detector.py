"""
Flag columns that likely contain PII using column-name rules and Microsoft Presidio.

Consumes :class:`DatabaseProfile` for sample values and :class:`DatabaseSchema` for FK hints.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
import spacy
import structlog
from sqlalchemy.engine import Engine

from agents.data_profiler import DatabaseProfile, TableProfile
from agents.schema_extractor import DatabaseSchema, TableSchema

logger = structlog.get_logger(__name__)

_STATE_TOKEN = re.compile(r"(^|_)state($|_)", re.IGNORECASE)

_PRESIDIO_ENTITY_TO_PII: dict[str, str] = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "PERSON": "NAME",
    "LOCATION": "LOCATION",
    "CREDIT_CARD": "FINANCIAL",
    "IBAN_CODE": "FINANCIAL",
}

_ANALYZER = None


@dataclass
class PIIFinding:
    """Single PII hit for a table column."""

    table_name: str
    column_name: str
    pii_type: str
    detection_method: str
    confidence: float
    sample_trigger: str | None


@dataclass
class PIIReport:
    """Aggregated PII scan output."""

    findings: list[PIIFinding]
    total_pii_columns: int
    tables_with_pii: list[str]
    scan_summary: dict


def _has_lat_token(column_lower: str) -> bool:
    """
    Heuristic for latitude-related column names without matching arbitrary ``lat`` substrings.

    Args:
        column_lower: Lowercased column name.

    Returns:
        True if the name suggests latitude coordinates.
    """
    c = column_lower
    return (
        "latitude" in c
        or c.endswith("lat")
        or "_lat" in c
        or c.startswith("lat_")
    )


def _has_lng_token(column_lower: str) -> bool:
    """
    Heuristic for longitude / ``lng`` style column names.

    Args:
        column_lower: Lowercased column name.

    Returns:
        True if the name suggests longitude coordinates.
    """
    c = column_lower
    return "longitude" in c or "lng" in c


def detect_pii_by_column_name(table: str, column: str) -> PIIFinding | None:
    """
    Apply deterministic column-name rules for likely PII.

    The ``_id`` suffix rule is always emitted when it matches; callers should drop it
    when the column is a known foreign key (see :func:`scan_for_pii`).

    Args:
        table: Table name (passed through to :class:`PIIFinding`).
        column: Column name to inspect.

    Returns:
        A :class:`PIIFinding` or ``None`` if no rule matches.
    """
    c = column.lower()

    if "email" in c:
        return PIIFinding(
            table_name=table,
            column_name=column,
            pii_type="EMAIL",
            detection_method="rule_based",
            confidence=0.95,
            sample_trigger=None,
        )
    if "phone" in c or "fax" in c:
        return PIIFinding(
            table_name=table,
            column_name=column,
            pii_type="PHONE",
            detection_method="rule_based",
            confidence=0.95,
            sample_trigger=None,
        )
    if "address" in c or "street" in c:
        return PIIFinding(
            table_name=table,
            column_name=column,
            pii_type="LOCATION",
            detection_method="rule_based",
            confidence=0.90,
            sample_trigger=None,
        )
    if (
        "zip" in c
        or "postal" in c
        or "city" in c
        or _STATE_TOKEN.search(c)
    ):
        return PIIFinding(
            table_name=table,
            column_name=column,
            pii_type="LOCATION",
            detection_method="rule_based",
            confidence=0.75,
            sample_trigger=None,
        )
    if "dob" in c or "birth" in c:
        return PIIFinding(
            table_name=table,
            column_name=column,
            pii_type="DATE_OF_BIRTH",
            detection_method="rule_based",
            confidence=0.95,
            sample_trigger=None,
        )
    if "card" in c or "account" in c or "iban" in c:
        return PIIFinding(
            table_name=table,
            column_name=column,
            pii_type="FINANCIAL",
            detection_method="rule_based",
            confidence=0.95,
            sample_trigger=None,
        )
    if _has_lat_token(c) or _has_lng_token(c):
        return PIIFinding(
            table_name=table,
            column_name=column,
            pii_type="LOCATION",
            detection_method="rule_based",
            confidence=0.85,
            sample_trigger=None,
        )
    if "name" in c and "username" not in c:
        return PIIFinding(
            table_name=table,
            column_name=column,
            pii_type="NAME",
            detection_method="rule_based",
            confidence=0.80,
            sample_trigger=None,
        )
    if c.endswith("_id"):
        return PIIFinding(
            table_name=table,
            column_name=column,
            pii_type="ID",
            detection_method="rule_based",
            confidence=0.70,
            sample_trigger=None,
        )
    return None


def _get_presidio_analyzer():
    """
    Lazily construct a Presidio :class:`~presidio_analyzer.AnalyzerEngine` with SpaCy lg.

    Returns:
        Analyzer instance or ``None`` if Presidio/SpaCy/model setup fails.
    """
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import SpacyNlpEngine

        nlp_engine = SpacyNlpEngine(
            models=[{"lang_code": "en", "model_name": "en_core_web_lg"}]
        )
        nlp_engine.load()
        _ANALYZER = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en"],
        )
        logger.info("presidio_analyzer_ready", model="en_core_web_lg")
    except Exception as e:
        logger.warning(
            "presidio_analyzer_init_failed",
            error=str(e),
            exc_info=True,
        )
        _ANALYZER = False
    return _ANALYZER if _ANALYZER is not False else None


def detect_pii_by_presidio(
    table: str,
    column: str,
    sample_values: list[str],
) -> list[PIIFinding]:
    """
    Run Presidio's analyzer on sample cell values and map entities to PII types.

    Uses ``en_core_web_lg`` via :class:`~presidio_analyzer.nlp_engine.SpacyNlpEngine`.
    Only entities in ``EMAIL_ADDRESS``, ``PHONE_NUMBER``, ``PERSON``, ``LOCATION``,
    ``CREDIT_CARD``, ``IBAN_CODE`` with score ``>= 0.7`` are returned.

    Args:
        table: Table name.
        column: Column name.
        sample_values: Non-null sample strings (e.g. from :class:`ColumnProfile`).

    Returns:
        Zero or more :class:`PIIFinding` rows with ``detection_method="presidio"``.
    """
    analyzer = _get_presidio_analyzer()
    if analyzer is None:
        return []

    allowed = set(_PRESIDIO_ENTITY_TO_PII.keys())
    min_score = 0.7
    findings: list[PIIFinding] = []

    for raw in sample_values:
        text = (raw or "").strip()
        if not text:
            continue
        try:
            results = analyzer.analyze(text=text, language="en")
        except Exception as e:
            logger.warning(
                "presidio_analyze_failed",
                table=table,
                column=column,
                error=str(e),
                exc_info=True,
            )
            continue

        for r in results:
            if r.entity_type not in allowed:
                continue
            score = float(r.score)
            if score < min_score:
                continue
            pii_type = _PRESIDIO_ENTITY_TO_PII[r.entity_type]
            trig = text if len(text) <= 200 else text[:197] + "..."
            findings.append(
                PIIFinding(
                    table_name=table,
                    column_name=column,
                    pii_type=pii_type,
                    detection_method="presidio",
                    confidence=score,
                    sample_trigger=trig,
                )
            )

    logger.debug(
        "presidio_column_scanned",
        table=table,
        column=column,
        hits=len(findings),
    )
    return findings


def _column_foreign_key_flag(schema: DatabaseSchema, table: str, column: str) -> bool:
    """
    Return True if ``schema`` marks ``column`` on ``table`` as a foreign key.

    Args:
        schema: Extracted schema.
        table: Table name.
        column: Column name.

    Returns:
        Whether the column is flagged as FK (or False if unknown).
    """
    ts = _get_table_schema(schema, table)
    if ts is None:
        return False
    for col in ts.columns:
        if col.name == column:
            return bool(col.is_foreign_key)
    return False


def _get_table_schema(schema: DatabaseSchema, table: str) -> TableSchema | None:
    """Return :class:`TableSchema` by name or ``None``."""
    for ts in schema.tables:
        if ts.table_name == table:
            return ts
    return None


def _dedupe_findings(findings: list[PIIFinding]) -> list[PIIFinding]:
    """
    Keep the highest-confidence finding per (table, column, PII type).

    Args:
        findings: Raw hits from rules and/or Presidio.

    Returns:
        Deduplicated list.
    """
    best: dict[tuple[str, str, str], PIIFinding] = {}
    for f in findings:
        key = (f.table_name, f.column_name, f.pii_type)
        cur = best.get(key)
        if cur is None or f.confidence > cur.confidence:
            best[key] = f
    return list(best.values())


def scan_for_pii(
    engine: Engine,
    schema: DatabaseSchema,
    profile: DatabaseProfile,
) -> PIIReport:
    """
    Scan every profiled column: rules first, then Presidio on samples if no rule hit.

    Drops rule-based ``ID`` findings for declared foreign-key columns. Merges duplicate
    (table, column, type) rows, keeping the strongest confidence.

    Args:
        engine: SQLAlchemy engine (reserved for future live sampling; logged only).
        schema: Schema with FK flags.
        profile: Profile containing ``sample_values`` per column.

    Returns:
        :class:`PIIReport` with summary statistics.
    """
    try:
        dialect = engine.dialect.name
    except Exception:
        dialect = "unknown"
    logger.info(
        "pii_scan_start",
        source_path=schema.source_path,
        tables=schema.total_tables,
        dialect=dialect,
        spacy_version=spacy.__version__,
    )

    raw: list[PIIFinding] = []
    rule_hits = 0
    presidio_hits = 0
    columns_scanned = 0

    for tp in profile.tables:
        logger.info("pii_scan_table_start", table=tp.table_name, columns=len(tp.columns))
        for cp in tp.columns:
            columns_scanned += 1
            rule = detect_pii_by_column_name(tp.table_name, cp.column_name)
            if rule is not None and rule.pii_type == "ID":
                if _column_foreign_key_flag(schema, tp.table_name, cp.column_name):
                    rule = None
                    logger.debug(
                        "pii_rule_id_skipped_fk",
                        table=tp.table_name,
                        column=cp.column_name,
                    )

            col_findings: list[PIIFinding] = []
            if rule is not None:
                col_findings.append(rule)
                rule_hits += 1
                logger.debug(
                    "pii_rule_hit",
                    table=tp.table_name,
                    column=cp.column_name,
                    pii_type=rule.pii_type,
                    confidence=rule.confidence,
                )
            else:
                pres = detect_pii_by_presidio(
                    tp.table_name,
                    cp.column_name,
                    cp.sample_values,
                )
                col_findings.extend(pres)
                presidio_hits += len(pres)
                if pres:
                    logger.debug(
                        "pii_presidio_hits",
                        table=tp.table_name,
                        column=cp.column_name,
                        count=len(pres),
                    )

            raw.extend(col_findings)

        logger.info("pii_scan_table_done", table=tp.table_name)

    deduped = _dedupe_findings(raw)
    pii_columns = {(f.table_name, f.column_name) for f in deduped}
    tables_with = sorted({f.table_name for f in deduped})

    by_type: dict[str, int] = {}
    for f in deduped:
        by_type[f.pii_type] = by_type.get(f.pii_type, 0) + 1

    summary = {
        "columns_scanned": columns_scanned,
        "rule_based_hits": rule_hits,
        "presidio_finding_rows": presidio_hits,
        "findings_after_dedupe": len(deduped),
        "by_pii_type": by_type,
    }

    report = PIIReport(
        findings=sorted(
            deduped,
            key=lambda x: (x.table_name, x.column_name, x.pii_type),
        ),
        total_pii_columns=len(pii_columns),
        tables_with_pii=tables_with,
        scan_summary=summary,
    )

    logger.info(
        "pii_scan_done",
        total_pii_columns=report.total_pii_columns,
        tables_with_pii=len(tables_with),
        findings=len(deduped),
        scan_summary=summary,
    )
    return report


def pii_report_to_dict(report: PIIReport) -> dict:
    """
    Serialise a :class:`PIIReport` to nested dicts.

    Args:
        report: Output of :func:`scan_for_pii`.

    Returns:
        JSON-friendly structure.
    """
    return asdict(report)


def _print_findings_by_table(report: PIIReport) -> None:
    """
    Print all findings grouped by table and the total PII column count.

    Args:
        report: Report to display.
    """
    by_table: dict[str, list[PIIFinding]] = {}
    for f in report.findings:
        by_table.setdefault(f.table_name, []).append(f)

    print(f"\nTotal PII columns (distinct table.column): {report.total_pii_columns}")
    print(f"Tables with PII: {report.tables_with_pii}")
    for table in sorted(by_table.keys()):
        print(f"\n[{table}]")
        for f in by_table[table]:
            src = f"[{f.detection_method}]"
            trig = f" sample={f.sample_trigger!r}" if f.sample_trigger else ""
            print(
                f"  {src} {f.column_name}: {f.pii_type} "
                f"confidence={f.confidence:.3f}{trig}"
            )


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

    from connectors.csv_loader import load_csv_folder
    from connectors.sqlite_connector import load_sqlite

    from agents.data_profiler import profile_database
    from agents.schema_extractor import extract_schema

    def _run(label: str, eng: Engine, source: str) -> None:
        sch = extract_schema(eng, source)
        prof = profile_database(eng, sch, sample_size=5000)
        rep = scan_for_pii(eng, sch, prof)
        print(f"\n======== {label} ========")
        _print_findings_by_table(rep)
        print(f"\nscan_summary: {rep.scan_summary}")

    _olist = _proj / "sample_data" / "olist"
    _chinook = _proj / "sample_data" / "chinook" / "Chinook_Sqlite.sqlite"

    _e1 = load_csv_folder(str(_olist))
    _run("olist", _e1, str(_olist.resolve()))

    _e2 = load_sqlite(str(_chinook))
    _run("chinook", _e2, str(_chinook.resolve()))

"""
Run the full data-dictionary pipeline end-to-end from any supported input source.

This orchestrator wires connectors + agents and returns one typed result object.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import structlog
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from agents.data_profiler import DatabaseProfile, profile_database, profile_to_dict
from agents.llm_summariser import DatabaseSummary, summarise_database, summary_to_dict
from agents.pii_detector import PIIReport, pii_report_to_dict, scan_for_pii
from agents.relationship_mapper import (
    RelationshipMap,
    build_relationship_map,
    relationship_map_to_dict,
)
from agents.schema_extractor import DatabaseSchema, extract_schema, schema_to_dict
from connectors.csv_loader import load_csv_folder
from connectors.sqlite_connector import load_sqlite

logger = structlog.get_logger(__name__)


class InputType(str, Enum):
    """Supported pipeline inputs."""

    CSV_FOLDER = "csv_folder"
    SQLITE_FILE = "sqlite_file"
    POSTGRES = "postgres"
    MYSQL = "mysql"


@dataclass
class PipelineConfig:
    """Runtime settings for one pipeline execution."""

    input_path: str
    input_type: InputType
    sample_size: int = 10000
    enable_llm: bool = True
    enable_pii: bool = True
    output_dir: str = "./output/generated"


@dataclass
class DictionaryResult:
    """Complete pipeline output."""

    config: PipelineConfig
    schema: DatabaseSchema
    profile: DatabaseProfile
    relationships: RelationshipMap
    pii_report: PIIReport | None
    summary: DatabaseSummary | None
    generated_at: str
    pipeline_duration_seconds: float


def _utc_iso() -> str:
    """Return current UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _step_timer() -> float:
    """Short alias for perf timing start markers."""
    return time.perf_counter()


def _elapsed(start: float) -> float:
    """Elapsed seconds from ``start``."""
    return time.perf_counter() - start


def detect_input_type(input_path: str) -> InputType:
    """
    Detect input kind from path/URI.

    Rules:
    - suffix ``.sqlite``/``.db`` -> SQLITE_FILE
    - suffix ``.sql`` -> unsupported (raises ValueError)
    - existing directory -> CSV_FOLDER
    - prefix ``postgresql://`` -> POSTGRES
    - prefix ``mysql://`` -> MYSQL

    Args:
        input_path: Folder path, file path, or connection URI.

    Returns:
        Detected :class:`InputType`.

    Raises:
        ValueError: Unsupported or unrecognised input.
    """
    p = input_path.strip()
    lower = p.lower()
    if lower.endswith(".sqlite") or lower.endswith(".db"):
        return InputType.SQLITE_FILE
    if lower.endswith(".sql"):
        raise ValueError("SQL dump input (.sql) is not supported yet.")
    path_obj = Path(p)
    if path_obj.exists() and path_obj.is_dir():
        return InputType.CSV_FOLDER
    if lower.startswith("postgresql://"):
        return InputType.POSTGRES
    if lower.startswith("mysql://"):
        return InputType.MYSQL
    raise ValueError(f"Could not detect input type for: {input_path}")


def _load_engine(config: PipelineConfig, detected: InputType) -> Engine:
    """
    Create/load SQLAlchemy engine based on detected input type.

    Args:
        config: Pipeline configuration.
        detected: Input type from :func:`detect_input_type`.

    Returns:
        SQLAlchemy engine.
    """
    if detected == InputType.CSV_FOLDER:
        return load_csv_folder(config.input_path)
    if detected == InputType.SQLITE_FILE:
        return load_sqlite(config.input_path)
    if detected in (InputType.POSTGRES, InputType.MYSQL):
        eng = create_engine(config.input_path, future=True)
        # Early connectivity check so failures happen in step 1.
        with eng.connect():
            pass
        return eng
    raise ValueError(f"Unsupported detected input type: {detected}")


def _build_output_file_path(config: PipelineConfig, generated_at: str) -> Path:
    """Deterministically build output JSON filename for one run."""
    base = Path(config.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = generated_at.replace(":", "-")
    safe_input = config.input_type.value
    return base / f"dictionary_result_{safe_input}_{stamp}.json"


def run_pipeline(config: PipelineConfig) -> DictionaryResult:
    """
    Run all pipeline stages and save full JSON output.

    Steps:
    1. Detect input type and load connector
    2. Extract schema
    3. Build relationship map
    4. Profile database
    5. Detect PII (optional)
    6. Generate LLM summaries (optional)
    7. Save full result JSON to output_dir
    8. Return :class:`DictionaryResult`

    Args:
        config: Pipeline settings.

    Returns:
        Completed pipeline result.
    """
    t_total = _step_timer()
    logger.info(
        "pipeline_start",
        input_path=config.input_path,
        configured_input_type=config.input_type.value,
        sample_size=config.sample_size,
        enable_pii=config.enable_pii,
        enable_llm=config.enable_llm,
        output_dir=config.output_dir,
    )

    try:
        t1 = _step_timer()
        detected = detect_input_type(config.input_path)
        if detected != config.input_type:
            logger.warning(
                "pipeline_input_type_mismatch",
                configured=config.input_type.value,
                detected=detected.value,
            )
        engine = _load_engine(config, detected)
        logger.info(
            "pipeline_step_done",
            step="load_connector",
            detected_input_type=detected.value,
            seconds=round(_elapsed(t1), 4),
        )

        t2 = _step_timer()
        schema = extract_schema(engine, config.input_path)
        logger.info(
            "pipeline_step_done",
            step="extract_schema",
            total_tables=schema.total_tables,
            total_columns=schema.total_columns,
            seconds=round(_elapsed(t2), 4),
        )

        t3 = _step_timer()
        relationships = build_relationship_map(schema)
        logger.info(
            "pipeline_step_done",
            step="build_relationship_map",
            total_relationships=relationships.total_relationships,
            seconds=round(_elapsed(t3), 4),
        )

        t4 = _step_timer()
        profile = profile_database(engine, schema, sample_size=config.sample_size)
        logger.info(
            "pipeline_step_done",
            step="profile_database",
            overall_completeness=round(profile.overall_completeness, 6),
            seconds=round(_elapsed(t4), 4),
        )

        pii_report: PIIReport | None = None
        if config.enable_pii:
            t5 = _step_timer()
            pii_report = scan_for_pii(engine, schema, profile)
            logger.info(
                "pipeline_step_done",
                step="scan_for_pii",
                total_pii_columns=pii_report.total_pii_columns,
                seconds=round(_elapsed(t5), 4),
            )
        else:
            logger.info("pipeline_step_skipped", step="scan_for_pii", reason="disabled")

        summary: DatabaseSummary | None = None
        if config.enable_llm:
            t6 = _step_timer()
            effective_pii = pii_report or PIIReport(
                findings=[],
                total_pii_columns=0,
                tables_with_pii=[],
                scan_summary={},
            )
            summary = summarise_database(schema, profile, relationships, effective_pii)
            logger.info(
                "pipeline_step_done",
                step="summarise_database",
                model=summary.model_used,
                seconds=round(_elapsed(t6), 4),
            )
        else:
            logger.info(
                "pipeline_step_skipped", step="summarise_database", reason="disabled"
            )

        generated_at = _utc_iso()
        duration = _elapsed(t_total)
        result = DictionaryResult(
            config=config,
            schema=schema,
            profile=profile,
            relationships=relationships,
            pii_report=pii_report,
            summary=summary,
            generated_at=generated_at,
            pipeline_duration_seconds=duration,
        )

        t7 = _step_timer()
        out_file = _build_output_file_path(config, generated_at)
        out_file.write_text(
            json.dumps(result_to_dict(result), indent=2),
            encoding="utf-8",
        )
        logger.info(
            "pipeline_step_done",
            step="save_result_json",
            output_file=str(out_file.resolve()),
            seconds=round(_elapsed(t7), 4),
        )

        logger.info(
            "pipeline_done",
            seconds=round(duration, 4),
            output_file=str(out_file.resolve()),
        )
        return result
    except Exception as e:
        logger.error(
            "pipeline_failed",
            error=str(e),
            seconds=round(_elapsed(t_total), 4),
            exc_info=True,
        )
        raise


def result_to_dict(result: DictionaryResult) -> dict:
    """
    Convert full pipeline result to plain Python structures for JSON.

    Uses each agent's own serializer helpers for consistency.
    """
    return {
        "config": {
            "input_path": result.config.input_path,
            "input_type": result.config.input_type.value,
            "sample_size": result.config.sample_size,
            "enable_llm": result.config.enable_llm,
            "enable_pii": result.config.enable_pii,
            "output_dir": result.config.output_dir,
        },
        "schema": schema_to_dict(result.schema),
        "profile": profile_to_dict(result.profile),
        "relationships": relationship_map_to_dict(result.relationships),
        "pii_report": (
            None if result.pii_report is None else pii_report_to_dict(result.pii_report)
        ),
        "summary": (
            None if result.summary is None else summary_to_dict(result.summary)
        ),
        "generated_at": result.generated_at,
        "pipeline_duration_seconds": result.pipeline_duration_seconds,
    }


def _print_run_summary(result: DictionaryResult) -> None:
    """Print concise run summary for CLI usage."""
    output_path = _build_output_file_path(result.config, result.generated_at)
    pii_total = 0 if result.pii_report is None else result.pii_report.total_pii_columns
    print("\n=== Pipeline Summary ===")
    print(f"Input: {result.config.input_path}")
    print(f"Total tables: {result.schema.total_tables}")
    print(f"Total columns: {result.schema.total_columns}")
    print(f"Overall completeness score: {result.profile.overall_completeness:.4f}")
    print(f"Total relationships found: {result.relationships.total_relationships}")
    print(f"Total PII columns found: {pii_total}")
    print(f"LLM summaries generated: {result.summary is not None}")
    print(f"Output file path: {output_path.resolve()}")
    print(f"Pipeline duration: {result.pipeline_duration_seconds:.2f}s")


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
    _olist = _proj / "sample_data" / "olist"

    _cfg_chinook = PipelineConfig(
        input_path=str(_chinook.resolve()),
        input_type=InputType.SQLITE_FILE,
        sample_size=10000,
        enable_llm=True,
        enable_pii=True,
    )
    _res_chinook = run_pipeline(_cfg_chinook)
    print("\n--- Chinook ---")
    _print_run_summary(_res_chinook)

    _cfg_olist = PipelineConfig(
        input_path=str(_olist.resolve()),
        input_type=InputType.CSV_FOLDER,
        sample_size=10000,
        enable_llm=True,
        enable_pii=True,
    )
    _res_olist = run_pipeline(_cfg_olist)
    print("\n--- Olist ---")
    _print_run_summary(_res_olist)

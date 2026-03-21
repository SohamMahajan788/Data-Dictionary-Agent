"""
FastAPI REST API for the data dictionary pipeline and stored artifacts.

Exposes health checks, pipeline execution, snapshot metadata, drift reports,
table summaries, PII exports, and HTML report downloads.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from orchestrator.pipeline import (
    InputType,
    PipelineConfig,
    result_to_dict,
    run_pipeline,
)
from output.report_generator import ReportConfig, generate_report
from storage.versioning import (
    DriftReport,
    SchemaSnapshot,
    VersionStore,
    drift_report_to_dict,
)

logger = structlog.get_logger(__name__)

API_VERSION = "1.0.0"


def _project_root() -> Path:
    """Return repository root (parent of ``api`` package)."""
    return Path(__file__).resolve().parent.parent


_STRUCTLOG_CONFIGURED = False


def _ensure_structlog_configured() -> None:
    """Configure structlog once for API process logging."""
    global _STRUCTLOG_CONFIGURED
    if _STRUCTLOG_CONFIGURED:
        return
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    _STRUCTLOG_CONFIGURED = True


_ensure_structlog_configured()

VERSION_STORE = VersionStore(
    store_path=str(_project_root() / "storage" / "version_store.json")
)

_REPORT_INDEX_PATH = _project_root() / "storage" / "api_report_index.json"
_OUTPUT_GENERATED = _project_root() / "output" / "generated"

_INPUT_TYPE_ALIASES: dict[str, InputType] = {
    "csv_folder": InputType.CSV_FOLDER,
    "sqlite_file": InputType.SQLITE_FILE,
    "postgres": InputType.POSTGRES,
    "mysql": InputType.MYSQL,
}


class RunPipelineRequest(BaseModel):
    """Request body to execute the full dictionary pipeline."""

    input_path: str = Field(..., description="CSV folder, SQLite file, or DB URI")
    input_type: str = Field(
        ...,
        description="One of: csv_folder, sqlite_file, postgres, mysql",
    )
    sample_size: int = Field(default=10000, ge=1, le=1_000_000)
    enable_llm: bool = True
    enable_pii: bool = True
    report_title: str = "Data Dictionary Report"

    @field_validator("input_type")
    @classmethod
    def _validate_input_type(cls, v: str) -> str:
        """Ensure ``input_type`` is a supported literal."""
        key = v.strip().lower()
        if key not in _INPUT_TYPE_ALIASES:
            raise ValueError(
                "input_type must be one of: csv_folder, sqlite_file, postgres, mysql"
            )
        return key


class TableSummaryResponse(BaseModel):
    """Per-table roll-up for discovery endpoints."""

    table_name: str
    row_count: int
    column_count: int
    completeness_score: float
    has_quality_issues: bool
    pii_column_count: int
    relationships: list[str]


class HealthResponse(BaseModel):
    """Service liveness and store statistics."""

    status: str
    version: str
    store_snapshot_count: int


class SnapshotListItem(BaseModel):
    """Compact snapshot row for catalogue endpoints."""

    snapshot_id: str
    source_path: str
    captured_at: str
    table_count: int
    schema_hash: str


class SnapshotDetailResponse(BaseModel):
    """Full snapshot metadata."""

    snapshot_id: str
    source_path: str
    captured_at: str
    table_names: list[str]
    column_counts: dict[str, int]
    row_counts: dict[str, int]
    completeness_scores: dict[str, float]
    schema_hash: str


class DriftReportResponse(BaseModel):
    """Serialised drift comparison between two snapshots."""

    snapshot_id_before: str
    snapshot_id_after: str
    captured_at: str
    source_path: str
    tables_added: list[str]
    tables_removed: list[str]
    columns_added: dict[str, list[str]]
    columns_removed: dict[str, list[str]]
    completeness_changes: dict[str, dict[str, Any]]
    row_count_changes: dict[str, dict[str, Any]]
    has_breaking_changes: bool
    summary: str


class TableDetailResponse(BaseModel):
    """Rich table payload from the latest saved pipeline JSON."""

    table_name: str
    table_schema: dict[str, Any]
    profile: dict[str, Any]
    relationships: list[str]
    pii_findings: list[dict[str, Any]]
    summary: dict[str, Any] | None = None


def _snapshot_count(store: VersionStore) -> int:
    """Return number of persisted snapshots without mutating the store."""
    return len(getattr(store, "_snapshots", []))


def _paths_match(a: str, b: str) -> bool:
    """
    Return True if two source path strings refer to the same logical source.

    Compares exact, case-insensitive, resolved filesystem paths, and normalized paths.
    """
    la = (a or "").strip()
    lb = (b or "").strip()
    if la == lb:
        return True
    if la.casefold() == lb.casefold():
        return True
    if la.startswith(("postgresql://", "mysql://")):
        return la.casefold() == lb.casefold()
    try:
        pa = Path(la)
        pb = Path(lb)
        if pa.exists() and pb.exists() and pa.resolve() == pb.resolve():
            return True
    except OSError:
        pass
    return os.path.normcase(os.path.normpath(la)) == os.path.normcase(
        os.path.normpath(lb)
    )


def _canonical_source_candidates(raw: str) -> list[str]:
    """Build a small list of path strings to match against stored snapshot paths."""
    p = unquote((raw or "").strip())
    out: list[str] = [p]
    if p.startswith(("postgresql://", "mysql://")):
        return out
    try:
        path_obj = Path(p)
        if path_obj.exists():
            out.append(str(path_obj.resolve()))
    except OSError:
        pass
    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        k = item.casefold()
        if k not in seen:
            seen.add(k)
            deduped.append(item)
    return deduped


def _artifact_storage_key(path_str: str) -> str:
    """Normalised key for the HTML report sidecar index."""
    p = unquote(path_str.strip())
    if p.startswith(("postgresql://", "mysql://")):
        return p.casefold()
    try:
        return str(Path(p).resolve()).casefold()
    except OSError:
        return p.casefold()


def _load_report_index() -> dict[str, dict[str, str]]:
    """Load ``source_key -> {html_path, generated_at}`` mapping."""
    if not _REPORT_INDEX_PATH.exists():
        return {}
    try:
        raw = json.loads(_REPORT_INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("report_index_load_failed", error=str(e))
        return {}
    entries = raw.get("entries", raw)
    if not isinstance(entries, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for k, v in entries.items():
        if isinstance(v, dict) and "html_path" in v:
            out[str(k)] = {
                "html_path": str(v["html_path"]),
                "generated_at": str(v.get("generated_at", "")),
            }
    return out


def _save_report_index(entries: dict[str, dict[str, str]]) -> None:
    """Persist report index next to the version store."""
    _REPORT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": entries}
    _REPORT_INDEX_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _set_latest_report_path(source_path: str, html_path: str, generated_at: str) -> None:
    """Record the newest HTML report path for a pipeline source."""
    idx = _load_report_index()
    key = _artifact_storage_key(source_path)
    idx[key] = {"html_path": html_path, "generated_at": generated_at}
    _save_report_index(idx)
    logger.info("report_index_updated", source_key=key, html_path=html_path)


def _get_latest_snapshot(store: VersionStore, path_param: str) -> SchemaSnapshot | None:
    """Resolve the newest snapshot matching any candidate form of ``path_param``."""
    candidates = _canonical_source_candidates(path_param)
    matches: list[SchemaSnapshot] = []
    for snap in getattr(store, "_snapshots", []):
        for c in candidates:
            if _paths_match(snap.source_path, c):
                matches.append(snap)
                break
    if not matches:
        return None
    return max(matches, key=lambda s: s.captured_at)


def _get_latest_drift(store: VersionStore, path_param: str) -> DriftReport | None:
    """Return the most recent drift report for a source, if any."""
    candidates = _canonical_source_candidates(path_param)
    matches: list[DriftReport] = []
    for rep in getattr(store, "_drift_reports", []):
        for c in candidates:
            if _paths_match(rep.source_path, c):
                matches.append(rep)
                break
    if not matches:
        return None
    return max(matches, key=lambda r: r.captured_at)


def _scan_latest_pipeline_json(canonical_source: str) -> dict[str, Any] | None:
    """
    Find the newest ``dictionary_result_*.json`` whose ``config.input_path`` matches.

    Args:
        canonical_source: Path or URI as stored on snapshots (authoritative).
    """
    if not _OUTPUT_GENERATED.is_dir():
        return None
    best: dict[str, Any] | None = None
    best_at = ""
    for fp in _OUTPUT_GENERATED.glob("dictionary_result_*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("pipeline_json_skip", path=str(fp), error=str(e))
            continue
        cfg_path = (data.get("config") or {}).get("input_path", "")
        if not _paths_match(str(cfg_path), canonical_source):
            continue
        gen = str(data.get("generated_at", ""))
        if gen >= best_at:
            best_at = gen
            best = data
    return best


def _relationship_strings_for_table(
    relationships: list[dict[str, Any]], table_name: str
) -> list[str]:
    """Format relationship edges that touch ``table_name`` as readable strings."""
    out: list[str] = []
    for r in relationships:
        if r.get("from_table") != table_name and r.get("to_table") != table_name:
            continue
        ft = r.get("from_table", "")
        fc = r.get("from_column", "")
        tt = r.get("to_table", "")
        tc = r.get("to_column", "")
        out.append(f"{ft}.{fc} -> {tt}.{tc}")
    return sorted(set(out))


def _pii_column_counts(data: dict[str, Any]) -> dict[str, int]:
    """Count distinct columns flagged as PII per table."""
    pr = data.get("pii_report")
    if not isinstance(pr, dict):
        return {}
    findings = pr.get("findings") or []
    per_table: dict[str, set[str]] = {}
    for f in findings:
        if not isinstance(f, dict):
            continue
        tn = str(f.get("table_name", ""))
        cn = str(f.get("column_name", ""))
        per_table.setdefault(tn, set()).add(cn)
    return {k: len(v) for k, v in per_table.items()}


def _table_summaries_from_result(data: dict[str, Any]) -> list[TableSummaryResponse]:
    """Build :class:`TableSummaryResponse` rows from serialised pipeline output."""
    schema = data.get("schema") or {}
    profile = data.get("profile") or {}
    tables_schema = {t["table_name"]: t for t in schema.get("tables", [])}
    tables_profile = {t["table_name"]: t for t in profile.get("tables", [])}
    rels = (data.get("relationships") or {}).get("relationships") or []
    pii_counts = _pii_column_counts(data)
    out: list[TableSummaryResponse] = []
    for name, ts in sorted(tables_schema.items()):
        tp = tables_profile.get(name)
        if tp is None:
            continue
        out.append(
            TableSummaryResponse(
                table_name=name,
                row_count=int(ts.get("row_count", 0)),
                column_count=int(ts.get("column_count", 0)),
                completeness_score=float(tp.get("completeness_score", 0.0)),
                has_quality_issues=bool(tp.get("has_data_quality_issues", False)),
                pii_column_count=int(pii_counts.get(name, 0)),
                relationships=_relationship_strings_for_table(rels, name),
            )
        )
    return out


def _table_summary_lookup(
    summaries: list[TableSummaryResponse], table_name: str
) -> TableSummaryResponse | None:
    """Find one table summary by name."""
    for s in summaries:
        if s.table_name == table_name:
            return s
    return None


def _validate_run_input_exists(req: RunPipelineRequest) -> None:
    """Raise HTTP 400 when the configured path is missing on disk."""
    it = _INPUT_TYPE_ALIASES[req.input_type]
    p = Path(req.input_path.strip())
    if it == InputType.CSV_FOLDER:
        if not p.exists() or not p.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"CSV folder does not exist or is not a directory: {req.input_path}",
            )
    elif it == InputType.SQLITE_FILE:
        if not p.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"SQLite file does not exist: {req.input_path}",
            )


def create_app() -> FastAPI:
    """Construct and return the configured FastAPI application."""
    application = FastAPI(
        title="Data Dictionary Agent API",
        version=API_VERSION,
        description="REST API for automated data dictionary generation",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Any:
        """Log each HTTP request with timing and status (structlog)."""
        t0 = time.perf_counter()
        logger.info(
            "http_request_start",
            method=request.method,
            path=request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "http_request_error",
                method=request.method,
                path=request.url.path,
                seconds=round(time.perf_counter() - t0, 4),
            )
            raise
        logger.info(
            "http_request_done",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            seconds=round(time.perf_counter() - t0, 4),
        )
        return response

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness probe and snapshot store size."""
        return HealthResponse(
            status="ok",
            version=API_VERSION,
            store_snapshot_count=_snapshot_count(VERSION_STORE),
        )

    @application.post("/run")
    def run(req: RunPipelineRequest) -> dict[str, Any]:
        """
        Execute the pipeline, persist a schema snapshot, render HTML report, return JSON.

        Raises:
            HTTPException: 400 for missing paths; 500 for pipeline failures.
        """
        _validate_run_input_exists(req)
        it = _INPUT_TYPE_ALIASES[req.input_type]
        cfg = PipelineConfig(
            input_path=req.input_path.strip(),
            input_type=it,
            sample_size=req.sample_size,
            enable_llm=req.enable_llm,
            enable_pii=req.enable_pii,
            output_dir=str(_OUTPUT_GENERATED),
        )
        try:
            result = run_pipeline(cfg)
        except Exception as e:
            logger.error("api_run_pipeline_failed", error=repr(e), exc_info=False)
            raise HTTPException(status_code=500, detail=str(e)) from e

        VERSION_STORE.save_snapshot(result)
        report_cfg = ReportConfig(
            output_dir=str(_OUTPUT_GENERATED),
            report_title=req.report_title,
        )
        try:
            gen_out = generate_report(result, report_cfg)
            html_path = gen_out.get("html_path", "")
            if html_path:
                _set_latest_report_path(result.config.input_path, html_path, result.generated_at)
        except Exception as e:
            logger.warning("api_report_generation_failed", error=repr(e))

        return result_to_dict(result)

    @application.get("/snapshots", response_model=list[SnapshotListItem])
    def list_snapshots() -> list[SnapshotListItem]:
        """All snapshots, newest first by ``captured_at``."""
        snaps: list[SchemaSnapshot] = list(getattr(VERSION_STORE, "_snapshots", []))
        snaps.sort(key=lambda s: s.captured_at, reverse=True)
        return [
            SnapshotListItem(
                snapshot_id=s.snapshot_id,
                source_path=s.source_path,
                captured_at=s.captured_at,
                table_count=len(s.table_names),
                schema_hash=s.schema_hash,
            )
            for s in snaps
        ]

    @application.get("/snapshots/{snapshot_id}", response_model=SnapshotDetailResponse)
    def get_snapshot(snapshot_id: str) -> SnapshotDetailResponse:
        """Return one snapshot by id."""
        for s in getattr(VERSION_STORE, "_snapshots", []):
            if s.snapshot_id == snapshot_id:
                return SnapshotDetailResponse(
                    snapshot_id=s.snapshot_id,
                    source_path=s.source_path,
                    captured_at=s.captured_at,
                    table_names=list(s.table_names),
                    column_counts={str(k): int(v) for k, v in s.column_counts.items()},
                    row_counts={str(k): int(v) for k, v in s.row_counts.items()},
                    completeness_scores={
                        str(k): float(v) for k, v in s.completeness_scores.items()
                    },
                    schema_hash=s.schema_hash,
                )
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {snapshot_id}")

    @application.get("/drift/{source_path:path}", response_model=DriftReportResponse)
    def get_latest_drift_for_source(source_path: str) -> DriftReportResponse:
        """Latest drift report for a source (404 if none)."""
        drift = _get_latest_drift(VERSION_STORE, source_path)
        if drift is None:
            raise HTTPException(
                status_code=404,
                detail="No drift report found for this source yet.",
            )
        d = drift_report_to_dict(drift)
        return DriftReportResponse.model_validate(d)

    @application.get(
        "/tables/{source_path:path}/{table_name}",
        response_model=TableDetailResponse,
    )
    def get_table_detail(source_path: str, table_name: str) -> TableDetailResponse:
        """Schema, profile, relationships, PII, and LLM slice for one table."""
        snap = _get_latest_snapshot(VERSION_STORE, source_path)
        if snap is None:
            raise HTTPException(
                status_code=404,
                detail="No snapshot exists for this source path.",
            )
        data = _scan_latest_pipeline_json(snap.source_path)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail="No saved pipeline JSON found for the latest snapshot.",
            )
        summaries = _table_summaries_from_result(data)
        if _table_summary_lookup(summaries, table_name) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Table not found in latest result: {table_name}",
            )
        schema_tables = {t["table_name"]: t for t in (data.get("schema") or {}).get("tables", [])}
        prof_tables = {t["table_name"]: t for t in (data.get("profile") or {}).get("tables", [])}
        ts = schema_tables.get(table_name)
        tp = prof_tables.get(table_name)
        if ts is None or tp is None:
            raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")
        rels = (data.get("relationships") or {}).get("relationships") or []
        pii_block = data.get("pii_report")
        findings: list[dict[str, Any]] = []
        if isinstance(pii_block, dict):
            for f in pii_block.get("findings") or []:
                if isinstance(f, dict) and f.get("table_name") == table_name:
                    findings.append(dict(f))
        summary_obj: dict[str, Any] | None = None
        summary_root = data.get("summary")
        if isinstance(summary_root, dict):
            for tsum in summary_root.get("tables") or []:
                if isinstance(tsum, dict) and tsum.get("table_name") == table_name:
                    summary_obj = dict(tsum)
                    break
        return TableDetailResponse(
            table_name=table_name,
            table_schema=dict(ts),
            profile=dict(tp),
            relationships=_relationship_strings_for_table(rels, table_name),
            pii_findings=findings,
            summary=summary_obj,
        )

    @application.get("/tables/{source_path:path}", response_model=list[TableSummaryResponse])
    def list_tables_for_source(source_path: str) -> list[TableSummaryResponse]:
        """Table roll-ups from the newest saved pipeline JSON for this source."""
        snap = _get_latest_snapshot(VERSION_STORE, source_path)
        if snap is None:
            raise HTTPException(
                status_code=404,
                detail="No snapshot exists for this source path.",
            )
        data = _scan_latest_pipeline_json(snap.source_path)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail="No saved pipeline JSON found for the latest snapshot.",
            )
        return _table_summaries_from_result(data)

    @application.get("/pii/{source_path:path}")
    def get_pii_for_source(source_path: str) -> dict[str, Any]:
        """PII report object from the latest pipeline JSON for this source."""
        snap = _get_latest_snapshot(VERSION_STORE, source_path)
        if snap is None:
            raise HTTPException(
                status_code=404,
                detail="No snapshot exists for this source path.",
            )
        data = _scan_latest_pipeline_json(snap.source_path)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail="No saved pipeline JSON found for the latest snapshot.",
            )
        pr = data.get("pii_report")
        if pr is None:
            return {
                "findings": [],
                "total_pii_columns": 0,
                "tables_with_pii": [],
                "scan_summary": {},
            }
        if isinstance(pr, dict):
            return dict(pr)
        return {"raw": pr}

    @application.get("/report/{source_path:path}")
    def download_report(source_path: str) -> FileResponse:
        """Download the most recent HTML report recorded for this source."""
        snap = _get_latest_snapshot(VERSION_STORE, source_path)
        if snap is None:
            raise HTTPException(
                status_code=404,
                detail="No snapshot exists for this source path.",
            )
        key = _artifact_storage_key(snap.source_path)
        idx = _load_report_index()
        entry = idx.get(key)
        if not entry:
            raise HTTPException(
                status_code=404,
                detail="No HTML report on record for this source; run POST /run first.",
            )
        html_path = Path(entry["html_path"])
        if not html_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Report file missing on disk: {html_path}",
            )
        return FileResponse(
            path=str(html_path),
            media_type="text/html",
            filename=html_path.name,
        )

    return application


app = create_app()


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    uvicorn.run(
        "api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

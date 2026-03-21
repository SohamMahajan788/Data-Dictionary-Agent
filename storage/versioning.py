"""
Snapshot and drift tracking for pipeline outputs over time.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog

from orchestrator.pipeline import DictionaryResult, InputType, PipelineConfig, run_pipeline

logger = structlog.get_logger(__name__)


@dataclass
class SchemaSnapshot:
    """Stored snapshot of one pipeline run for a specific source."""

    snapshot_id: str
    source_path: str
    captured_at: str
    table_names: list[str]
    column_counts: dict
    row_counts: dict
    completeness_scores: dict
    schema_hash: str


@dataclass
class DriftReport:
    """Differences detected between two snapshots."""

    snapshot_id_before: str
    snapshot_id_after: str
    captured_at: str
    source_path: str
    tables_added: list[str]
    tables_removed: list[str]
    columns_added: dict
    columns_removed: dict
    completeness_changes: dict
    row_count_changes: dict
    has_breaking_changes: bool
    summary: str


def _utc_iso() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _snapshot_from_dict(data: dict) -> SchemaSnapshot:
    """Hydrate `SchemaSnapshot` from persisted dict."""
    return SchemaSnapshot(
        snapshot_id=str(data["snapshot_id"]),
        source_path=str(data["source_path"]),
        captured_at=str(data["captured_at"]),
        table_names=list(data.get("table_names", [])),
        column_counts=dict(data.get("column_counts", {})),
        row_counts=dict(data.get("row_counts", {})),
        completeness_scores=dict(data.get("completeness_scores", {})),
        schema_hash=str(data.get("schema_hash", "")),
    )


def _drift_from_dict(data: dict) -> DriftReport:
    """Hydrate `DriftReport` from persisted dict."""
    return DriftReport(
        snapshot_id_before=str(data["snapshot_id_before"]),
        snapshot_id_after=str(data["snapshot_id_after"]),
        captured_at=str(data["captured_at"]),
        source_path=str(data["source_path"]),
        tables_added=list(data.get("tables_added", [])),
        tables_removed=list(data.get("tables_removed", [])),
        columns_added=dict(data.get("columns_added", {})),
        columns_removed=dict(data.get("columns_removed", {})),
        completeness_changes=dict(data.get("completeness_changes", {})),
        row_count_changes=dict(data.get("row_count_changes", {})),
        has_breaking_changes=bool(data.get("has_breaking_changes", False)),
        summary=str(data.get("summary", "")),
    )


def _schema_hash(result: DictionaryResult) -> str:
    """
    Compute stable schema hash from sorted table/column names.

    Hash input format:
    - one line per table: `table_name:col1,col2,...`
    - tables and columns sorted lexicographically
    """
    lines: list[str] = []
    for ts in sorted(result.schema.tables, key=lambda t: t.table_name):
        cols = sorted(c.name for c in ts.columns)
        lines.append(f"{ts.table_name}:{','.join(cols)}")
    raw = "\n".join(lines).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def drift_report_to_dict(report: DriftReport) -> dict:
    """Serialize drift report to plain dict."""
    return asdict(report)


class VersionStore:
    """Persistent store for schema snapshots and drift history."""

    def __init__(self, store_path: str = "./storage/version_store.json"):
        """
        Initialize store from JSON file or create a new one.

        The store file keeps:
        - `snapshots`: list of snapshot dicts
        - `drift_reports`: list of drift report dicts
        - `snapshot_columns`: snapshot_id -> table_name -> [column names]
        """
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshots: list[SchemaSnapshot] = []
        self._drift_reports: list[DriftReport] = []
        self._snapshot_columns: dict[str, dict[str, list[str]]] = {}
        self._load_store()

    def _load_store(self) -> None:
        """Load persisted store data from disk."""
        if not self.store_path.exists():
            logger.info("version_store_init_empty", store_path=str(self.store_path.resolve()))
            return
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
            self._snapshots = [_snapshot_from_dict(x) for x in payload.get("snapshots", [])]
            self._drift_reports = [_drift_from_dict(x) for x in payload.get("drift_reports", [])]
            sc = payload.get("snapshot_columns", {})
            self._snapshot_columns = {
                str(k): {str(t): list(cols) for t, cols in dict(v).items()}
                for k, v in dict(sc).items()
            }
            logger.info(
                "version_store_loaded",
                store_path=str(self.store_path.resolve()),
                snapshots=len(self._snapshots),
                drift_reports=len(self._drift_reports),
            )
        except Exception as e:
            logger.warning(
                "version_store_load_failed",
                store_path=str(self.store_path.resolve()),
                error=str(e),
                exc_info=True,
            )
            self._snapshots = []
            self._drift_reports = []
            self._snapshot_columns = {}

    def _save_store(self) -> None:
        """Persist in-memory store to JSON."""
        payload = {
            "snapshots": [asdict(s) for s in self._snapshots],
            "drift_reports": [asdict(d) for d in self._drift_reports],
            "snapshot_columns": self._snapshot_columns,
        }
        self.store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "version_store_saved",
            store_path=str(self.store_path.resolve()),
            snapshots=len(self._snapshots),
            drift_reports=len(self._drift_reports),
        )

    def save_snapshot(self, result: DictionaryResult) -> SchemaSnapshot:
        """
        Create and persist a snapshot from one pipeline result.

        Returns:
            The saved snapshot.
        """
        sid = str(uuid.uuid4())
        table_names = sorted(t.table_name for t in result.schema.tables)
        col_counts = {t.table_name: t.column_count for t in result.schema.tables}
        row_counts = {t.table_name: t.row_count for t in result.schema.tables}
        comp_scores = {t.table_name: t.completeness_score for t in result.profile.tables}

        snapshot = SchemaSnapshot(
            snapshot_id=sid,
            source_path=result.config.input_path,
            captured_at=_utc_iso(),
            table_names=table_names,
            column_counts=col_counts,
            row_counts=row_counts,
            completeness_scores=comp_scores,
            schema_hash=_schema_hash(result),
        )
        self._snapshots.append(snapshot)
        self._snapshot_columns[sid] = {
            t.table_name: sorted(c.name for c in t.columns) for t in result.schema.tables
        }
        self._save_store()
        logger.info(
            "snapshot_saved",
            snapshot_id=snapshot.snapshot_id,
            source_path=snapshot.source_path,
            table_count=len(snapshot.table_names),
        )
        return snapshot

    def get_snapshots(self, source_path: str) -> list[SchemaSnapshot]:
        """Return snapshots for `source_path`, sorted oldest -> newest."""
        out = [s for s in self._snapshots if s.source_path == source_path]
        out.sort(key=lambda s: s.captured_at)
        return out

    def get_latest_snapshot(self, source_path: str) -> SchemaSnapshot | None:
        """Return latest snapshot for `source_path`, if any."""
        snaps = self.get_snapshots(source_path)
        if not snaps:
            return None
        return snaps[-1]

    def compare_snapshots(self, before: SchemaSnapshot, after: SchemaSnapshot) -> DriftReport:
        """
        Compare two snapshots and compute a drift report.

        Threshold rules:
        - completeness change reported if abs(delta) > 0.02
        - row count change reported if abs(delta_pct) > 0.05
        - breaking if any tables removed or columns removed
        """
        before_tables = set(before.table_names)
        after_tables = set(after.table_names)
        tables_added = sorted(after_tables - before_tables)
        tables_removed = sorted(before_tables - after_tables)

        before_cols = self._snapshot_columns.get(before.snapshot_id, {})
        after_cols = self._snapshot_columns.get(after.snapshot_id, {})
        shared = sorted(before_tables & after_tables)

        columns_added: dict[str, list[str]] = {}
        columns_removed: dict[str, list[str]] = {}
        for t in shared:
            bset = set(before_cols.get(t, []))
            aset = set(after_cols.get(t, []))
            add = sorted(aset - bset)
            rem = sorted(bset - aset)
            if add:
                columns_added[t] = add
            if rem:
                columns_removed[t] = rem

        completeness_changes: dict[str, dict] = {}
        for t in shared:
            b = float(before.completeness_scores.get(t, 0.0))
            a = float(after.completeness_scores.get(t, 0.0))
            delta = a - b
            if abs(delta) > 0.02:
                completeness_changes[t] = {
                    "before": b,
                    "after": a,
                    "delta": delta,
                }

        row_count_changes: dict[str, dict] = {}
        for t in shared:
            b = int(before.row_counts.get(t, 0))
            a = int(after.row_counts.get(t, 0))
            if b == 0:
                delta_pct = 0.0 if a == 0 else 1.0
            else:
                delta_pct = (a - b) / b
            if abs(delta_pct) > 0.05:
                row_count_changes[t] = {
                    "before": b,
                    "after": a,
                    "delta_pct": delta_pct,
                }

        has_breaking = bool(tables_removed) or any(columns_removed.values())
        summary_parts: list[str] = []
        if tables_added:
            summary_parts.append(f"{len(tables_added)} table(s) added")
        if tables_removed:
            summary_parts.append(f"{len(tables_removed)} table(s) removed")
        if columns_added:
            summary_parts.append(f"columns added in {len(columns_added)} table(s)")
        if columns_removed:
            summary_parts.append(f"columns removed in {len(columns_removed)} table(s)")
        if completeness_changes:
            summary_parts.append(
                f"completeness shifted in {len(completeness_changes)} table(s)"
            )
        if row_count_changes:
            summary_parts.append(f"row-count drift in {len(row_count_changes)} table(s)")
        if not summary_parts:
            summary = "No drift detected between snapshots."
        else:
            summary = "; ".join(summary_parts) + "."

        report = DriftReport(
            snapshot_id_before=before.snapshot_id,
            snapshot_id_after=after.snapshot_id,
            captured_at=_utc_iso(),
            source_path=after.source_path,
            tables_added=tables_added,
            tables_removed=tables_removed,
            columns_added=columns_added,
            columns_removed=columns_removed,
            completeness_changes=completeness_changes,
            row_count_changes=row_count_changes,
            has_breaking_changes=has_breaking,
            summary=summary,
        )
        logger.info(
            "snapshot_compared",
            source_path=report.source_path,
            before=report.snapshot_id_before,
            after=report.snapshot_id_after,
            breaking=report.has_breaking_changes,
            summary=report.summary,
        )
        return report

    def detect_drift(self, result: DictionaryResult) -> DriftReport | None:
        """
        Save a new snapshot and compare to previous snapshot for same source.

        Returns:
            Drift report if a previous snapshot exists, else None.
        """
        previous = self.get_latest_snapshot(result.config.input_path)
        latest = self.save_snapshot(result)
        if previous is None:
            logger.info("drift_not_available_first_run", source_path=result.config.input_path)
            return None
        report = self.compare_snapshots(previous, latest)
        self._drift_reports.append(report)
        self._save_store()
        logger.info("drift_detected", source_path=result.config.input_path)
        return report

    def get_drift_history(self, source_path: str) -> list[DriftReport]:
        """Return drift reports for one source, oldest -> newest."""
        out = [d for d in self._drift_reports if d.source_path == source_path]
        out.sort(key=lambda d: d.captured_at)
        return out


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
    _source = str(_chinook.resolve())
    _store = VersionStore()

    _cfg = PipelineConfig(
        input_path=_source,
        input_type=InputType.SQLITE_FILE,
        sample_size=10000,
        enable_llm=False,
        enable_pii=True,
    )

    _result_1 = run_pipeline(_cfg)
    _drift_1 = _store.detect_drift(_result_1)
    if _drift_1 is None:
        print("First run - no drift to compare")

    _result_2 = run_pipeline(_cfg)
    _drift_2 = _store.detect_drift(_result_2)
    if _drift_2 is None:
        print("No previous snapshot found.")
    else:
        print("\nSecond run drift report:")
        print(json.dumps(drift_report_to_dict(_drift_2), indent=2))

    _history = _store.get_snapshots(_source)
    print("\nSnapshot history for chinook:")
    for s in _history:
        print(f"- {s.snapshot_id} @ {s.captured_at} tables={len(s.table_names)}")

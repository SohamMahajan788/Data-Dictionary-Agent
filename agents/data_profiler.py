"""
Compute per-column and per-table data-quality metrics from a live database sample.

Uses a row cap (``LIMIT`` on SQLite) so profiling stays bounded on large tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine

from agents.schema_extractor import DatabaseSchema, TableSchema

logger = structlog.get_logger(__name__)


@dataclass
class ColumnProfile:
    """Statistical summary for one column over the sampled rows."""

    column_name: str
    data_type: str
    total_rows: int
    null_count: int
    null_rate: float
    unique_count: int
    uniqueness_rate: float
    min_value: str | None
    max_value: str | None
    mean_value: float | None
    sample_values: list[str]
    is_constant: bool
    is_unique_key: bool


@dataclass
class TableProfile:
    """Aggregated metrics for a single table."""

    table_name: str
    total_rows: int
    total_columns: int
    columns: list[ColumnProfile]
    completeness_score: float
    duplicate_row_count: int
    has_data_quality_issues: bool


@dataclass
class DatabaseProfile:
    """Full profiling result for a database."""

    tables: list[TableProfile]
    total_tables: int
    overall_completeness: float
    profiled_at: str


def _quote_ident(name: str) -> str:
    """
    Quote a SQLite identifier safely.

    Args:
        name: Table or column name from trusted schema metadata.

    Returns:
        Double-quoted identifier with internal quotes escaped.
    """
    return '"' + name.replace('"', '""') + '"'


def _is_numeric_sql_type(data_type: str) -> bool:
    """
    Return True if the reflected SQL type name suggests numeric aggregates.

    Args:
        data_type: Type string from :class:`ColumnInfo`.

    Returns:
        Whether ``mean_value`` / numeric min/max are meaningful.
    """
    t = data_type.upper()
    keywords = (
        "INT",
        "INTEGER",
        "BIGINT",
        "SMALLINT",
        "REAL",
        "FLOAT",
        "DOUBLE",
        "NUMERIC",
        "DECIMAL",
    )
    return any(k in t for k in keywords)


def _null_mask(series: pd.Series) -> pd.Series:
    """True where values are missing (NaN/NA)."""
    return series.isna()


def _display_value(value: object) -> str:
    """Format a single cell for ``sample_values`` and min/max strings."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _profile_column(
    col_name: str,
    data_type: str,
    series: pd.Series,
    total_rows: int,
) -> ColumnProfile:
    """
    Build a :class:`ColumnProfile` from a pandas Series (already truncated sample).

    Args:
        col_name: Column name.
        data_type: Declared SQL type string.
        series: Sampled column values.
        total_rows: Number of rows in the sample (``len(series)``).

    Returns:
        Populated column profile.
    """
    if total_rows == 0:
        return ColumnProfile(
            column_name=col_name,
            data_type=data_type,
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

    null_mask = _null_mask(series)
    null_count = int(null_mask.sum())
    non_null = series[~null_mask]
    null_rate = null_count / total_rows

    unique_count = int(non_null.nunique(dropna=False))
    uniqueness_rate = unique_count / total_rows

    sample_vals = (
        non_null.drop_duplicates()
        .head(5)
        .map(_display_value)
        .tolist()
    )

    min_value: str | None = None
    max_value: str | None = None
    mean_value: float | None = None

    if len(non_null) > 0:
        if _is_numeric_sql_type(data_type):
            num = pd.to_numeric(non_null, errors="coerce")
            valid = num.dropna()
            if len(valid) > 0:
                mean_value = float(valid.mean())
                min_value = _display_value(valid.min())
                max_value = _display_value(valid.max())
            else:
                sorted_str = sorted(non_null.astype(str).unique())
                min_value = sorted_str[0]
                max_value = sorted_str[-1]
        else:
            str_vals = non_null.astype(str)
            sorted_str = sorted(str_vals.unique())
            min_value = sorted_str[0]
            max_value = sorted_str[-1]

    is_constant = unique_count == 1
    is_unique_key = uniqueness_rate == 1.0

    return ColumnProfile(
        column_name=col_name,
        data_type=data_type,
        total_rows=total_rows,
        null_count=null_count,
        null_rate=null_rate,
        unique_count=unique_count,
        uniqueness_rate=uniqueness_rate,
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
        sample_values=sample_vals,
        is_constant=is_constant,
        is_unique_key=is_unique_key,
    )


def _read_sample(
    engine: Engine,
    table_name: str,
    limit: int,
) -> pd.DataFrame:
    """
    Load up to ``limit`` rows from ``table_name`` using SQLite ``LIMIT``.

    Args:
        engine: SQLAlchemy engine.
        table_name: Table from schema (trusted).
        limit: Maximum rows to read.

    Returns:
        DataFrame of the sample (possibly empty).
    """
    q = text(f"SELECT * FROM {_quote_ident(table_name)} LIMIT :lim")
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn, params={"lim": int(limit)})


def _table_sample_limit(ts: TableSchema, sample_size: int) -> int:
    """Effective sample size capped by declared row count."""
    if ts.row_count <= 0:
        return 0
    return min(int(ts.row_count), int(sample_size))


def _profile_table(
    engine: Engine,
    ts: TableSchema,
    sample_size: int,
) -> TableProfile:
    """
    Profile one table: load sample, per-column stats, duplicates, completeness.

    Args:
        engine: Live database engine.
        ts: Table schema entry.
        sample_size: Upper bound on rows to scan.

    Returns:
        :class:`TableProfile` for ``ts``.
    """
    logger.info(
        "profile_table_start",
        table=ts.table_name,
        declared_rows=ts.row_count,
        sample_cap=sample_size,
    )
    lim = _table_sample_limit(ts, sample_size)
    if lim == 0:
        cols_out = [
            _profile_column(c.name, c.data_type, pd.Series(dtype=object), 0)
            for c in ts.columns
        ]
        tp = TableProfile(
            table_name=ts.table_name,
            total_rows=0,
            total_columns=len(cols_out),
            columns=cols_out,
            completeness_score=1.0,
            duplicate_row_count=0,
            has_data_quality_issues=False,
        )
        logger.info(
            "profile_table_done",
            table=ts.table_name,
            sample_rows=0,
            completeness=tp.completeness_score,
            duplicates=0,
        )
        return tp

    df = _read_sample(engine, ts.table_name, lim)
    total_rows = len(df)
    dup_count = int(df.duplicated().sum())

    col_profiles: list[ColumnProfile] = []
    for col_info in ts.columns:
        if col_info.name not in df.columns:
            logger.warning(
                "profile_column_missing_in_sample",
                table=ts.table_name,
                column=col_info.name,
            )
            series = pd.Series([None] * total_rows)
        else:
            series = df[col_info.name]
        col_profiles.append(
            _profile_column(col_info.name, col_info.data_type, series, total_rows)
        )

    if col_profiles:
        completeness = sum(1.0 - c.null_rate for c in col_profiles) / len(
            col_profiles
        )
    else:
        completeness = 1.0

    bad_null = any(c.null_rate > 0.3 for c in col_profiles)
    has_issues = bad_null or dup_count > 0

    tp = TableProfile(
        table_name=ts.table_name,
        total_rows=total_rows,
        total_columns=len(col_profiles),
        columns=col_profiles,
        completeness_score=completeness,
        duplicate_row_count=dup_count,
        has_data_quality_issues=has_issues,
    )
    logger.info(
        "profile_table_done",
        table=ts.table_name,
        sample_rows=total_rows,
        completeness=round(completeness, 6),
        duplicates=dup_count,
        quality_flag=has_issues,
    )
    return tp


def profile_database(
    engine: Engine,
    schema: DatabaseSchema,
    sample_size: int = 10000,
) -> DatabaseProfile:
    """
    Profile every table in ``schema`` using at most ``sample_size`` rows per table.

    On SQLite, sampling is implemented with ``SELECT ... LIMIT n`` (no ``TABLESAMPLE``).
    Duplicate detection uses pandas :meth:`DataFrame.duplicated` on the same sample.

    Args:
        engine: Engine returned by a connector.
        schema: Schema from :func:`agents.schema_extractor.extract_schema`.
        sample_size: Maximum rows to load per table.

    Returns:
        :class:`DatabaseProfile` with ISO ``profiled_at`` timestamp (UTC).
    """
    logger.info(
        "profile_database_start",
        source_path=schema.source_path,
        total_tables=schema.total_tables,
        sample_size=sample_size,
    )
    tables_out: list[TableProfile] = []
    for ts in schema.tables:
        tables_out.append(_profile_table(engine, ts, sample_size))

    if tables_out:
        overall = sum(t.completeness_score for t in tables_out) / len(tables_out)
    else:
        overall = 1.0

    stamp = datetime.now(timezone.utc).isoformat()
    result = DatabaseProfile(
        tables=tables_out,
        total_tables=len(tables_out),
        overall_completeness=overall,
        profiled_at=stamp,
    )
    logger.info(
        "profile_database_done",
        source_path=schema.source_path,
        total_tables=result.total_tables,
        overall_completeness=round(overall, 6),
        profiled_at=stamp,
    )
    return result


def profile_to_dict(profile: DatabaseProfile) -> dict:
    """
    Convert a :class:`DatabaseProfile` to nested dicts for JSON encoding.

    Args:
        profile: Result of :func:`profile_database`.

    Returns:
        JSON-friendly structure.
    """
    return asdict(profile)


def _print_table_high_nulls(tp: TableProfile, top_n: int = 3) -> None:
    """
    Print completeness, duplicate count, and the top ``top_n`` columns by null rate.

    Args:
        tp: Table profile to summarise.
        top_n: How many high-null columns to show.
    """
    ranked = sorted(tp.columns, key=lambda c: c.null_rate, reverse=True)
    top = ranked[:top_n]
    top_desc = [f"{c.column_name}={c.null_rate:.3f}" for c in top]
    print(
        f"  {tp.table_name}: completeness={tp.completeness_score:.4f} "
        f"duplicates={tp.duplicate_row_count} issues={tp.has_data_quality_issues} "
        f"top_nulls={top_desc}"
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

    from agents.schema_extractor import extract_schema

    _olist = _proj / "sample_data" / "olist"
    _chinook = _proj / "sample_data" / "chinook" / "Chinook_Sqlite.sqlite"

    print("\n--- olist ---")
    _e1 = load_csv_folder(str(_olist))
    _s1 = extract_schema(_e1, str(_olist.resolve()))
    _p1 = profile_database(_e1, _s1, sample_size=10000)
    for t in _p1.tables:
        _print_table_high_nulls(t)

    print("\n--- chinook ---")
    _e2 = load_sqlite(str(_chinook))
    _s2 = extract_schema(_e2, str(_chinook.resolve()))
    _p2 = profile_database(_e2, _s2, sample_size=10000)
    for t in _p2.tables:
        _print_table_high_nulls(t)

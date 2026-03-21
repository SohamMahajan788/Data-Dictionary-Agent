"""
Extract full database schema (tables, columns, keys, constraints) from a SQLAlchemy engine.

Feeds the data-dictionary pipeline and downstream JSON serialisation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import structlog
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

logger = structlog.get_logger(__name__)


@dataclass
class ColumnInfo:
    """Metadata for a single column."""

    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    foreign_key_references: str | None
    default_value: str | None


@dataclass
class TableSchema:
    """Schema snapshot for one table."""

    table_name: str
    columns: list[ColumnInfo]
    primary_keys: list[str]
    foreign_keys: list[dict]
    row_count: int
    column_count: int


@dataclass
class DatabaseSchema:
    """Complete schema for a database behind ``engine``."""

    tables: list[TableSchema]
    total_tables: int
    total_columns: int
    source_path: str


def _default_to_str(default: object | None) -> str | None:
    """
    Normalise SQLAlchemy column default to a string for serialisation.

    Args:
        default: Raw default from inspector (may be ``None`` or dialect-specific).

    Returns:
        ``None`` if unset, otherwise ``str(default)``.
    """
    if default is None:
        return None
    return str(default)


def _normalize_foreign_key(fk: dict) -> dict:
    """
    Copy foreign-key metadata from the inspector into JSON-friendly primitives.

    Args:
        fk: Dict returned by :meth:`sqlalchemy.engine.reflection.Inspector.get_foreign_keys`.

    Returns:
        A new dict with list/dict/str values suitable for :func:`schema_to_dict`.
    """
    opts = fk.get("options")
    return {
        "name": fk.get("name"),
        "constrained_columns": list(fk.get("constrained_columns") or []),
        "referred_schema": fk.get("referred_schema"),
        "referred_table": fk.get("referred_table"),
        "referred_columns": list(fk.get("referred_columns") or []),
        "options": dict(opts) if isinstance(opts, dict) else opts,
    }


def _column_foreign_key_targets(foreign_keys: list[dict]) -> dict[str, str]:
    """
    Map local column names to ``referred_table.referred_column`` strings.

    Args:
        foreign_keys: Normalised FK dicts (see :func:`_normalize_foreign_key`).

    Returns:
        Mapping from constrained column name to a single reference string.
    """
    targets: dict[str, str] = {}
    for fk in foreign_keys:
        table = fk.get("referred_table") or ""
        for local_col, remote_col in zip(
            fk.get("constrained_columns") or [],
            fk.get("referred_columns") or [],
        ):
            ref = f"{table}.{remote_col}" if table else str(remote_col)
            targets[str(local_col)] = ref
    return targets


def _row_count(engine: Engine, table: str) -> int:
    """
    Return ``COUNT(*)`` for ``table`` using a quoted SQLite-style identifier.

    Args:
        engine: Connected SQLAlchemy engine.
        table: Table name from reflection.

    Returns:
        Number of rows.

    Raises:
        Exception: If the query fails (e.g. invalid identifier).
    """
    safe = table.replace('"', '""')
    stmt = f'SELECT COUNT(*) AS c FROM "{safe}"'
    with engine.connect() as conn:
        return int(conn.exec_driver_sql(stmt).scalar_one())


def _extract_table_schema(engine: Engine, table_name: str) -> TableSchema:
    """
    Reflect one table into a :class:`TableSchema`.

    Args:
        engine: SQLAlchemy engine.
        table_name: Name of the table to reflect.

    Returns:
        Populated :class:`TableSchema`.

    Raises:
        Exception: Reflection or row-count failures propagate after logging.
    """
    logger.info("schema_extract_table_start", table=table_name)
    insp = inspect(engine)

    raw_fks = insp.get_foreign_keys(table_name)
    foreign_keys = [_normalize_foreign_key(fk) for fk in raw_fks]
    fk_targets = _column_foreign_key_targets(foreign_keys)

    pk_info = insp.get_pk_constraint(table_name)
    pk_cols = list(pk_info.get("constrained_columns") or [])

    columns_out: list[ColumnInfo] = []
    for col in insp.get_columns(table_name):
        cname = str(col["name"])
        dtype = str(col.get("type", ""))
        nullable = bool(col.get("nullable", True))
        is_pk = cname in pk_cols
        is_fk = cname in fk_targets
        ref = fk_targets.get(cname) if is_fk else None
        columns_out.append(
            ColumnInfo(
                name=cname,
                data_type=dtype,
                is_nullable=nullable,
                is_primary_key=is_pk,
                is_foreign_key=is_fk,
                foreign_key_references=ref,
                default_value=_default_to_str(col.get("default")),
            )
        )

    rows = _row_count(engine, table_name)
    col_count = len(columns_out)

    logger.info(
        "schema_extract_table_done",
        table=table_name,
        columns=col_count,
        rows=rows,
        primary_keys=pk_cols,
        foreign_key_constraints=len(foreign_keys),
    )

    return TableSchema(
        table_name=table_name,
        columns=columns_out,
        primary_keys=pk_cols,
        foreign_keys=foreign_keys,
        row_count=rows,
        column_count=col_count,
    )


def extract_schema(engine: Engine, source_path: str) -> DatabaseSchema:
    """
    Build a :class:`DatabaseSchema` for every table visible to ``engine``.

    Uses :func:`sqlalchemy.inspect` for columns, primary keys, and foreign keys,
    and ``COUNT(*)`` per table for row counts.

    Args:
        engine: Engine from any project connector.
        source_path: Logical source (folder path, file path, or URI) for reporting.

    Returns:
        Fully populated :class:`DatabaseSchema`.

    Raises:
        Exception: Propagates reflection or connection errors after logging.
    """
    logger.info("schema_extract_start", source_path=source_path)
    try:
        insp = inspect(engine)
        table_names = sorted(insp.get_table_names())
    except Exception as e:
        logger.error(
            "schema_extract_inspect_failed",
            source_path=source_path,
            error=str(e),
            exc_info=True,
        )
        raise

    if not table_names:
        logger.warning("schema_extract_no_tables", source_path=source_path)

    tables: list[TableSchema] = []
    for name in table_names:
        try:
            tables.append(_extract_table_schema(engine, name))
        except Exception as e:
            logger.error(
                "schema_extract_table_failed",
                table=name,
                source_path=source_path,
                error=str(e),
                exc_info=True,
            )
            raise

    total_columns = sum(t.column_count for t in tables)
    result = DatabaseSchema(
        tables=tables,
        total_tables=len(tables),
        total_columns=total_columns,
        source_path=source_path,
    )
    logger.info(
        "schema_extract_finished",
        source_path=source_path,
        total_tables=result.total_tables,
        total_columns=result.total_columns,
    )
    return result


def schema_to_dict(schema: DatabaseSchema) -> dict:
    """
    Convert a :class:`DatabaseSchema` to a plain dict for JSON encoding.

    Args:
        schema: Output of :func:`extract_schema`.

    Returns:
        Nested dict/list structure with only primitive-friendly values.
    """
    return asdict(schema)


def _print_dataset_summary(label: str, schema: DatabaseSchema) -> None:
    """
    Print per-table column count, row count, PKs, and FK constraint summaries.

    Args:
        label: Human-readable dataset name (e.g. ``olist``).
        schema: Schema to summarise.
    """
    print(f"\n=== {label} ===")
    print(f"source_path={schema.source_path}")
    print(f"total_tables={schema.total_tables} total_columns={schema.total_columns}")
    for t in schema.tables:
        fk_parts: list[str] = []
        for fk in t.foreign_keys:
            rt = fk.get("referred_table") or ""
            for local_col, remote_col in zip(
                fk.get("constrained_columns") or [],
                fk.get("referred_columns") or [],
            ):
                fk_parts.append(f"{local_col} -> {rt}.{remote_col}")
        fk_label = fk_parts if fk_parts else "[]"
        print(
            f"  {t.table_name}: columns={t.column_count} rows={t.row_count} "
            f"PKs={t.primary_keys} FKs={fk_label}"
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

    _olist = _proj / "sample_data" / "olist"
    _chinook = _proj / "sample_data" / "chinook" / "Chinook_Sqlite.sqlite"

    _engine_olist = load_csv_folder(str(_olist))
    _schema_olist = extract_schema(_engine_olist, str(_olist.resolve()))
    _print_dataset_summary("olist (csv_loader -> in-memory SQLite)", _schema_olist)

    _engine_chinook = load_sqlite(str(_chinook))
    _schema_chinook = extract_schema(_engine_chinook, str(_chinook.resolve()))
    _print_dataset_summary("chinook (sqlite_connector)", _schema_chinook)

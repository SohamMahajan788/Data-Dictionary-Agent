"""
Connect to an on-disk SQLite database and expose a SQLAlchemy engine.

Used by the data-dictionary pipeline for bundled SQLite samples (e.g. Chinook).
"""

from __future__ import annotations

from pathlib import Path

import structlog
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from connectors.csv_loader import get_table_names

logger = structlog.get_logger(__name__)

_SQLITE_MAGIC = b"SQLite format 3\x00"


def _read_sqlite_header_bytes(path: Path) -> bytes:
    """
    Read the first 16 bytes from disk for SQLite header validation.

    Args:
        path: Path to the candidate database file.

    Returns:
        Exactly 16 bytes (or fewer if the file is shorter than 16 bytes).
    """
    with path.open("rb") as f:
        return f.read(16)


def load_sqlite(filepath: str) -> Engine:
    """
    Open a SQLite database file and return a SQLAlchemy engine attached to it.

    Verifies the path exists, is a regular file, and begins with the standard
    SQLite 16-byte header (``SQLite format 3`` plus null terminator).

    Args:
        filepath: Path to a ``.sqlite`` or ``.db`` file.

    Returns:
        A SQLAlchemy :class:`~sqlalchemy.engine.Engine` using ``sqlite:///`` and
        the resolved file path.

    Raises:
        FileNotFoundError: If the path does not exist or is not a file.
        ValueError: If the file does not contain a valid SQLite header.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"SQLite database file does not exist: {path.resolve()}"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"SQLite path is not a regular file: {path.resolve()}"
        )

    header = _read_sqlite_header_bytes(path)
    if header != _SQLITE_MAGIC:
        raise ValueError(
            "File is not a valid SQLite database (missing 'SQLite format 3' header): "
            f"{path.resolve()}"
        )

    url = f"sqlite:///{path.resolve().as_posix()}"
    engine = create_engine(url, future=True)

    tables = get_table_names(engine)
    logger.info(
        "sqlite_loaded",
        filepath=str(path.resolve()),
        table_count=len(tables),
        tables=tables,
    )
    return engine


def get_row_count(engine: Engine, table: str) -> int:
    """
    Return the number of rows in ``table`` using ``SELECT COUNT(*)``.

    The table name is escaped for SQLite identifier quoting (embedded ``"`` doubled).

    Args:
        engine: SQLAlchemy engine connected to the database.
        table: Table name as returned by :func:`get_table_names`.

    Returns:
        Row count as an integer.

    Raises:
        Exception: Propagates database errors if the table is missing or the query fails.
    """
    safe = table.replace('"', '""')
    stmt = f'SELECT COUNT(*) AS c FROM "{safe}"'
    with engine.connect() as conn:
        count = conn.exec_driver_sql(stmt).scalar_one()
    return int(count)


if __name__ == "__main__":
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    _base = Path(__file__).resolve().parent.parent
    _chinook = _base / "sample_data" / "chinook" / "Chinook_Sqlite.sqlite"
    _engine = load_sqlite(str(_chinook))
    _names = get_table_names(_engine)
    print("Tables:", _names)
    for t in _names:
        n = get_row_count(_engine, t)
        print(f"  {t}: rows={n}")

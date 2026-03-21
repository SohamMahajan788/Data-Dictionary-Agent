"""
Load all CSV files from a folder into an in-memory SQLite database via SQLAlchemy.

Used by the data-dictionary pipeline as the universal CSV input connector.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

logger = structlog.get_logger(__name__)


def detect_delimiter(filepath: str) -> str:
    """
    Auto-detect whether a CSV uses comma, semicolon, or tab as the field delimiter.

    Reads a sample of the file and delegates to :class:`csv.Sniffer`. Falls back to
    comma if the sample is empty or sniffing fails.

    Args:
        filepath: Absolute or relative path to the CSV file.

    Returns:
        A single-character delimiter: comma, semicolon, tab, or comma if detection fails.
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(65536)
    if not sample.strip():
        logger.debug("empty_sample_for_sniff", filepath=str(path))
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delim = dialect.delimiter
        logger.debug(
            "delimiter_detected",
            filepath=str(path),
            delimiter_char=delim,
            delimiter_repr=repr(delim),
        )
        return delim
    except csv.Error as e:
        logger.debug("delimiter_sniff_failed", filepath=str(path), error=str(e))
        return ","


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase column names and replace runs of whitespace with single underscores.

    Args:
        df: DataFrame whose columns should be normalized for SQLite-friendly names.

    Returns:
        A copy of ``df`` with renamed columns.
    """
    out = df.copy()
    out.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in out.columns]
    return out


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize string content, parse date/time columns by name, empty strings to NA,
    and drop fully duplicate rows.

    Columns whose names contain ``date`` or ``time`` (case-insensitive) are parsed
    with :func:`pandas.to_datetime` (invalid values become NaT).

    Args:
        df: Input frame (typically after column names are cleaned).

    Returns:
        A new DataFrame with cleaned data.
    """
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )

    for col in out.columns:
        if "date" in col.lower() or "time" in col.lower():
            out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].replace(r"^\s*$", pd.NA, regex=True)
            out[col] = out[col].replace("", pd.NA)

    out = out.drop_duplicates()
    return out


def _infer_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns to int (nullable), float, datetime, or string where appropriate.

    Datetime columns are left as-is. Numeric dtypes may be narrowed to integer.
    Object columns are probed with :func:`pandas.to_numeric`; if all non-null values
    parse, the column becomes numeric (integer if all values are whole).
    Otherwise values are stored as pandas StringDtype.

    Args:
        df: DataFrame after :func:`clean_dataframe`.

    Returns:
        A copy with inferred dtypes applied.
    """
    out = df.copy()
    for col in out.columns:
        series = out[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                continue
            if pd.api.types.is_float_dtype(series) and (non_null % 1 == 0).all():
                out[col] = pd.to_numeric(series, errors="coerce").astype("Int64")
            continue

        valid = series.notna()
        if not valid.any():
            out[col] = series.astype("string")
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        parsed_count = (numeric.notna() & valid).sum()
        if parsed_count == valid.sum():
            nn = numeric.dropna()
            if len(nn) == 0:
                out[col] = numeric.astype("Float64")
            elif (nn % 1 == 0).all():
                out[col] = numeric.astype("Int64")
            else:
                out[col] = numeric.astype(float)
        else:
            out[col] = series.astype("string")

    return out


def _table_name_from_path(csv_path: Path) -> str:
    """
    Derive the SQLite table name from a CSV path (filename without ``.csv``).

    Args:
        csv_path: Path to the ``.csv`` file.

    Returns:
        The path's stem, used as the table name.
    """
    return csv_path.stem


def load_csv_folder(folder_path: str) -> Engine:
    """
    Load every ``*.csv`` in ``folder_path`` into a single in-memory SQLite database.

    For each file: detect delimiter, read with pandas, clean column names, run
    :func:`clean_dataframe`, infer dtypes, then ``to_sql`` into its own table.
    Tables are named after the file stem (without ``.csv``).

    Args:
        folder_path: Directory containing CSV files.

    Returns:
        A SQLAlchemy :class:`~sqlalchemy.engine.Engine` connected to ``sqlite:///:memory:``.

    Raises:
        FileNotFoundError: If ``folder_path`` is missing or not a directory.
        ValueError: If there are no ``.csv`` files, or every file failed to load.
    """
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(
            f"CSV folder does not exist or is not reachable: {root.resolve()}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"Path is not a directory: {root.resolve()}")

    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        raise ValueError(
            f"No .csv files found in folder: {root.resolve()}"
        )

    logger.info(
        "csv_folder_scan_complete",
        folder=str(root.resolve()),
        file_count=len(csv_files),
    )

    engine = create_engine("sqlite:///:memory:", future=True)

    for csv_path in csv_files:
        try:
            delim = detect_delimiter(str(csv_path))
            logger.info(
                "loading_csv",
                filepath=str(csv_path.resolve()),
                delimiter_char=delim,
                delimiter_repr=repr(delim),
            )
            df = pd.read_csv(
                csv_path,
                sep=delim,
                encoding="utf-8",
                encoding_errors="replace",
                low_memory=False,
            )
            df = _clean_column_names(df)
            df = clean_dataframe(df)
            df = _infer_column_dtypes(df)
            rows, cols = df.shape
            table_name = _table_name_from_path(csv_path)
            logger.info(
                "csv_loaded_shape",
                filepath=str(csv_path.resolve()),
                table=table_name,
                rows=rows,
                columns=cols,
            )
            df.to_sql(table_name, engine, index=False, if_exists="replace")
            logger.info(
                "csv_written_to_sqlite",
                table=table_name,
                rows=rows,
                columns=cols,
            )
        except Exception as e:
            logger.warning(
                "csv_load_skipped",
                filepath=str(csv_path.resolve()),
                error=str(e),
                exc_info=True,
            )

    final_tables = get_table_names(engine)
    if not final_tables:
        raise ValueError(
            f"All CSV files failed to load; no tables in database for folder: {root.resolve()}"
        )

    logger.info(
        "load_csv_folder_finished",
        folder=str(root.resolve()),
        tables=final_tables,
    )
    return engine


def get_table_names(engine: Engine) -> list[str]:
    """
    Return all table names visible to the given SQLAlchemy engine.

    Args:
        engine: An :class:`~sqlalchemy.engine.Engine` (e.g. from :func:`load_csv_folder`).

    Returns:
        Sorted list of table names in the connected database.
    """
    insp = inspect(engine)
    return sorted(insp.get_table_names())


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
    _olist = _base / "sample_data" / "olist"
    _engine = load_csv_folder(str(_olist))
    names = get_table_names(_engine)
    print("Tables:", names)
    with _engine.connect() as conn:
        for name in names:
            n = conn.exec_driver_sql(f'SELECT COUNT(*) AS c FROM "{name}"').scalar()
            # Column count from SQLite pragma
            r = conn.exec_driver_sql(f'PRAGMA table_info("{name}")').fetchall()
            col_count = len(r)
            print(f"  {name}: rows={n}, columns={col_count}")

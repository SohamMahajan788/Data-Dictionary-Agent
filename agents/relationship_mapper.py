"""
Build a directed relationship graph from :class:`DatabaseSchema`, including inferred links.

Combines declared foreign keys with naming heuristics (typical for CSV-loaded SQLite).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import sqlalchemy as sa
import structlog

from agents.schema_extractor import DatabaseSchema, TableSchema

logger = structlog.get_logger(__name__)


@dataclass
class Relationship:
    """A directed table link via a column pair."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str
    is_explicit: bool
    confidence: float


@dataclass
class RelationshipMap:
    """All detected relationships plus a NetworkX-friendly adjacency view."""

    relationships: list[Relationship]
    total_relationships: int
    tables: list[str]
    graph_data: dict


def _get_table(schema: DatabaseSchema, table_name: str) -> TableSchema | None:
    """
    Look up a :class:`TableSchema` by name.

    Args:
        schema: Full database schema.
        table_name: Table identifier.

    Returns:
        Matching :class:`TableSchema`, or ``None``.
    """
    for ts in schema.tables:
        if ts.table_name == table_name:
            return ts
    return None


def _is_junction_table(ts: TableSchema) -> bool:
    """
    Return True if ``ts`` looks like a junction/link table (composite PK of FK columns).

    Args:
        ts: Candidate table schema.

    Returns:
        Whether the junction heuristic matches.
    """
    if len(ts.foreign_keys) < 2:
        return False
    referred = {
        fk.get("referred_table")
        for fk in ts.foreign_keys
        if fk.get("referred_table")
    }
    if len(referred) < 2:
        return False
    fk_cols: set[str] = set()
    for fk in ts.foreign_keys:
        fk_cols.update(fk.get("constrained_columns") or [])
    pk_set = set(ts.primary_keys)
    return bool(pk_set) and fk_cols == pk_set


def _endpoints_have_junction(schema: DatabaseSchema, a: str, b: str) -> bool:
    """
    Return True if some third table is a junction linking both endpoint tables.

    Args:
        schema: Full schema.
        a: First entity table name.
        b: Second entity table name.

    Returns:
        Whether a junction pattern exists between ``a`` and ``b``.
    """
    for ts in schema.tables:
        if ts.table_name in (a, b):
            continue
        if not _is_junction_table(ts):
            continue
        referred = {
            fk.get("referred_table")
            for fk in ts.foreign_keys
            if fk.get("referred_table")
        }
        if a in referred and b in referred:
            return True
    return False


def detect_relationship_type(
    from_table: str,
    to_table: str,
    schema: DatabaseSchema,
) -> str:
    """
    Classify cardinality between ``from_table`` (referencing) and ``to_table`` (referenced).

    - ``many_to_many`` if a separate junction table links the two endpoints.
    - ``one_to_one`` if a referencing column is both PK and FK to ``to_table``.
    - ``one_to_many`` otherwise (default parent/child FK pattern).

    Args:
        from_table: Table that holds the foreign key (child side).
        to_table: Referenced parent table.
        schema: Full :class:`DatabaseSchema`.

    Returns:
        One of ``"one_to_many"``, ``"many_to_many"``, or ``"one_to_one"``.
    """
    ft = _get_table(schema, from_table)
    if ft and _is_junction_table(ft):
        return "one_to_many"
    if _endpoints_have_junction(schema, from_table, to_table):
        return "many_to_many"
    if ft is not None:
        pk_set = set(ft.primary_keys)
        for col in ft.columns:
            if not col.is_foreign_key:
                continue
            if col.name not in pk_set:
                continue
            ref = col.foreign_key_references or ""
            if ref.startswith(f"{to_table}."):
                if len(pk_set) == 1:
                    return "one_to_one"
    return "one_to_many"


def _entity_core(table_name: str) -> str:
    """
    Strip common Olist-style prefixes/suffixes for name matching.

    Args:
        table_name: Physical table name.

    Returns:
        Lowercased core token (e.g. ``olist_customers_dataset`` -> ``customers``).
    """
    n = table_name.lower()
    if n.startswith("olist_"):
        n = n[6:]
    if n.endswith("_dataset"):
        n = n[:-8]
    return n.strip("_")


def _stem_from_id_column(column_name: str) -> str | None:
    """
    Return the stem before ``_id`` when the column follows ``*_id`` naming.

    Args:
        column_name: Column name.

    Returns:
        Stem without trailing ``_id``, or ``None``.
    """
    c = column_name.lower().strip()
    if c.endswith("_id"):
        s = c[:-3].strip("_")
        return s or None
    return None


def _is_hub_identifier_column(child: TableSchema, col_name: str) -> bool:
    """
    Return True when ``col_name`` is the natural primary identifier for ``child``.

    Examples: ``orders.order_id``, ``customers.customer_id``. These columns usually
    do not point at sibling fact tables; skipping them reduces reversed implicit edges.

    Args:
        child: Table that owns the column.
        col_name: Column to test.

    Returns:
        Whether this looks like a hub / owner id, not a child FK.
    """
    stem = _stem_from_id_column(col_name)
    if not stem:
        return False
    if col_name.lower() != f"{stem}_id":
        return False
    core = _entity_core(child.table_name)
    plural = f"{stem}s"
    return core in (stem, plural, stem + "s")


def _column_name_map(ts: TableSchema) -> dict[str, str]:
    """
    Map lowercased column names to their canonical spelling on ``ts``.

    Args:
        ts: Table whose columns are indexed.

    Returns:
        Dict keyed by lowercased name.
    """
    return {c.name.lower(): c.name for c in ts.columns}


def _resolve_ref_column(from_col: str, to_ts: TableSchema) -> str | None:
    """
    Pick the referenced column on ``to_ts`` for an implicit FK.

    Prefers a column with the same name as ``from_col``, then a single PK,
    then ``{stem}_id``, then the first PK column, else the first column.

    Args:
        from_col: Referencing column on the child table.
        to_ts: Candidate parent table schema.

    Returns:
        Target column name, or ``None`` if ``to_ts`` has no columns.
    """
    cmap = _column_name_map(to_ts)
    fc = from_col.lower()
    if fc in cmap:
        return cmap[fc]
    pks = to_ts.primary_keys
    if len(pks) == 1:
        return pks[0]
    stem = _stem_from_id_column(from_col)
    if stem:
        sid = f"{stem}_id".lower()
        if sid in cmap:
            return cmap[sid]
    if pks:
        return pks[0]
    if to_ts.columns:
        return to_ts.columns[0].name
    return None


def _implicit_pair_confidence(from_col: str, to_ts: TableSchema) -> float:
    """
    Score how strongly ``from_col`` on some child table references ``to_ts``.

    Scoring (implicit only):
    - ``0.95``: entity stem aligns with table core and parent has same-named column.
    - ``0.80``: ``*_id`` column and stem aligns with parent core (singular/plural).
    - ``0.60``: partial stem/core overlap (length >= 3).

    Args:
        from_col: Child column name.
        to_ts: Candidate parent table.

    Returns:
        Confidence in ``[0, 1]``; ``0`` means no heuristic match.
    """
    stem = _stem_from_id_column(from_col)
    core = _entity_core(to_ts.table_name)
    cmap = _column_name_map(to_ts)
    fc = from_col.lower()

    entity_exact = False
    if stem:
        entity_exact = (
            core == stem
            or core == f"{stem}s"
            or core.rstrip("s") == stem
            or stem == core.rstrip("s")
        )

    if entity_exact and fc in cmap:
        return 0.95

    if from_col.lower().endswith("_id") and stem:
        if entity_exact:
            return 0.80
        parts = core.split("_")
        if stem in parts or core.startswith(stem) or core == f"{stem}s":
            return 0.80
        if len(stem) >= 3 and (stem in core or core in stem):
            return 0.60

    if stem and len(stem) >= 3 and (stem in core or core in stem):
        return 0.60

    return 0.0


def _pk_name_match_confidence(from_col: str, to_ts: TableSchema) -> float:
    """
    ``0.95`` when ``from_col`` matches a primary-key column name on ``to_ts``.

    Args:
        from_col: Child column name.
        to_ts: Candidate referenced table.

    Returns:
        ``0.95`` or ``0.0``.
    """
    pk_lower = {p.lower() for p in to_ts.primary_keys}
    if from_col.lower() in pk_lower:
        return 0.95
    return 0.0


def _explicit_relationship_key(
    from_table: str,
    from_column: str,
    to_table: str,
    to_column: str,
) -> tuple[str, str, str, str]:
    """Normalised tuple key for deduplication."""
    return (from_table, from_column, to_table, to_column)


def _collect_explicit_relationships(
    schema: DatabaseSchema,
    seen: set[tuple[str, str, str, str]],
) -> list[Relationship]:
    """
    Build :class:`Relationship` rows from ``TableSchema.foreign_keys``.

    Args:
        schema: Database schema.
        seen: Mutable set of keys already used (updated in place).

    Returns:
        List of explicit relationships.
    """
    out: list[Relationship] = []
    for ts in schema.tables:
        for fk in ts.foreign_keys:
            parent = fk.get("referred_table")
            if not parent:
                continue
            for local_c, remote_c in zip(
                fk.get("constrained_columns") or [],
                fk.get("referred_columns") or [],
            ):
                key = _explicit_relationship_key(
                    ts.table_name, str(local_c), parent, str(remote_c)
                )
                if key in seen:
                    continue
                seen.add(key)
                rtype = detect_relationship_type(ts.table_name, parent, schema)
                out.append(
                    Relationship(
                        from_table=ts.table_name,
                        from_column=str(local_c),
                        to_table=parent,
                        to_column=str(remote_c),
                        relationship_type=rtype,
                        is_explicit=True,
                        confidence=1.0,
                    )
                )
    logger.info(
        "relationship_explicit_collected",
        count=len(out),
        source_path=schema.source_path,
    )
    return out


def _best_implicit_for_column(
    schema: DatabaseSchema,
    child: TableSchema,
    col_name: str,
    seen: set[tuple[str, str, str, str]],
) -> Relationship | None:
    """
    At most one implicit link per (child, column), choosing the highest confidence.

    Args:
        schema: Full schema.
        child: Table containing ``col_name``.
        col_name: Column to infer from.
        seen: Keys for edges already taken (explicit or prior implicit).

    Returns:
        A :class:`Relationship` or ``None``.
    """
    col_obj = next((c for c in child.columns if c.name == col_name), None)
    if col_obj is None:
        return None
    if col_obj.is_foreign_key and col_obj.foreign_key_references:
        return None

    pks = child.primary_keys
    if len(pks) == 1 and pks[0] == col_name:
        return None

    if _is_hub_identifier_column(child, col_name):
        return None

    candidates: list[tuple[float, str, str]] = []
    stem = _stem_from_id_column(col_name)

    for other in schema.tables:
        if other.table_name == child.table_name:
            continue
        pk_score = _pk_name_match_confidence(col_name, other)
        name_score = _implicit_pair_confidence(col_name, other)
        score = max(pk_score, name_score)
        if score < 0.60:
            continue
        to_col = _resolve_ref_column(col_name, other)
        if to_col is None:
            continue
        cmap_other = _column_name_map(other)
        cn = col_name.lower()
        if cn.endswith("_id"):
            if cn not in cmap_other and (not stem or f"{stem}_id".lower() not in cmap_other):
                continue
        key = _explicit_relationship_key(
            child.table_name, col_name, other.table_name, to_col
        )
        if key in seen:
            continue
        candidates.append((score, other.table_name, to_col))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    best_score, parent_table, to_col = candidates[0]
    key = _explicit_relationship_key(
        child.table_name, col_name, parent_table, to_col
    )
    seen.add(key)
    rtype = detect_relationship_type(child.table_name, parent_table, schema)
    return Relationship(
        from_table=child.table_name,
        from_column=col_name,
        to_table=parent_table,
        to_column=to_col,
        relationship_type=rtype,
        is_explicit=False,
        confidence=best_score,
    )


def _collect_implicit_relationships(
    schema: DatabaseSchema,
    seen: set[tuple[str, str, str, str]],
) -> list[Relationship]:
    """
    Infer relationships using naming rules and PK name overlap.

    Args:
        schema: Database schema.
        seen: Keys already used (updated in place).

    Returns:
        Implicit :class:`Relationship` instances with confidence ``>= 0.60``.
    """
    out: list[Relationship] = []
    for ts in schema.tables:
        for col in ts.columns:
            rel = _best_implicit_for_column(schema, ts, col.name, seen)
            if rel is not None:
                out.append(rel)
    logger.info(
        "relationship_implicit_collected",
        count=len(out),
        source_path=schema.source_path,
    )
    return out


def _build_digraph(relationships: list[Relationship]) -> nx.DiGraph:
    """
    Construct a directed graph with serialisable edge payloads.

    Args:
        relationships: All edges to add.

    Returns:
        A :class:`networkx.DiGraph` with an ``edges`` list on each arc.
    """
    g: nx.DiGraph = nx.DiGraph()
    for r in relationships:
        g.add_node(r.from_table)
        g.add_node(r.to_table)
        if not g.has_edge(r.from_table, r.to_table):
            g.add_edge(r.from_table, r.to_table, edges=[])
        payload = {
            "from_column": r.from_column,
            "to_column": r.to_column,
            "relationship_type": r.relationship_type,
            "is_explicit": r.is_explicit,
            "confidence": r.confidence,
        }
        g[r.from_table][r.to_table]["edges"].append(payload)
    return g


def build_relationship_map(schema: DatabaseSchema) -> RelationshipMap:
    """
    Combine explicit foreign keys and inferred column/table-name links into a map.

    Logs SQLAlchemy version for stack context, then explicit pass, implicit pass,
    graph build, and summary counts.

    Args:
        schema: Output of :func:`agents.schema_extractor.extract_schema`.

    Returns:
        :class:`RelationshipMap` with relationships, totals, table list, and adjacency dict.
    """
    logger.info(
        "relationship_map_start",
        source_path=schema.source_path,
        total_tables=schema.total_tables,
        sqlalchemy_version=sa.__version__,
    )
    seen: set[tuple[str, str, str, str]] = set()
    explicit = _collect_explicit_relationships(schema, seen)
    implicit = _collect_implicit_relationships(schema, seen)
    relationships = explicit + implicit

    g = _build_digraph(relationships)
    graph_data: dict = nx.to_dict_of_dicts(g)

    tables_sorted = sorted({t.table_name for t in schema.tables})
    result = RelationshipMap(
        relationships=relationships,
        total_relationships=len(relationships),
        tables=tables_sorted,
        graph_data=graph_data,
    )
    logger.info(
        "relationship_map_finished",
        source_path=schema.source_path,
        total_relationships=result.total_relationships,
        explicit=len(explicit),
        implicit=len(implicit),
        graph_nodes=g.number_of_nodes(),
        graph_edges=g.number_of_edges(),
    )
    return result


def relationship_map_to_dict(rel_map: RelationshipMap) -> dict:
    """
    Convert a :class:`RelationshipMap` to nested dicts/lists for JSON encoding.

    Args:
        rel_map: Relationship map from :func:`build_relationship_map`.

    Returns:
        JSON-friendly structure (includes :func:`networkx` adjacency dict).
    """
    return asdict(rel_map)


def _print_relationship_report(label: str, rel_map: RelationshipMap) -> None:
    """
    Print every relationship with explicit/implicit label and confidence.

    Args:
        label: Dataset label.
        rel_map: Map to print.
    """
    print(f"\n=== {label} ===")
    edge_arcs = sum(len(v) for v in rel_map.graph_data.values()) if rel_map.graph_data else 0
    print(
        f"total_relationships={rel_map.total_relationships} "
        f"tables={len(rel_map.tables)} graph_nodes={len(rel_map.graph_data)} "
        f"graph_arcs={edge_arcs}"
    )
    for r in rel_map.relationships:
        kind = "explicit" if r.is_explicit else "implicit"
        conf = f" confidence={r.confidence:.2f}" if not r.is_explicit else ""
        print(
            f"  [{kind}] {r.from_table}.{r.from_column} -> "
            f"{r.to_table}.{r.to_column} "
            f"type={r.relationship_type}{conf}"
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

    _engine_o = load_csv_folder(str(_olist))
    _schema_o = extract_schema(_engine_o, str(_olist.resolve()))
    _map_o = build_relationship_map(_schema_o)
    _print_relationship_report("olist (CSV / implicit + any explicit)", _map_o)

    _engine_c = load_sqlite(str(_chinook))
    _schema_c = extract_schema(_engine_c, str(_chinook.resolve()))
    _map_c = build_relationship_map(_schema_c)
    _print_relationship_report("chinook (SQLite / explicit FKs)", _map_c)

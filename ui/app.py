"""
Streamlit UI for running the Data Dictionary pipeline and downloading reports.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import streamlit as st
import structlog

from orchestrator.pipeline import DictionaryResult, InputType, PipelineConfig, run_pipeline
from output.report_generator import ReportConfig, generate_report

logger = structlog.get_logger(__name__)

st.set_page_config(
    page_title="Data Dictionary Agent",
    page_icon="📘",
    layout="wide",
)


def _init_session_state() -> None:
    """Initialize app-level session keys."""
    defaults = {
        "pipeline_result": None,
        "report_path": None,
        "is_running": False,
        "error_message": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_state() -> None:
    """Clear run state for a new execution."""
    st.session_state.pipeline_result = None
    st.session_state.report_path = None
    st.session_state.is_running = False
    st.session_state.error_message = None


def _render_welcome() -> None:
    """Render the initial landing content (state 1)."""
    st.title("Data Dictionary Agent")
    st.write(
        "Upload a SQLite database or provide a CSV folder path to generate a "
        "full data dictionary with schema, quality, relationships, PII flags, and AI summaries."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "### Schema Analysis\n"
            "- Tables, columns, data types\n"
            "- Primary/foreign keys\n"
            "- Relationship structure"
        )
    with c2:
        st.markdown(
            "### Quality Profiling\n"
            "- Completeness per table\n"
            "- Null and uniqueness metrics\n"
            "- Duplicate row detection"
        )
    with c3:
        st.markdown(
            "### AI Summaries\n"
            "- Business descriptions\n"
            "- Data quality notes\n"
            "- Domain-level overview"
        )

    st.markdown("---")
    st.subheader("Sample Datasets")
    st.write(
        "- **Chinook** (`sample_data/chinook`) for a compact SQLite example.\n"
        "- **Olist** (`sample_data/olist`) for multi-table CSV ingestion."
    )


def _render_overview_tab(result: DictionaryResult) -> None:
    """Render Overview tab content."""
    summary = result.summary
    if summary is None:
        st.info("LLM summary is not available for this run.")
    else:
        st.subheader("Database Description")
        st.write(summary.database_description)
        st.caption(f"Business domain: {summary.business_domain}")

    st.subheader("Overall Stats")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tables", result.schema.total_tables)
    m2.metric("Columns", result.schema.total_columns)
    m3.metric("Completeness", f"{result.profile.overall_completeness * 100:.2f}%")
    m4.metric("Relationships", result.relationships.total_relationships)
    pii_total = 0 if result.pii_report is None else result.pii_report.total_pii_columns
    m5.metric("PII Columns", pii_total)


def _render_tables_tab(result: DictionaryResult) -> None:
    """Render Tables tab with one expander per table."""
    summary_map = {}
    if result.summary is not None:
        summary_map = {t.table_name: t for t in result.summary.tables}
    pii_by_table: dict[str, int] = {}
    if result.pii_report is not None:
        for f in result.pii_report.findings:
            key = f"{f.table_name}.{f.column_name}"
            pii_by_table.setdefault(f.table_name, 0)
            # Count unique PII columns per table
            pii_by_table[f.table_name] = len(
                {
                    x.column_name
                    for x in result.pii_report.findings
                    if x.table_name == f.table_name
                }
            )

    for tp in result.profile.tables:
        ts = next((x for x in result.schema.tables if x.table_name == tp.table_name), None)
        if ts is None:
            continue
        t_sum = summary_map.get(tp.table_name)
        pii_count = pii_by_table.get(tp.table_name, 0)

        with st.expander(f"{tp.table_name}"):
            st.write(
                f"**Columns:** {tp.total_columns} | "
                f"**Rows (sampled):** {tp.total_rows} | "
                f"**Completeness:** {tp.completeness_score * 100:.2f}% | "
                f"**PII columns:** {pii_count}"
            )
            if t_sum is not None:
                st.write(f"**Business description:** {t_sum.business_description}")
                st.write(f"**Business purpose:** {t_sum.business_purpose}")
            else:
                st.caption("No LLM summary available for this table.")

            col_data = []
            for c in ts.columns:
                cp = next((x for x in tp.columns if x.column_name == c.name), None)
                col_data.append(
                    {
                        "column": c.name,
                        "type": c.data_type,
                        "nullable": c.is_nullable,
                        "pk": c.is_primary_key,
                        "fk": c.is_foreign_key,
                        "null_rate": None if cp is None else round(cp.null_rate, 4),
                    }
                )
            st.dataframe(col_data, use_container_width=True)


def _render_quality_tab(result: DictionaryResult) -> None:
    """Render quality-focused diagnostics."""
    issues = [t for t in result.profile.tables if t.has_data_quality_issues]
    if not issues:
        st.success("No tables flagged with quality issues.")
        return

    st.warning(f"{len(issues)} table(s) flagged with quality issues.")
    for t in issues:
        top_nulls = sorted(t.columns, key=lambda c: c.null_rate, reverse=True)[:3]
        top_text = ", ".join(f"{c.column_name} ({c.null_rate * 100:.1f}%)" for c in top_nulls)
        st.write(
            f"- **{t.table_name}** | completeness: {t.completeness_score * 100:.2f}% | "
            f"duplicates: {t.duplicate_row_count} | top nulls: {top_text}"
        )


def _render_pii_tab(result: DictionaryResult) -> None:
    """Render PII findings grouped by table."""
    if result.pii_report is None:
        st.info("PII detection was disabled for this run.")
        return
    if not result.pii_report.findings:
        st.success("No PII findings detected.")
        return

    grouped: dict[str, list] = {}
    for f in result.pii_report.findings:
        grouped.setdefault(f.table_name, []).append(f)

    for table in sorted(grouped.keys()):
        with st.expander(table):
            rows = [
                {
                    "column": f.column_name,
                    "pii_type": f.pii_type,
                    "method": f.detection_method,
                    "confidence": round(f.confidence, 3),
                    "sample_trigger": f.sample_trigger,
                }
                for f in grouped[table]
            ]
            st.dataframe(rows, use_container_width=True)


def _render_results(result: DictionaryResult, report_path: str | None) -> None:
    """Render completed state (state 3)."""
    st.success(f"Pipeline completed in {result.pipeline_duration_seconds:.2f}s")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tables", result.schema.total_tables)
    m2.metric("Columns", result.schema.total_columns)
    m3.metric("Completeness", f"{result.profile.overall_completeness * 100:.2f}%")
    m4.metric("Relationships", result.relationships.total_relationships)

    # Warning if LLM was enabled but output looks fallback-heavy.
    if result.config.enable_llm:
        if result.summary is None or result.summary.database_description == "Auto-generated description unavailable":
            st.warning("LLM summaries may have failed. Core schema/profile output is still available.")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Tables", "Quality", "PII Report"])
    with tab1:
        _render_overview_tab(result)
    with tab2:
        _render_tables_tab(result)
    with tab3:
        _render_quality_tab(result)
    with tab4:
        _render_pii_tab(result)

    st.markdown("---")
    if report_path:
        report_file = Path(report_path)
        if report_file.exists():
            st.download_button(
                "Download HTML Report",
                data=report_file.read_bytes(),
                file_name=report_file.name,
                mime="text/html",
                use_container_width=True,
            )

    if st.button("Run Again", use_container_width=True):
        _reset_state()
        st.rerun()


def _run_pipeline_from_ui(
    input_mode: str,
    uploaded_sqlite,
    csv_folder_path: str,
    sample_size: int,
    enable_llm: bool,
    enable_pii: bool,
    report_title: str,
) -> None:
    """Execute pipeline + report generation with status updates."""
    st.session_state.is_running = True
    st.session_state.error_message = None

    tmp_dir: Path | None = None
    sqlite_tmp_path: Path | None = None

    status = st.status("Running pipeline...", expanded=True)
    try:
        status.write("Validating input...")
        if input_mode == "SQLite / .db file":
            if uploaded_sqlite is None:
                raise ValueError("Please upload a .sqlite or .db file.")
            tmp_dir = Path(tempfile.mkdtemp(prefix="dda_sqlite_"))
            sqlite_tmp_path = tmp_dir / uploaded_sqlite.name
            sqlite_tmp_path.write_bytes(uploaded_sqlite.getbuffer())
            input_path = str(sqlite_tmp_path.resolve())
            input_type = InputType.SQLITE_FILE
        else:
            if not csv_folder_path.strip():
                raise ValueError("Please enter a CSV folder path.")
            input_path = csv_folder_path.strip()
            input_type = InputType.CSV_FOLDER

        config = PipelineConfig(
            input_path=input_path,
            input_type=input_type,
            sample_size=sample_size,
            enable_llm=enable_llm,
            enable_pii=enable_pii,
            output_dir="./output/generated",
        )

        status.write("Running full dictionary pipeline...")
        logger.info("ui_pipeline_run_start", input_type=input_type.value, input_path=input_path)
        result = run_pipeline(config)
        st.session_state.pipeline_result = result

        status.write("Generating HTML report...")
        report_cfg = ReportConfig(
            output_dir="./output/generated",
            report_title=report_title,
            include_pii_section=enable_pii,
            include_quality_section=True,
            include_relationships_section=True,
            include_llm_summaries=enable_llm,
        )
        report_out = generate_report(result, report_cfg)
        st.session_state.report_path = report_out["html_path"]

        status.update(label="Pipeline complete", state="complete")
    except Exception as e:
        logger.error("ui_pipeline_run_failed", error=str(e), exc_info=True)
        st.session_state.error_message = str(e)
        status.update(label="Pipeline failed", state="error")
    finally:
        st.session_state.is_running = False
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        # Explicit cleanup confirmation for uploaded SQLite temp file.
        if sqlite_tmp_path is not None:
            logger.info("ui_temp_sqlite_cleanup_done", temp_path=str(sqlite_tmp_path))


def main() -> None:
    """Render and run the Streamlit application."""
    _init_session_state()

    with st.sidebar:
        st.title("Data Dictionary Agent")
        st.write(
            "Generate a complete data dictionary from SQLite databases or CSV folders."
        )

        input_mode = st.radio(
            "Input Type",
            options=["SQLite / .db file", "CSV folder path"],
        )

        uploaded_sqlite = None
        csv_folder_path = ""
        if input_mode == "SQLite / .db file":
            uploaded_sqlite = st.file_uploader(
                "Upload database file",
                type=["sqlite", "db"],
            )
        else:
            csv_folder_path = st.text_input("CSV folder path", value="")

        with st.expander("Advanced Options", expanded=False):
            sample_size = st.slider("Sample size", 1000, 50000, 10000, step=1000)
            enable_llm = st.toggle("Enable LLM summaries", value=True)
            enable_pii = st.toggle("Enable PII detection", value=True)
            report_title = st.text_input("Report title", value="Data Dictionary Report")

        st.markdown("---")
        run_clicked = st.button("Generate Dictionary", use_container_width=True, type="primary")

    # Main content states
    if st.session_state.error_message:
        st.error(f"Pipeline failed: {st.session_state.error_message}")

    if run_clicked:
        _run_pipeline_from_ui(
            input_mode=input_mode,
            uploaded_sqlite=uploaded_sqlite,
            csv_folder_path=csv_folder_path,
            sample_size=sample_size,
            enable_llm=enable_llm,
            enable_pii=enable_pii,
            report_title=report_title,
        )

    if st.session_state.is_running:
        with st.spinner("Pipeline is running..."):
            st.info("Step-by-step updates appear in the status panel while the run executes.")
        return

    result = st.session_state.pipeline_result
    report_path = st.session_state.report_path
    if result is None:
        _render_welcome()
    else:
        _render_results(result, report_path)


if __name__ == "__main__":
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    main()

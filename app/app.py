"""Mapping Review App -- review agent-proposed taxonomy mappings.

Deployed as a Databricks App (Streamlit). Reads proposed mappings from
Delta, lets reviewers approve/reject/edit, and links to MLflow traces.
"""

import os
from datetime import datetime

import streamlit as st
from databricks import sql as dbsql
from databricks.sdk import WorkspaceClient

CATALOG = os.environ.get("CATALOG", "main")
SCHEMA = os.environ.get("SCHEMA", "graph_hierarchy_demo")
WORKSPACE_URL = os.environ.get(
    "DATABRICKS_HOST",
    os.environ.get("WORKSPACE_URL", ""),
)


@st.cache_resource
def get_connection():
    w = WorkspaceClient()
    warehouses = w.warehouses.list()
    warehouse = next(
        (wh for wh in warehouses if wh.state.value == "RUNNING"),
        next(iter(warehouses), None),
    )
    if not warehouse:
        st.error("No SQL warehouse found.")
        st.stop()

    return dbsql.connect(
        server_hostname=w.config.host,
        http_path=f"/sql/1.0/warehouses/{warehouse.id}",
        credentials_provider=lambda: w.config._header_factory(),
    )


def run_query(query: str, params: dict | None = None) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, params or {})
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    return [dict(zip(columns, row)) for row in rows]


def update_mapping(v1_node_id: str, status: str, note: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        UPDATE {CATALOG}.{SCHEMA}.proposed_mappings
        SET status = %(status)s,
            reviewer_note = %(note)s,
            reviewed_at = current_timestamp()
        WHERE v1_node_id = %(v1_id)s AND status = 'pending'
        """,
        {"status": status, "note": note, "v1_id": v1_node_id},
    )
    cursor.close()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Mapping Review", layout="wide")
st.title("Taxonomy Mapping Review")

# Stats bar
stats = run_query(f"""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
        SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved,
        SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
        ROUND(AVG(confidence), 2) as avg_confidence
    FROM {CATALOG}.{SCHEMA}.proposed_mappings
""")

if stats:
    s = stats[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", s["total"] or 0)
    c2.metric("Pending", s["pending"] or 0)
    c3.metric("Approved", s["approved"] or 0)
    c4.metric("Rejected", s["rejected"] or 0)
    c5.metric("Avg Confidence", s["avg_confidence"] or 0)

st.divider()

# Filters
col_filter, col_sort = st.columns(2)
with col_filter:
    status_filter = st.selectbox(
        "Status", ["pending", "approved", "rejected", "all"], index=0,
    )
with col_sort:
    sort_by = st.selectbox(
        "Sort by", ["confidence DESC", "confidence ASC", "v1_path ASC"],
    )

where = "" if status_filter == "all" else f"WHERE status = '{status_filter}'"
proposals = run_query(f"""
    SELECT v1_node_id, v2_node_id, v1_name, v2_name,
           v1_path, v2_path, confidence, reasoning,
           status, reviewer_note, trace_id, created_at
    FROM {CATALOG}.{SCHEMA}.proposed_mappings
    {where}
    ORDER BY {sort_by}
    LIMIT 100
""")

if not proposals:
    st.info("No proposals found. Run notebook 02 to generate mappings.")
    st.stop()

# Main review area
for i, p in enumerate(proposals):
    with st.expander(
        f"{'**' if p['status'] == 'pending' else ''}"
        f"{p['v1_name']} -> {p['v2_name'] or '???'} "
        f"(confidence: {p['confidence']:.0%})"
        f"{'**' if p['status'] == 'pending' else ''}",
        expanded=(i == 0 and status_filter == "pending"),
    ):
        left, right = st.columns(2)

        with left:
            st.markdown("**V1 (old taxonomy)**")
            st.code(p["v1_path"], language=None)

        with right:
            st.markdown("**V2 (new taxonomy)**")
            st.code(p["v2_path"] or "No mapping proposed", language=None)

        st.markdown(f"**Agent reasoning:** {p['reasoning']}")

        # MLflow trace link
        if p["trace_id"] and WORKSPACE_URL:
            trace_url = (
                f"https://{WORKSPACE_URL}/ml/experiments/"
                f"?searchFilter=trace.request_id%3D%27{p['trace_id']}%27"
            )
            st.markdown(f"[View MLflow Trace]({trace_url})")
        elif p["trace_id"]:
            st.caption(f"Trace ID: {p['trace_id']}")

        # Actions
        if p["status"] == "pending":
            action_cols = st.columns(4)
            key_prefix = f"action_{p['v1_node_id']}_{i}"

            with action_cols[0]:
                if st.button("Approve", key=f"{key_prefix}_approve", type="primary"):
                    update_mapping(p["v1_node_id"], "approved")
                    st.rerun()

            with action_cols[1]:
                if st.button("Reject", key=f"{key_prefix}_reject"):
                    update_mapping(p["v1_node_id"], "rejected")
                    st.rerun()

            with action_cols[2]:
                note = st.text_input(
                    "Note", key=f"{key_prefix}_note", placeholder="Optional note...",
                )

            with action_cols[3]:
                if st.button("Reject with note", key=f"{key_prefix}_reject_note"):
                    update_mapping(p["v1_node_id"], "rejected", note)
                    st.rerun()
        else:
            st.caption(
                f"Status: **{p['status']}**"
                + (f" -- {p['reviewer_note']}" if p["reviewer_note"] else "")
            )

# Ground truth export
st.divider()
st.subheader("Ground Truth Export")
st.markdown(
    "Approved mappings become ground truth for evaluating future agent runs."
)

approved_count = run_query(f"""
    SELECT COUNT(*) as cnt FROM {CATALOG}.{SCHEMA}.proposed_mappings
    WHERE status = 'approved'
""")

if approved_count and approved_count[0]["cnt"] > 0:
    st.success(f"{approved_count[0]['cnt']} approved mappings available as ground truth.")

    if st.button("Export ground truth to Delta table"):
        run_query(f"""
            CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.ground_truth AS
            SELECT v1_node_id, v2_node_id, v1_name, v2_name,
                   v1_path, v2_path, reviewer_note,
                   reviewed_at
            FROM {CATALOG}.{SCHEMA}.proposed_mappings
            WHERE status = 'approved'
        """)
        st.success(
            f"Exported to `{CATALOG}.{SCHEMA}.ground_truth`"
        )
else:
    st.info("No approved mappings yet. Review proposals above.")

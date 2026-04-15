"""Mapping Review App -- review agent-proposed taxonomy mappings.

Deployed as a Databricks App (Streamlit). Reads proposed mappings from
Delta, lets reviewers approve/reject/edit, and links to MLflow traces.
Includes graph visualization of hierarchy context for each mapping.
"""

import os
import textwrap

import streamlit as st
from databricks import sql as dbsql
from databricks.sdk import WorkspaceClient

CATALOG = os.environ.get("CATALOG", "shm")
SCHEMA = os.environ.get("SCHEMA", "graph")
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "b5d9b3c3fc993bf3")
MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "/Shared/taxonomy-mapping-agent",
)


@st.cache_resource
def get_workspace_client():
    return WorkspaceClient()


@st.cache_resource
def get_connection():
    w = get_workspace_client()
    hostname = w.config.host
    if hostname.startswith("https://"):
        hostname = hostname[len("https://"):]
    if hostname.startswith("http://"):
        hostname = hostname[len("http://"):]
    hostname = hostname.rstrip("/")

    headers = w.config._header_factory()
    token = headers.get("Authorization", "").removeprefix("Bearer ")

    return dbsql.connect(
        server_hostname=hostname,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        access_token=token,
    )


@st.cache_data(ttl=3600)
def get_experiment_id() -> str | None:
    """Resolve the MLflow experiment ID from the experiment name."""
    try:
        w = get_workspace_client()
        resp = w.workspace.get_status(MLFLOW_EXPERIMENT_NAME)
        return str(resp.object_id) if resp else None
    except Exception:
        return None


def get_workspace_host() -> str:
    w = get_workspace_client()
    return w.config.host.rstrip("/")


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
# Graph visualization helpers
# ---------------------------------------------------------------------------

def _wrap_label(name: str, width: int = 22) -> str:
    """Wrap a category name for clean graph rendering."""
    escaped = name.replace('"', '\\"').replace("&", "&amp;")
    lines = textwrap.wrap(escaped, width=width)
    return "\\n".join(lines) if lines else escaped


def build_mapping_graph(
    v1_path: str,
    v2_path: str,
    v1_name: str,
    v2_name: str,
    confidence: float,
) -> str:
    """Build a Graphviz DOT string showing the hierarchy mapping."""
    v1_chain = [p.strip() for p in v1_path.split(" > ")] if v1_path else []
    v2_chain = [p.strip() for p in v2_path.split(" > ")] if v2_path else []

    lines = [
        'digraph G {',
        '    rankdir=LR;',
        '    graph [fontname="Helvetica" bgcolor="transparent" pad="0.4"'
        ' nodesep="0.3" ranksep="0.5"];',
        '    node [fontname="Helvetica" fontsize="11" shape=box'
        ' style="rounded,filled" height="0.4" margin="0.12,0.06"];',
        '    edge [fontname="Helvetica" fontsize="9"];',
        '',
    ]

    # V1 subgraph (blue theme)
    lines.append('    subgraph cluster_v1 {')
    lines.append('        label="V1 Taxonomy (2015)";')
    lines.append('        style=dashed; color="#1565C0"; fontcolor="#1565C0";'
                 ' fontsize="12"; fontname="Helvetica Bold";')

    for idx, name in enumerate(v1_chain):
        is_leaf = idx == len(v1_chain) - 1
        label = _wrap_label(name)
        if is_leaf:
            lines.append(
                f'        v1_{idx} [label="{label}"'
                f' fillcolor="#1565C0" fontcolor="white" penwidth="2"];'
            )
        else:
            lines.append(
                f'        v1_{idx} [label="{label}"'
                f' fillcolor="#E3F2FD" color="#1565C0"];'
            )

    for idx in range(len(v1_chain) - 1):
        lines.append(f'        v1_{idx} -> v1_{idx + 1}'
                     f' [color="#90CAF9" arrowsize="0.7"];')

    lines.append('    }')
    lines.append('')

    # V2 subgraph (green theme)
    lines.append('    subgraph cluster_v2 {')
    lines.append('        label="V2 Taxonomy (2021)";')
    lines.append('        style=dashed; color="#2E7D32"; fontcolor="#2E7D32";'
                 ' fontsize="12"; fontname="Helvetica Bold";')

    for idx, name in enumerate(v2_chain):
        is_leaf = idx == len(v2_chain) - 1
        label = _wrap_label(name)
        if is_leaf:
            lines.append(
                f'        v2_{idx} [label="{label}"'
                f' fillcolor="#2E7D32" fontcolor="white" penwidth="2"];'
            )
        else:
            lines.append(
                f'        v2_{idx} [label="{label}"'
                f' fillcolor="#E8F5E9" color="#2E7D32"];'
            )

    for idx in range(len(v2_chain) - 1):
        lines.append(f'        v2_{idx} -> v2_{idx + 1}'
                     f' [color="#A5D6A7" arrowsize="0.7"];')

    lines.append('    }')
    lines.append('')

    # Mapping arrow between leaf nodes
    if v1_chain and v2_chain:
        conf_pct = f"{confidence:.0%}" if confidence else "?"
        lines.append(
            f'    v1_{len(v1_chain) - 1} -> v2_{len(v2_chain) - 1}'
            f' [style=dashed color="#E65100" penwidth="2.5"'
            f' label="  {conf_pct}" fontcolor="#E65100"'
            f' fontsize="12" fontname="Helvetica Bold"];'
        )

    lines.append('}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Taxonomy Mapping Review", layout="wide")
st.title("Taxonomy Mapping Review")
st.caption("Agent-proposed category mappings with hierarchy context and MLflow tracing")

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

# Resolve experiment ID for trace links
experiment_id = get_experiment_id()
host = get_workspace_host()

# Main review area
for i, p in enumerate(proposals):
    with st.expander(
        f"{'**' if p['status'] == 'pending' else ''}"
        f"{p['v1_name']} -> {p['v2_name'] or '???'} "
        f"(confidence: {p['confidence']:.0%})"
        f"{'**' if p['status'] == 'pending' else ''}",
        expanded=(i == 0 and status_filter == "pending"),
    ):
        # Graph visualization
        if p["v1_path"] and p["v2_path"]:
            dot = build_mapping_graph(
                p["v1_path"], p["v2_path"],
                p["v1_name"], p["v2_name"],
                p["confidence"],
            )
            st.graphviz_chart(dot, use_container_width=True)
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("**V1 (old taxonomy)**")
                st.code(p["v1_path"], language=None)
            with right:
                st.markdown("**V2 (new taxonomy)**")
                st.code(p["v2_path"] or "No mapping proposed", language=None)

        st.markdown(f"**Agent reasoning:** {p['reasoning']}")

        # MLflow trace link
        if p["trace_id"] and experiment_id and host:
            trace_url = (
                f"{host}/ml/experiments/{experiment_id}"
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

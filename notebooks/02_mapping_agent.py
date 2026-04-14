# Databricks notebook source

# MAGIC %md
# MAGIC # 02 -- Mapping Agent: Hierarchy Navigation + Semantic Matching
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Defines agent tools that use **recursive CTEs** for hierarchy traversal
# MAGIC 2. Uses **Vector Search** (optional) for semantic matching between taxonomies
# MAGIC 3. Runs an LLM agent that proposes v1 -> v2 category mappings
# MAGIC 4. Logs all proposals with **MLflow tracing** for audit and review
# MAGIC
# MAGIC **Prerequisite:** Run `01_data_setup` first.

# COMMAND ----------

# Configuration -- must match notebook 01
CATALOG = "main"
SCHEMA = "graph_hierarchy_demo"
VS_ENDPOINT = None  # Set to match notebook 01
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
BATCH_SIZE = 20  # Number of unmapped categories to process per run

# COMMAND ----------

import json

import mlflow
from pyspark.sql import functions as F

mlflow.tracing.enable()


def _sq(val: str) -> str:
    """Escape single quotes for safe SQL interpolation."""
    return str(val).replace("'", "''")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Agent Tools -- Recursive CTE Traversal

# COMMAND ----------

def get_node_context(node_id: str) -> str:
    """Get full context for a node: ancestors, siblings, and children.

    Uses recursive CTEs to walk the hierarchy in both directions.
    """
    # Get the node itself
    node = spark.sql(f"""
        SELECT node_id, name, full_path, level, taxonomy_version
        FROM {CATALOG}.{SCHEMA}.nodes
        WHERE node_id = '{_sq(node_id)}'
    """).collect()

    if not node:
        return f"Node '{node_id}' not found."

    n = node[0]

    # Get ancestors (walk up via PARENT_OF)
    ancestors = spark.sql(f"""
        WITH RECURSIVE ancestors AS (
            SELECT e.source_id as node_id, 1 as depth
            FROM {CATALOG}.{SCHEMA}.edges e
            WHERE e.target_id = '{node_id}'
              AND e.relationship_type = 'PARENT_OF'

            UNION ALL

            SELECT e2.source_id, a.depth + 1
            FROM ancestors a
            JOIN {CATALOG}.{SCHEMA}.edges e2
              ON e2.target_id = a.node_id
              AND e2.relationship_type = 'PARENT_OF'
            WHERE a.depth < 10
        )
        SELECT n.name, n.level, a.depth
        FROM ancestors a
        JOIN {CATALOG}.{SCHEMA}.nodes n ON n.node_id = a.node_id
        ORDER BY a.depth DESC
    """).collect()

    # Get children (direct)
    children = spark.sql(f"""
        SELECT n.name, n.level
        FROM {CATALOG}.{SCHEMA}.edges e
        JOIN {CATALOG}.{SCHEMA}.nodes n ON n.node_id = e.target_id
        WHERE e.source_id = '{node_id}'
          AND e.relationship_type = 'PARENT_OF'
        ORDER BY n.name
    """).collect()

    # Get siblings (same parent)
    siblings = spark.sql(f"""
        SELECT n2.name, n2.node_id
        FROM {CATALOG}.{SCHEMA}.edges e1
        JOIN {CATALOG}.{SCHEMA}.edges e2
          ON e1.source_id = e2.source_id
          AND e2.relationship_type = 'PARENT_OF'
        JOIN {CATALOG}.{SCHEMA}.nodes n2 ON n2.node_id = e2.target_id
        WHERE e1.target_id = '{node_id}'
          AND e1.relationship_type = 'PARENT_OF'
          AND n2.node_id != '{node_id}'
        ORDER BY n2.name
        LIMIT 10
    """).collect()

    lines = [
        f"Node: {n.name} (ID: {n.node_id}, Version: {n.taxonomy_version})",
        f"Full path: {n.full_path}",
        f"Level: {n.level}",
    ]

    if ancestors:
        lines.append(f"Ancestors: {' > '.join(a.name for a in ancestors)}")
    if children:
        lines.append(f"Children ({len(children)}): {', '.join(c.name for c in children[:10])}")
    if siblings:
        lines.append(f"Siblings ({len(siblings)}): {', '.join(s.name for s in siblings[:10])}")

    return "\n".join(lines)


def search_v2_candidates(v1_name: str, v1_path: str, limit: int = 5) -> str:
    """Find candidate v2 categories that might match a v1 category.

    Uses Vector Search if available, otherwise falls back to SQL LIKE matching.
    """
    if VS_ENDPOINT:
        from databricks.vector_search.client import VectorSearchClient
        vsc = VectorSearchClient()
        index = vsc.get_index(VS_ENDPOINT, f"{CATALOG}.{SCHEMA}.nodes_vs_index")

        results = index.similarity_search(
            query_text=v1_path,
            columns=["node_id", "name", "full_path", "taxonomy_version", "level"],
            num_results=limit * 2,
            filters={"taxonomy_version": ("=", "2")},
        )

        candidates = []
        for row in results.get("result", {}).get("data_array", []):
            candidates.append({
                "node_id": row[0],
                "name": row[1],
                "full_path": row[2],
                "score": row[-1] if len(row) > 5 else None,
            })

        if candidates:
            lines = ["Vector Search candidates (semantic similarity):"]
            for c in candidates[:limit]:
                score = f" (score: {c['score']:.3f})" if c['score'] else ""
                lines.append(f"  - {c['full_path']} [ID: {c['node_id']}]{score}")
            return "\n".join(lines)

    # Fallback: SQL-based fuzzy matching
    # Split the v1 name into words and search for v2 categories containing them
    words = [w for w in v1_name.replace("(", "").replace(")", "").split() if len(w) > 2]
    if not words:
        words = [v1_name]

    conditions = " OR ".join(f"LOWER(name) LIKE '%{w.lower()}%'" for w in words[:3])

    candidates = spark.sql(f"""
        SELECT node_id, name, full_path, level
        FROM {CATALOG}.{SCHEMA}.nodes
        WHERE taxonomy_version = '2'
          AND ({conditions})
        ORDER BY level, name
        LIMIT {limit}
    """).collect()

    if not candidates:
        # Try matching by parent path
        v1_parts = v1_path.split(" > ")
        if len(v1_parts) > 1:
            parent_name = v1_parts[-2]
            candidates = spark.sql(f"""
                SELECT node_id, name, full_path, level
                FROM {CATALOG}.{SCHEMA}.nodes
                WHERE taxonomy_version = '2'
                  AND LOWER(full_path) LIKE '%{parent_name.lower()}%'
                ORDER BY level DESC
                LIMIT {limit}
            """).collect()

    lines = ["SQL fuzzy-match candidates:"]
    for c in candidates:
        lines.append(f"  - {c.full_path} [ID: {c.node_id}]")

    return "\n".join(lines) if candidates else "No candidates found."


def get_unmapped_v1(limit: int = 10) -> str:
    """List v1 categories that don't have a MAPS_TO edge yet."""
    unmapped = spark.sql(f"""
        SELECT n.node_id, n.name, n.full_path, n.level
        FROM {CATALOG}.{SCHEMA}.nodes n
        WHERE n.taxonomy_version = '1'
          AND NOT EXISTS (
              SELECT 1 FROM {CATALOG}.{SCHEMA}.edges e
              WHERE e.source_id = n.node_id
                AND e.relationship_type = 'MAPS_TO'
          )
          AND NOT EXISTS (
              SELECT 1 FROM {CATALOG}.{SCHEMA}.proposed_mappings p
              WHERE p.v1_node_id = n.node_id
                AND p.status != 'rejected'
          )
        ORDER BY n.level, n.name
        LIMIT {limit}
    """).collect()

    if not unmapped:
        return "All v1 categories are mapped or have pending proposals."

    lines = [f"Unmapped v1 categories ({len(unmapped)} shown):"]
    for u in unmapped:
        lines.append(f"  - [{u.node_id}] {u.full_path} (level {u.level})")
    return "\n".join(lines)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Agent Definition

# COMMAND ----------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_node_context",
            "description": (
                "Get full hierarchy context for a node: "
                "its ancestors, siblings, and children. "
                "Use this to understand where a category sits in the taxonomy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node ID (e.g. 'v1_1234' or 'v2_5678')",
                    }
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_v2_candidates",
            "description": (
                "Find v2 taxonomy categories that might match a v1 category. "
                "Uses semantic similarity or fuzzy name matching. "
                "Returns a ranked list of candidates with their full paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "v1_name": {
                        "type": "string",
                        "description": "The name of the v1 category to find matches for",
                    },
                    "v1_path": {
                        "type": "string",
                        "description": "The full path of the v1 category (e.g. 'Electronics > Computers > Laptops')",
                    },
                },
                "required": ["v1_name", "v1_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unmapped_v1",
            "description": (
                "List v1 categories that don't have a mapping to v2 yet. "
                "Use this to find the next batch of categories to map."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max number of unmapped categories to return",
                        "default": 10,
                    }
                },
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_node_context": get_node_context,
    "search_v2_candidates": search_v2_candidates,
    "get_unmapped_v1": get_unmapped_v1,
}

# COMMAND ----------

SYSTEM_PROMPT = """You are a taxonomy mapping specialist. Your job is to map categories
from taxonomy v1 to their best match in taxonomy v2.

For each unmapped v1 category:
1. Use get_node_context to understand the v1 category's position in the hierarchy
2. Use search_v2_candidates to find potential v2 matches
3. Use get_node_context on the top v2 candidates to verify the match makes sense
4. Propose a mapping with a confidence score (0.0 to 1.0) and reasoning

Output your proposal in this exact JSON format for each mapping:
```json
{
  "v1_node_id": "v1_XXXX",
  "v2_node_id": "v2_XXXX",
  "confidence": 0.85,
  "reasoning": "Why this is the best match"
}
```

Confidence guide:
- 0.9+: Names are nearly identical, hierarchy position matches
- 0.7-0.9: Clear semantic match, minor naming differences
- 0.5-0.7: Plausible match, but hierarchy context differs
- <0.5: Best guess, needs human review

If no good match exists, say so and set confidence to 0.0."""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run the Agent

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")


def _get(obj, key, default=None):
    """Get a value from a dict or object attribute."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_dict(obj):
    """Convert a response object or dict to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return dict(obj)


@mlflow.trace(name="mapping_agent_turn")
def run_agent_turn(messages: list[dict]) -> tuple[list[dict], str]:
    """Run one turn of the agent loop. Returns updated messages and final text."""
    response = client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "max_tokens": 2048,
        },
    )

    # Handle both object-style and dict-style responses
    choices = _get(response, "choices", [])
    choice = choices[0] if choices else {}
    assistant_msg = _get(choice, "message", choice)

    messages.append(_to_dict(assistant_msg))

    # If the LLM wants to call tools, execute them
    tool_calls = _get(assistant_msg, "tool_calls")
    if tool_calls:
        for tc in tool_calls:
            fn_obj = _get(tc, "function", {})
            fn_name = _get(fn_obj, "name")
            fn_args = json.loads(_get(fn_obj, "arguments", "{}"))
            tc_id = _get(tc, "id", "")

            with mlflow.start_span(name=f"tool:{fn_name}") as span:
                span.set_inputs(fn_args)
                result = TOOL_FUNCTIONS[fn_name](**fn_args)
                span.set_outputs({"result": result[:500]})

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })

        # Get the LLM's response after tool results
        return run_agent_turn(messages)

    content = _get(assistant_msg, "content", "")
    return messages, content

# COMMAND ----------

@mlflow.trace(name="map_category")
def map_single_category(v1_node_id: str, v1_name: str, v1_path: str) -> dict | None:
    """Run the agent to map a single v1 category to v2."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Map this v1 category to its best v2 match:\n"
                f"  ID: {v1_node_id}\n"
                f"  Name: {v1_name}\n"
                f"  Path: {v1_path}\n\n"
                "Use the tools to explore both taxonomies, then propose a mapping."
            ),
        },
    ]

    _, final_text = run_agent_turn(messages)

    # Parse the JSON proposal from the response
    try:
        json_start = final_text.index("{")
        json_end = final_text.rindex("}") + 1
        proposal = json.loads(final_text[json_start:json_end])
        proposal["v1_node_id"] = v1_node_id
        return proposal
    except (ValueError, json.JSONDecodeError):
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Process unmapped categories

# COMMAND ----------

# Get unmapped v1 categories
unmapped = spark.sql(f"""
    SELECT n.node_id, n.name, n.full_path
    FROM {CATALOG}.{SCHEMA}.nodes n
    WHERE n.taxonomy_version = '1'
      AND NOT EXISTS (
          SELECT 1 FROM {CATALOG}.{SCHEMA}.edges e
          WHERE e.source_id = n.node_id AND e.relationship_type = 'MAPS_TO'
      )
      AND NOT EXISTS (
          SELECT 1 FROM {CATALOG}.{SCHEMA}.proposed_mappings p
          WHERE p.v1_node_id = n.node_id AND p.status != 'rejected'
      )
    ORDER BY n.level, n.name
    LIMIT {BATCH_SIZE}
""").collect()

print(f"Processing {len(unmapped)} unmapped categories...")

# COMMAND ----------

proposals = []
for i, row in enumerate(unmapped):
    print(f"[{i + 1}/{len(unmapped)}] Mapping: {row.full_path}")

    # Get the active MLflow trace ID
    proposal = map_single_category(row.node_id, row.name, row.full_path)

    if proposal and proposal.get("v2_node_id"):
        # Look up v2 name and path (sanitize to prevent SQL injection)
        v2_info = spark.sql(f"""
            SELECT name, full_path FROM {CATALOG}.{SCHEMA}.nodes
            WHERE node_id = '{_sq(proposal["v2_node_id"])}'
        """).collect()

        try:
            trace = mlflow.get_last_active_trace()
            trace_id = trace.info.request_id if trace else None
        except AttributeError:
            trace_id = None

        proposals.append({
            "v1_node_id": row.node_id,
            "v2_node_id": proposal["v2_node_id"],
            "v1_name": row.name,
            "v2_name": v2_info[0].name if v2_info else "",
            "v1_path": row.full_path,
            "v2_path": v2_info[0].full_path if v2_info else "",
            "confidence": proposal.get("confidence", 0.0),
            "reasoning": proposal.get("reasoning", ""),
            "method": "agent",
            "status": "pending",
            "trace_id": trace_id,
        })
        print(f"  -> {proposal['v2_node_id']} (confidence: {proposal.get('confidence', '?')})")
    else:
        print("  -> No mapping proposed")

print(f"\nTotal proposals: {len(proposals)}")

# COMMAND ----------

# Write proposals to Delta
if proposals:
    from pyspark.sql.functions import current_timestamp

    proposals_df = spark.createDataFrame(proposals)
    proposals_df = proposals_df.withColumn("created_at", current_timestamp())

    proposals_df.write.mode("append").saveAsTable(
        f"{CATALOG}.{SCHEMA}.proposed_mappings"
    )
    print(f"Wrote {len(proposals)} proposals to {CATALOG}.{SCHEMA}.proposed_mappings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Review Summary

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT status, COUNT(*) as count,
               ROUND(AVG(confidence), 2) as avg_confidence
        FROM {CATALOG}.{SCHEMA}.proposed_mappings
        GROUP BY status
    """)
)

# COMMAND ----------

# Show high-confidence proposals
display(
    spark.sql(f"""
        SELECT v1_path, v2_path, confidence, reasoning, trace_id
        FROM {CATALOG}.{SCHEMA}.proposed_mappings
        WHERE status = 'pending'
        ORDER BY confidence DESC
        LIMIT 20
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: Review App
# MAGIC
# MAGIC Deploy `app/` as a Databricks App to review proposals interactively.
# MAGIC Each proposal links to its MLflow trace for full audit trail.

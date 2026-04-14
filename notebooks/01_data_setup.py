# Databricks notebook source

# MAGIC %md
# MAGIC # 01 -- Data Setup: Taxonomy Parsing + GraphFrames Analytics
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Downloads two versions of the Google Product Taxonomy
# MAGIC 2. Parses them into **nodes** and **edges** Delta tables
# MAGIC 3. Runs GraphFrames batch analytics (connected components, PageRank, split/merge detection)
# MAGIC 4. Optionally creates a Vector Search index for semantic matching
# MAGIC
# MAGIC **Dataset:** [Google Product Taxonomy](https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt)
# MAGIC -- free, public, ~6,600 categories, 3-5 levels deep.

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "graph_hierarchy_demo"
VS_ENDPOINT = None  # Set to your Vector Search endpoint name to enable semantic matching

# Taxonomy URLs -- current version + cached older version
TAXONOMY_V2_URL = "https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt"
# V1 is a cached snapshot from an earlier date (bundled in data/)
TAXONOMY_V1_PATH = None  # Set if using local file, otherwise we'll generate a synthetic v1

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download and Parse Taxonomies

# COMMAND ----------

import requests


def download_taxonomy(url: str) -> str:
    """Download a Google Product Taxonomy file."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_taxonomy(text: str, version: str) -> tuple[list[dict], list[dict]]:
    """Parse Google Product Taxonomy text into nodes and edges.

    Format: "ID - Level1 > Level2 > Level3"
    Returns (nodes_list, edges_list).
    """
    nodes = []
    edges = []
    seen_ids = set()

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Parse "ID - Category > Subcategory > ..."
        if " - " not in line:
            continue

        raw_id, path_str = line.split(" - ", 1)
        parts = [p.strip() for p in path_str.split(">")]
        node_id = f"v{version}_{raw_id.strip()}"
        full_path = " > ".join(parts)
        name = parts[-1]
        level = len(parts)

        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)

        nodes.append({
            "node_id": node_id,
            "taxonomy_version": version,
            "google_id": raw_id.strip(),
            "name": name,
            "full_path": full_path,
            "level": level,
            "level_1": parts[0] if len(parts) > 0 else None,
            "level_2": parts[1] if len(parts) > 1 else None,
            "level_3": parts[2] if len(parts) > 2 else None,
            "level_4": parts[3] if len(parts) > 3 else None,
            "level_5": parts[4] if len(parts) > 4 else None,
        })

        # Build parent edge by finding the parent path
        if level > 1:
            parent_path = " > ".join(parts[:-1])
            # Find parent node_id
            for n in nodes:
                if (
                    n["taxonomy_version"] == version
                    and n["full_path"] == parent_path
                ):
                    edges.append({
                        "source_id": n["node_id"],
                        "target_id": node_id,
                        "relationship_type": "PARENT_OF",
                        "taxonomy_version": version,
                    })
                    break

    return nodes, edges

# COMMAND ----------

# Download current taxonomy (v2)
print("Downloading current taxonomy (v2)...")
v2_text = download_taxonomy(TAXONOMY_V2_URL)
v2_nodes, v2_edges = parse_taxonomy(v2_text, "2")
print(f"V2: {len(v2_nodes)} nodes, {len(v2_edges)} edges")

# COMMAND ----------

# Generate synthetic v1 by modifying v2
# (Simulates a real taxonomy migration: some categories renamed,
# some split, some merged, some removed, some added)
import random
import hashlib

random.seed(42)

v1_nodes = []
v1_edges = []

# Modifications to simulate a "previous version":
# 1. Rename ~10% of leaf categories
# 2. Remove ~5% of categories (they're "new in v2")
# 3. Merge some subcategories (2 v2 categories -> 1 v1 category)
# 4. Split some categories (1 v2 category -> 2 v1 categories)

removed_indices = set(random.sample(range(len(v2_nodes)), len(v2_nodes) // 20))
rename_indices = set(random.sample(range(len(v2_nodes)), len(v2_nodes) // 10))

# Rename patterns (simulates real taxonomy evolution)
rename_suffixes = [
    " (Legacy)", " - Classic", " & Related", " Products",
    " Items", " Goods", " Supplies", " Equipment",
]

for i, v2_node in enumerate(v2_nodes):
    if i in removed_indices:
        continue  # This category is "new in v2" -- doesn't exist in v1

    v1_node = dict(v2_node)
    v1_node["node_id"] = v1_node["node_id"].replace("v2_", "v1_")
    v1_node["taxonomy_version"] = "1"

    if i in rename_indices and v1_node["level"] >= 3:
        suffix = random.choice(rename_suffixes)
        v1_node["name"] = v1_node["name"] + suffix
        # Update the full path
        parts = v1_node["full_path"].split(" > ")
        parts[-1] = parts[-1] + suffix
        v1_node["full_path"] = " > ".join(parts)

    v1_nodes.append(v1_node)

# Build v1 edges from v1 nodes
v1_node_paths = {n["full_path"]: n["node_id"] for n in v1_nodes}
for node in v1_nodes:
    if node["level"] > 1:
        parent_path = " > ".join(node["full_path"].split(" > ")[:-1])
        if parent_path in v1_node_paths:
            v1_edges.append({
                "source_id": v1_node_paths[parent_path],
                "target_id": node["node_id"],
                "relationship_type": "PARENT_OF",
                "taxonomy_version": "1",
            })

print(f"V1 (synthetic): {len(v1_nodes)} nodes, {len(v1_edges)} edges")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Delta Tables

# COMMAND ----------

from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

node_schema = StructType([
    StructField("node_id", StringType(), False),
    StructField("taxonomy_version", StringType(), False),
    StructField("google_id", StringType(), True),
    StructField("name", StringType(), False),
    StructField("full_path", StringType(), False),
    StructField("level", IntegerType(), False),
    StructField("level_1", StringType(), True),
    StructField("level_2", StringType(), True),
    StructField("level_3", StringType(), True),
    StructField("level_4", StringType(), True),
    StructField("level_5", StringType(), True),
])

edge_schema = StructType([
    StructField("source_id", StringType(), False),
    StructField("target_id", StringType(), False),
    StructField("relationship_type", StringType(), False),
    StructField("taxonomy_version", StringType(), True),
])

# Combine both versions
all_nodes = v1_nodes + v2_nodes
all_edges = v1_edges + v2_edges

nodes_df = spark.createDataFrame(all_nodes, schema=node_schema)
edges_df = spark.createDataFrame(all_edges, schema=edge_schema)

nodes_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.nodes")
edges_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.edges")

print(f"Nodes table: {nodes_df.count()} rows")
print(f"Edges table: {edges_df.count()} rows")

# COMMAND ----------

# Summary stats
display(
    spark.sql(f"""
        SELECT taxonomy_version, COUNT(*) as node_count,
               MIN(level) as min_depth, MAX(level) as max_depth,
               ROUND(AVG(level), 1) as avg_depth
        FROM {CATALOG}.{SCHEMA}.nodes
        GROUP BY taxonomy_version
    """)
)

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT taxonomy_version, level, COUNT(*) as count
        FROM {CATALOG}.{SCHEMA}.nodes
        GROUP BY taxonomy_version, level
        ORDER BY taxonomy_version, level
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Seed Known Mappings (Exact Name Matches)

# COMMAND ----------

# Pre-seed MAPS_TO edges for categories with identical names across versions
v1_names = spark.sql(f"""
    SELECT node_id as v1_id, name, full_path as v1_path
    FROM {CATALOG}.{SCHEMA}.nodes WHERE taxonomy_version = '1'
""")
v2_names = spark.sql(f"""
    SELECT node_id as v2_id, name, full_path as v2_path
    FROM {CATALOG}.{SCHEMA}.nodes WHERE taxonomy_version = '2'
""")

exact_matches = v1_names.join(v2_names, "name")

from pyspark.sql.functions import lit

mapping_edges = exact_matches.select(
    exact_matches.v1_id.alias("source_id"),
    exact_matches.v2_id.alias("target_id"),
    lit("MAPS_TO").alias("relationship_type"),
    lit(None).cast("string").alias("taxonomy_version"),
)

# Append to edges table
mapping_edges.write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.edges")

print(f"Seeded {mapping_edges.count()} exact-match MAPS_TO edges")
print(f"Unmapped v1 categories: {v1_names.count() - mapping_edges.count()}")

# COMMAND ----------

# Create a tracking table for proposed mappings
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.proposed_mappings (
        v1_node_id STRING,
        v2_node_id STRING,
        v1_name STRING,
        v2_name STRING,
        v1_path STRING,
        v2_path STRING,
        confidence DOUBLE,
        reasoning STRING,
        method STRING,
        status STRING DEFAULT 'pending',
        reviewer_note STRING,
        trace_id STRING,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        reviewed_at TIMESTAMP
    )
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. GraphFrames Batch Analytics

# COMMAND ----------

from graphframes import GraphFrame

# Build graph from all nodes and PARENT_OF edges (within each version)
gf_nodes = spark.table(f"{CATALOG}.{SCHEMA}.nodes") \
    .withColumnRenamed("node_id", "id")

gf_edges = spark.table(f"{CATALOG}.{SCHEMA}.edges") \
    .filter("relationship_type = 'PARENT_OF'") \
    .withColumnRenamed("source_id", "src") \
    .withColumnRenamed("target_id", "dst")

g = GraphFrame(gf_nodes, gf_edges)
print(f"Graph: {g.vertices.count()} vertices, {g.edges.count()} edges")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connected Components
# MAGIC Find isolated subtrees -- these are independent parts of the taxonomy
# MAGIC that can be mapped separately.

# COMMAND ----------

sc.setCheckpointDir("/tmp/graphframes_checkpoints")
components = g.connectedComponents()

component_summary = components.groupBy("component") \
    .count() \
    .orderBy("count", ascending=False)

display(component_summary)
# Expect: v1 hierarchy as one component, v2 as another (they're not connected
# via PARENT_OF edges). MAPS_TO edges would connect them if included.

# COMMAND ----------

# MAGIC %md
# MAGIC ### PageRank
# MAGIC Identify the most "important" categories -- the ones that sit at key
# MAGIC branching points in the hierarchy.

# COMMAND ----------

pr = g.pageRank(resetProbability=0.15, maxIter=20)

top_nodes = pr.vertices.select(
    "id", "name", "taxonomy_version", "level", "pagerank"
).orderBy("pagerank", ascending=False)

display(top_nodes.limit(20))

# COMMAND ----------

# Save enriched nodes with PageRank scores
enriched = pr.vertices.join(
    components.select("id", "component"), "id"
)
enriched.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.nodes_enriched"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split/Merge Detection
# MAGIC Find v1 categories that map to multiple v2 categories (splits) or
# MAGIC v2 categories that receive mappings from multiple v1 categories (merges).

# COMMAND ----------

display(
    spark.sql(f"""
        WITH mapping_counts AS (
            SELECT source_id, COUNT(DISTINCT target_id) as v2_count
            FROM {CATALOG}.{SCHEMA}.edges
            WHERE relationship_type = 'MAPS_TO'
            GROUP BY source_id
            HAVING COUNT(DISTINCT target_id) > 1
        )
        SELECT mc.source_id, mc.v2_count, n.name, n.full_path
        FROM mapping_counts mc
        JOIN {CATALOG}.{SCHEMA}.nodes n ON n.node_id = mc.source_id
        ORDER BY mc.v2_count DESC
        LIMIT 20
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Vector Search Index (Optional)

# COMMAND ----------

if VS_ENDPOINT:
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()

    # Create a Delta Sync index on the nodes table
    # Embeds the 'full_path' column for semantic matching
    index_name = f"{CATALOG}.{SCHEMA}.nodes_vs_index"

    try:
        vsc.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT,
            index_name=index_name,
            source_table_name=f"{CATALOG}.{SCHEMA}.nodes",
            pipeline_type="TRIGGERED",
            primary_key="node_id",
            embedding_source_columns=[
                {"name": "full_path", "model_endpoint_name": "databricks-gte-large-en"}
            ],
            columns_to_sync=["node_id", "taxonomy_version", "name", "full_path", "level"],
        )
        print(f"Vector Search index created: {index_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Vector Search index already exists: {index_name}")
            # Trigger a sync
            vsc.get_index(VS_ENDPOINT, index_name).sync()
        else:
            raise
else:
    print("Skipping Vector Search (VS_ENDPOINT not set)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Data is ready:
# MAGIC - `nodes` -- all categories from both taxonomy versions
# MAGIC - `edges` -- PARENT_OF (hierarchy) + MAPS_TO (known mappings)
# MAGIC - `nodes_enriched` -- nodes with PageRank + component ID
# MAGIC - `proposed_mappings` -- empty, ready for the agent to populate
# MAGIC
# MAGIC Next: run `02_mapping_agent` to propose mappings for unmapped categories.

# COMMAND ----------

# Quick look at what the agent needs to map
unmapped = spark.sql(f"""
    SELECT COUNT(*) as unmapped_count FROM {CATALOG}.{SCHEMA}.nodes n
    WHERE n.taxonomy_version = '1'
      AND NOT EXISTS (
          SELECT 1 FROM {CATALOG}.{SCHEMA}.edges e
          WHERE e.source_id = n.node_id
            AND e.relationship_type = 'MAPS_TO'
      )
""")
display(unmapped)

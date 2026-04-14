# Databricks notebook source

# MAGIC %md
# MAGIC # 01 -- Data Setup: Taxonomy Parsing + GraphFrames Analytics
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads two real versions of the Google Product Taxonomy (2015 + 2021)
# MAGIC 2. Parses them into **nodes** and **edges** Delta tables
# MAGIC 3. Seeds exact-match mappings and identifies unmapped categories
# MAGIC 4. Runs GraphFrames batch analytics (connected components, PageRank)
# MAGIC 5. Optionally creates a Vector Search index for semantic matching
# MAGIC
# MAGIC **Dataset:** [Google Product Taxonomy](https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt)
# MAGIC -- free, public, ~5,500 categories, 3-7 levels deep.
# MAGIC
# MAGIC **Two real versions:**
# MAGIC - V1: 2015-02-19 (5,427 categories)
# MAGIC - V2: 2021-09-21 (5,595 categories)
# MAGIC - 5,404 exact matches, 23 removed/renamed, 191 new categories

# COMMAND ----------

# Configuration
CATALOG = "main"
SCHEMA = "graph_hierarchy_demo"
VS_ENDPOINT = None  # Set to your Vector Search endpoint name to enable semantic matching

# Taxonomy files -- bundled in data/ directory
# Override these if you want to load from UC Volumes or URLs instead
TAXONOMY_V1_URL = (
    "https://web.archive.org/web/20190301000000/"
    "https://www.google.com/basepages/producttype/"
    "taxonomy-with-ids.en-US.txt"
)
TAXONOMY_V2_URL = (
    "https://www.google.com/basepages/producttype/"
    "taxonomy-with-ids.en-US.txt"
)

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load and Parse Taxonomies

# COMMAND ----------

import requests


def download_taxonomy(url: str) -> str:
    """Download a Google Product Taxonomy file."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


def parse_taxonomy(text: str, version: str) -> tuple[list[dict], list[dict]]:
    """Parse Google Product Taxonomy text into nodes and edges.

    Each line: "ID - Level1 > Level2 > Level3"
    Returns (nodes, parent_of_edges).
    """
    nodes = []
    edges = []
    # Map full_path -> node_id for parent lookup
    path_to_id: dict[str, str] = {}

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " - " not in line:
            continue

        raw_id, path_str = line.split(" - ", 1)
        parts = [p.strip() for p in path_str.split(">")]
        node_id = f"v{version}_{raw_id.strip()}"
        full_path = " > ".join(parts)
        name = parts[-1]
        level = len(parts)

        nodes.append({
            "node_id": node_id,
            "taxonomy_version": version,
            "google_id": raw_id.strip(),
            "name": name,
            "full_path": full_path,
            "level": level,
        })
        path_to_id[full_path] = node_id

        # Parent edge
        if level > 1:
            parent_path = " > ".join(parts[:-1])
            parent_id = path_to_id.get(parent_path)
            if parent_id:
                edges.append({
                    "source_id": parent_id,
                    "target_id": node_id,
                    "relationship_type": "PARENT_OF",
                    "taxonomy_version": version,
                })

    return nodes, edges

# COMMAND ----------

print("Downloading V1 taxonomy (2015)...")
v1_text = download_taxonomy(TAXONOMY_V1_URL)
v1_nodes, v1_edges = parse_taxonomy(v1_text, "1")
print(f"  V1: {len(v1_nodes)} nodes, {len(v1_edges)} edges")

print("Downloading V2 taxonomy (2021)...")
v2_text = download_taxonomy(TAXONOMY_V2_URL)
v2_nodes, v2_edges = parse_taxonomy(v2_text, "2")
print(f"  V2: {len(v2_nodes)} nodes, {len(v2_edges)} edges")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Delta Tables

# COMMAND ----------

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

node_schema = StructType([
    StructField("node_id", StringType(), False),
    StructField("taxonomy_version", StringType(), False),
    StructField("google_id", StringType(), True),
    StructField("name", StringType(), False),
    StructField("full_path", StringType(), False),
    StructField("level", IntegerType(), False),
])

edge_schema = StructType([
    StructField("source_id", StringType(), False),
    StructField("target_id", StringType(), False),
    StructField("relationship_type", StringType(), False),
    StructField("taxonomy_version", StringType(), True),
])

all_nodes = v1_nodes + v2_nodes
all_edges = v1_edges + v2_edges

nodes_df = spark.createDataFrame(all_nodes, schema=node_schema)
edges_df = spark.createDataFrame(all_edges, schema=edge_schema)

nodes_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.nodes")
edges_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.edges")

print(f"Nodes: {nodes_df.count()} | Edges: {edges_df.count()}")

# COMMAND ----------

# Summary by version and depth
display(spark.sql(f"""
    SELECT taxonomy_version, level, COUNT(*) as count
    FROM {CATALOG}.{SCHEMA}.nodes
    GROUP BY taxonomy_version, level
    ORDER BY taxonomy_version, level
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Seed Known Mappings (Exact Name Matches)

# COMMAND ----------

from pyspark.sql import functions as F

v1_df = spark.sql(f"""
    SELECT node_id as v1_id, name, full_path as v1_path
    FROM {CATALOG}.{SCHEMA}.nodes WHERE taxonomy_version = '1'
""")
v2_df = spark.sql(f"""
    SELECT node_id as v2_id, name, full_path as v2_path
    FROM {CATALOG}.{SCHEMA}.nodes WHERE taxonomy_version = '2'
""")

# Exact match on full_path (not just name -- avoids ambiguous leaf matches)
exact_matches = v1_df.alias("v1").join(
    v2_df.alias("v2"), v1_df.v1_path == v2_df.v2_path,
)

mapping_edges = exact_matches.select(
    F.col("v1_id").alias("source_id"),
    F.col("v2_id").alias("target_id"),
    F.lit("MAPS_TO").alias("relationship_type"),
    F.lit(None).cast("string").alias("taxonomy_version"),
)

mapping_edges.write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.edges")

total_v1 = v1_df.count()
matched = mapping_edges.count()
unmapped = total_v1 - matched

print(f"Exact-match MAPS_TO edges: {matched}")
print(f"Unmapped V1 categories:    {unmapped}")
print(f"Match rate:                {matched / total_v1:.1%}")

# COMMAND ----------

# What's unmapped? These are the interesting ones -- renames, splits, removals
display(spark.sql(f"""
    SELECT n.node_id, n.name, n.full_path, n.level
    FROM {CATALOG}.{SCHEMA}.nodes n
    WHERE n.taxonomy_version = '1'
      AND NOT EXISTS (
          SELECT 1 FROM {CATALOG}.{SCHEMA}.edges e
          WHERE e.source_id = n.node_id AND e.relationship_type = 'MAPS_TO'
      )
    ORDER BY n.full_path
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Tracking Tables

# COMMAND ----------

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
        status STRING,
        reviewer_note STRING,
        trace_id STRING,
        created_at TIMESTAMP,
        reviewed_at TIMESTAMP
    )
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. GraphFrames Batch Analytics

# COMMAND ----------

try:
    from graphframes import GraphFrame
    GRAPHFRAMES_AVAILABLE = True
except ImportError:
    GRAPHFRAMES_AVAILABLE = False
    print("GraphFrames not available (serverless compute).")
    print("Skipping batch graph analytics. Run on ML Runtime for full demo.")

# COMMAND ----------

if GRAPHFRAMES_AVAILABLE:
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
# MAGIC V1 and V2 are separate trees (connected only via MAPS_TO edges, which
# MAGIC we excluded). Each version should be one large connected component.

# COMMAND ----------

if GRAPHFRAMES_AVAILABLE:
    sc.setCheckpointDir("/tmp/graphframes_checkpoints")
    components = g.connectedComponents()

    display(
        components.groupBy("component", "taxonomy_version")
        .count()
        .orderBy("count", ascending=False)
        .limit(10)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### PageRank
# MAGIC Categories with high PageRank are key branching points in the hierarchy.
# MAGIC These are the most important categories to map correctly.

# COMMAND ----------

if GRAPHFRAMES_AVAILABLE:
    pr = g.pageRank(resetProbability=0.15, maxIter=20)

    display(
        pr.vertices
        .select("id", "name", "taxonomy_version", "level", "pagerank")
        .orderBy("pagerank", ascending=False)
        .limit(20)
    )

# COMMAND ----------

# Save enriched nodes (or just copy nodes if GraphFrames unavailable)
if GRAPHFRAMES_AVAILABLE:
    enriched = pr.vertices.join(components.select("id", "component"), "id")
    enriched.write.mode("overwrite").saveAsTable(
        f"{CATALOG}.{SCHEMA}.nodes_enriched"
    )
else:
    spark.table(f"{CATALOG}.{SCHEMA}.nodes") \
        .withColumnRenamed("node_id", "id") \
        .write.mode("overwrite") \
        .saveAsTable(f"{CATALOG}.{SCHEMA}.nodes_enriched")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Vector Search Index (Optional)

# COMMAND ----------

if VS_ENDPOINT:
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()
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
            columns_to_sync=[
                "node_id", "taxonomy_version", "name", "full_path", "level",
            ],
        )
        print(f"Created Vector Search index: {index_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Index already exists: {index_name}")
            vsc.get_index(VS_ENDPOINT, index_name).sync()
        else:
            raise
else:
    print("Skipping Vector Search -- set VS_ENDPOINT to enable.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Table | Contents |
# MAGIC |-------|----------|
# MAGIC | `nodes` | All categories from both taxonomy versions |
# MAGIC | `edges` | PARENT_OF (hierarchy) + MAPS_TO (exact matches) |
# MAGIC | `nodes_enriched` | Nodes with PageRank + component ID |
# MAGIC | `proposed_mappings` | Empty -- agent populates in notebook 02 |
# MAGIC
# MAGIC **Next:** Run `02_mapping_agent` to map the unmapped categories.

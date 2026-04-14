# Graph Hierarchy Mapping

A Databricks demo for **hierarchy navigation + agent-based mapping** using GraphFrames (batch analytics), recursive CTEs (interactive traversal), Vector Search (semantic matching), and an LLM agent.

**Use cases:** product hierarchy migration, taxonomy mapping, dependency analysis, regulatory compliance tracing, supply chain impact analysis.

## Architecture

```
                    Unity Catalog Delta Tables
                    (nodes + edges, two taxonomy versions)
                           |
            +--------------+--------------+
            |              |              |
      GraphFrames     Recursive CTEs   Vector Search
      (Spark batch)   (Lakebase/SQL)   (semantic matching)
            |              |              |
      Components,     "What maps to    "Find categories
      PageRank,       category X in     similar to
      split/merge     the new version?" planning style"
      detection            |              |
            |              +----- + ------+
            |                     |
            |              Mapping Agent
            |              (proposes v1 -> v2 mappings)
            |                     |
            +---------------------+
                          |
                    Review App
                    (approve/reject/edit mappings)
                    (MLflow tracing for audit)
```

## Components

| Component | What it does |
|-----------|-------------|
| `notebooks/01_data_setup.py` | Downloads Google Product Taxonomy (two versions), parses into nodes + edges, runs GraphFrames batch analytics |
| `notebooks/02_mapping_agent.py` | Agent with CTE traversal tools + Vector Search for semantic matching. Proposes category mappings. |
| `app/` | Streamlit review UI for approving/rejecting agent proposals. MLflow traces linked. |

## Quick Start

```bash
# Clone and deploy
git clone https://github.com/ScottHMcKean/graph-hierarchy-mapping.git
cd graph-hierarchy-mapping
databricks bundle deploy -t dev

# Run data setup
databricks bundle run data_setup -t dev

# Run mapping agent
databricks bundle run mapping_agent -t dev

# Deploy review app
databricks bundle deploy -t dev  # app deploys automatically
```

## Dataset

Two real versions of the [Google Product Taxonomy](https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt):

| Version | Date | Categories | Source |
|---------|------|-----------|--------|
| V1 | 2015-02-19 | 5,427 | Wayback Machine archive |
| V2 | 2021-09-21 | 5,595 | Current Google release |

Real taxonomy evolution over 6 years: 5,404 exact matches, 23 removed/renamed categories (e.g. "Football" -> "American Football"), 191 new categories added. The unmapped categories are the interesting ones -- renames, splits, and restructurings that the agent must resolve.

## Requirements

- Databricks ML Runtime 15.4+ (GraphFrames pre-installed)
- Unity Catalog enabled
- Vector Search endpoint (optional -- agent works without it, just loses semantic matching)
- Foundation Model API access (for the LLM agent)

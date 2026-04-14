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
git clone https://github.com/your-org/graph-hierarchy-mapping.git
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

[Google Product Taxonomy](https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt) -- a free, public, versioned product classification with ~6,600 categories and 3-5 hierarchy levels. We use two versions to demonstrate mapping between an "old" and "new" taxonomy.

## Requirements

- Databricks ML Runtime 15.4+ (GraphFrames pre-installed)
- Unity Catalog enabled
- Vector Search endpoint (optional -- agent works without it, just loses semantic matching)
- Foundation Model API access (for the LLM agent)

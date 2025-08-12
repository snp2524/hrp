# RAG Pipeline Comparison Experiment

This project demonstrates multiple Retrieval Augmented Generation (RAG) strategies – including a **knowledge-graph-assisted** approach – on the open-source Synthea clinical notes dataset. It orchestrates data ingestion, graph construction in Neo4j, pipeline building with LangChain, and rich result analysis/visualisation.

---

## Contents

| Path                 | Purpose                                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `main.py`            | Orchestrates the full experiment (data → graph → pipelines → evaluation → analysis).                                     |
| `config.py`          | Centralised configuration helpers. Constructs the Gemini LLM instance and exposes flags such as `USE_GEMINI_EMBEDDINGS`. |
| `data_loader.py`     | Downloads the latest **Synthea** sample data, extracts diagnostic reports, and converts them into LangChain `Document`s. |
| `rag_pipelines.py`   | Builds four RAG pipelines: Standard, Parent-Document, Sentence-Window, and Knowledge-Graph RAG.                          |
| `knowledge_graph.py` | Extracts entity/relationship triples with an LLM and stores them in Neo4j; also exposes a Cypher-backed QA chain.        |
| `analysis.py`        | Aggregates timing & quality metrics, performs statistical tests, and writes publication-ready plots.                     |
| `requirements.txt`   | Locked list of Python dependencies.                                                                                      |

---

## Quick-start

```bash
# 1. Clone and enter the repo
$ git clone https://github.com/snp2524/hrp.git
$ cd hrp

# 2. (Recommended) create and activate a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install Python dependencies
$ pip install -r requirements.txt

# 4. Export the required environment variables (see below)
$ export GOOGLE_API_KEY="<your-google-api-key>"
$ export NEO4J_URI="bolt://localhost:7687"
$ export NEO4J_USERNAME="neo4j"
$ export NEO4J_PASSWORD="<password>"

# 5. Run the experiment
$ python main.py
```

> **Tip:** the first run downloads ~50 MB of Synthea data and can take a few minutes while the knowledge graph is built.

---

## Required Environment Variables

| Variable         | Description                                                          |
| ---------------- | -------------------------------------------------------------------- |
| `GOOGLE_API_KEY` | Google Generative AI key for the Gemini LLM and embedding models.    |
| `NEO4J_URI`      | Bolt endpoint of your Neo4j instance (e.g. `bolt://localhost:7687`). |
| `NEO4J_USERNAME` | Database username (default `neo4j`).                                 |
| `NEO4J_PASSWORD` | Password for the specified user.                                     |

Optional:

| Variable                | Purpose                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| `USE_GEMINI_EMBEDDINGS` | Set to `True` to use Google embeddings (otherwise HuggingFace is used). |

> All variables can be placed in a local `.env` file – automatically loaded by `python-dotenv`.

---

## Neo4j Setup

1. **Install Neo4j** – via [Neo4j Desktop](https://neo4j.com/download/) or Docker:
   ```bash
   docker run -d \
     --name neo4j \
     -p 7687:7687 -p 7474:7474 \
     -e NEO4J_AUTH=neo4j/<password> \
     neo4j:5
   ```
2. Expose the credentials with `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.
3. The `knowledge_graph.py` helper will **wipe** the database at the beginning of each run (`MATCH (n) DETACH DELETE n`). Point it at a dedicated workspace.

Alternatively, you can use the online free instance of Neo4j and just specify the relevant environment variables.

---

## Outputs

Upon completion the script writes:

- `rag_pipeline_results.csv` – per-question metrics.
- `rag_pipeline_comparison.png` – overview composite figure.
- Four individual PNGs (`plot_avg_response_time.png`, etc.).

These assets can be embedded directly into reports or presentations.

---

## License

This academic project is released under the MIT License. See `LICENSE` for details.

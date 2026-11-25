# rag-local

Local, simple, framework-free RAG (Retrieval-Augmented Generation). The project indexes the
documents in the `docs/` folder and answers questions in Portuguese **based only on them**,
always citing the source. Everything runs on your machine: Ollama for embeddings and generation,
DuckDB for storage and search.

It doesn't use LangChain, LlamaIndex or a dedicated vector database. Each piece is a small
module you can read in one sitting.

> The CLI messages, the answer prompt and the evaluation file keys (`pergunta`, `fonte`,
> `pagina`, `trecho`) are in Portuguese, because the target documents and questions are.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package and environment manager)
- [Ollama](https://ollama.com/download)

## Installation

### 1. uv

`requirements.txt` is only there to install uv. The project dependencies live in
`pyproject.toml`.

```powershell
python -m pip install -r requirements.txt
```

Alternatives: `winget install astral-sh.uv` (Windows) or the official installer
(`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`).

> **`uv` not recognized?** pip installs the executable in your Python's `Scripts` folder
> (e.g. `C:\Users\<you>\AppData\Local\Programs\Python\Python312\Scripts`), which isn't always on
> the PATH. Add that folder to the PATH, or run `python -m uv ...` with that same Python.
> Don't install uv **inside** the project's `.venv`: `uv sync` removes anything that isn't in
> `pyproject.toml`.

### 2. Project dependencies

```powershell
uv sync          # creates/updates .venv with the dependencies and the dev ones (pytest)
uv run rag --help
```

### 3. Ollama and models

Install Ollama and check that the server is running. On Windows and macOS the app starts the
server for you; on Linux, run `ollama serve`.

```powershell
ollama --version
curl http://localhost:11434/api/tags      # should list the installed models

ollama pull nomic-embed-text              # embeddings, 768 dimensions (required)
ollama pull qwen2.5:7b                    # generation (see "Generation model")
```

## Where to put your documents

Put `.md`, `.txt`, `.pdf` and `.docx` files in the **`docs/`** folder at the project root.
Subfolders are read recursively, and the source is stored as a relative path
(e.g. `contratos/2026/aditivo.pdf`). Hidden files and Office temporary files (`~$...`) are
skipped.

If `docs/` doesn't exist or is empty on the first ingestion, `rag ingest` creates three sample
files so you can try it right away. To use a different folder, change `docs_dir` in
`config.toml` or set `RAG_DOCS_DIR`.

> Scanned PDFs (images only) have no extractable text. `ingest` warns about them, but it doesn't
> run OCR.

## Commands

```powershell
uv run rag ingest                         # indexes docs/ (incremental)
uv run rag stats                          # files, chunks and index parameters
uv run rag ask "Qual o valor da taxa de inscrição?"
uv run rag ask "..." --verbose            # + chunk text and each search's rank
uv run rag ask "..." -k 6 --mode vector   # top k and search type (hybrid | vector | bm25)
uv run rag eval                           # hit rate and MRR for the three search modes
uv run rag eval -k 8 -v                   # different k, with the rank per question
uv run rag eval --mode hybrid -f other.yaml
```

With the `.venv` activated (`.venv\Scripts\activate`), you can call just `rag ...`.

### `rag ingest`

- Computes each file's SHA-256 and only reprocesses new or changed files.
- Removes from the index any files that left `docs/`.
- If you change `embed_model`, `embed_dim`, `embed_doc_prefix`, `chunk_size` or `chunk_overlap`,
  the old index no longer matches, so `ingest` reindexes everything automatically (and tells you
  why). `ask` and `eval` refuse to run until you reindex.
- A corrupted file doesn't stop the ingestion: it's reported as a failure and retried on the next
  run.

### `rag ask`

Shows the answer (streamed) and a table with the chunks used and each one's score. With
`--verbose`, it also shows each chunk's text and, in hybrid mode, the rank and score the chunk got
in the vector search and in BM25. That helps you see why a passage ended up in the context.

If the context doesn't contain the answer, the model is instructed to reply exactly
`Não encontrei nos documentos.` ("Not found in the documents.").

## Configuration

Priority order: **code defaults < `config.toml` < `RAG_*` environment variables** (which can
also come from a `.env` file; see `.env.example`).

| Key | Default | What it is |
|---|---|---|
| `docs_dir` | `docs` | documents folder |
| `db_path` | `rag.duckdb` | database file (outside `docs/`, ignored by git) |
| `eval_file` | `eval/perguntas.yaml` | questions for `rag eval` |
| `ollama_url` | `http://localhost:11434` | Ollama server |
| `embed_model` / `embed_dim` | `nomic-embed-text` / `768` | embedding model and dimension |
| `embed_doc_prefix` / `embed_query_prefix` | `search_document: ` / `search_query: ` | nomic task prefixes |
| `gen_model` | `llama3.1` (`config.toml` uses `qwen2.5-coder:7b`) | generation model |
| `gen_temperature` / `gen_num_ctx` | `0.1` / `8192` | generation parameters |
| `chunk_size` / `chunk_overlap` | `800` / `100` | chunking, in characters |
| `top_k` | `4` | chunks in the context |
| `search_mode` | `hybrid` | `hybrid`, `vector` or `bm25` |
| `candidates` | `20` | candidates from each search before fusion |
| `rrf_k` | `60` | RRF constant |

Example, switching the generation model for the current PowerShell session only:

```powershell
$env:RAG_GEN_MODEL = "qwen3:8b"; uv run rag ask "..."
```

### Generation model

`config.toml` ships with `qwen2.5-coder:7b`, which is there **only to validate the pipeline**.
It's specialized in code. For real use, prefer a **general instruction model**, which follows the
prompt rules better and writes more natural Portuguese:

- **`qwen3:8b`**: the top pick. It "thinks" before answering: slower, but the reasoning doesn't
  show up in the output, because the code only prints `message.content`.
- **`qwen2.5:7b`**: faster, good at Portuguese.

```powershell
ollama pull qwen3:8b
# and in config.toml: gen_model = "qwen3:8b"
```

Changing the generation model does **not** require reindexing. Only the embedding model does.

## Evaluation (`rag eval`)

`eval/perguntas.yaml` holds questions with expected answers. A retrieved chunk counts as a hit
when it meets **all** the criteria given:

```yaml
perguntas:
  - pergunta: Quanto custa a inscrição?                # question
    fonte: edital-dataprev_supe-versaofinal.pdf       # source: relative path in docs/
    pagina: 5                                         # page: optional (PDF)
    trecho: R$ 110,00                                 # passage: optional; compared ignoring accents/case
```

- **Hit rate@k**: share of questions where some relevant chunk came back in the top k.
- **MRR@k**: mean of `1/rank` of the first relevant chunk (0 when none comes back). It rewards
  putting the right passage first.

Without `--mode`, the command compares the three modes side by side. It also warns when a
question's expected answer matches **no** chunk in the index (wrong source, or a passage split
across two chunks). Without that warning, a mistake in the expected answers would show up as a
low score.

**How to compare tweaks:** change one parameter at a time, run `rag ingest` if it's an indexing
parameter, then run `rag eval -v`. Examples:

```powershell
$env:RAG_CHUNK_SIZE = "500"; $env:RAG_CHUNK_OVERLAP = "80"; uv run rag ingest; uv run rag eval -v
$env:RAG_CANDIDATES = "50"; uv run rag eval      # no reindex needed
Remove-Item Env:RAG_CHUNK_SIZE, Env:RAG_CHUNK_OVERLAP, Env:RAG_CANDIDATES; uv run rag ingest
```

Initial results (16 questions, 800/100 chunks, `nomic-embed-text`):

| Mode | Hit rate@4 | MRR@4 | Hit rate@8 | MRR@8 |
|---|---|---|---|---|
| hybrid | 56.2% | 0.448 | 75.0% | 0.477 |
| vector | 43.8% | 0.406 | 50.0% | 0.419 |
| bm25 | 68.8% | 0.516 | 75.0% | 0.523 |

On this corpus, a public-exam notice full of exact terms, BM25 alone beats vector search, and the
fusion lands in between. Treat this as a starting point for tuning, not a conclusion.

## Tests

```powershell
uv run pytest -q
```

The tests cover chunking, incremental ingestion, vector search, BM25 (accents, stemming and
stopwords), RRF, the prompt and the metrics. They **don't need Ollama**: a `FakeEmbedder`
(hashed bag-of-words) stands in for the embeddings, and the HTTP client is tested with
`httpx.MockTransport`. The first run needs internet access so DuckDB can download the `fts`
extension, if it isn't cached yet.

## Layout

```
├── config.toml              # configuration
├── docs/                    # your documents
├── eval/perguntas.yaml      # evaluation set
├── src/rag_local/
│   ├── config.py            # typed Settings: defaults < config.toml < RAG_*
│   ├── loaders.py           # md/txt/pdf/docx -> sections (text + page)
│   ├── chunking.py          # character window that breaks at paragraph/sentence
│   ├── ollama.py            # minimal HTTP client with actionable errors
│   ├── embeddings.py        # batched /api/embed, with task prefixes
│   ├── store.py             # DuckDB: schema, transactions, vector search and FTS
│   ├── ingest.py            # hash -> diff -> load -> chunk -> embed -> store
│   ├── retrieval.py         # vector, BM25 and RRF
│   ├── generation.py        # prompt and streamed /api/chat
│   ├── evaluation.py        # hit rate and MRR
│   ├── samples.py           # sample documents
│   └── cli.py               # typer: ingest, ask, stats, eval
└── tests/
```

## DuckDB schema

```sql
files  (source VARCHAR PRIMARY KEY, sha256 VARCHAR, size_bytes BIGINT, n_chunks INTEGER, ingested_at TIMESTAMP)
chunks (id BIGINT PRIMARY KEY DEFAULT nextval('chunks_id_seq'), source VARCHAR, page INTEGER,
        chunk_index INTEGER, text VARCHAR, embedding FLOAT[768])
meta   (key VARCHAR PRIMARY KEY, value VARCHAR)   -- parameters the index was built with
stopwords_pt (sw VARCHAR)                         -- stopwords for the FTS index
fts_main_chunks.*                                 -- created by the fts extension
```

You can inspect the database directly: `uv run python -c "import duckdb; print(duckdb.connect('rag.duckdb').sql('from files'))"`.

## Architecture decisions

- **No frameworks.** The whole pipeline fits in ~1,300 lines, CLI included. The goal is to
  understand each step, not hide the steps behind abstractions.
- **DuckDB as the vector store.** `FLOAT[768]` arrays with `array_cosine_similarity` do exact
  search, a full scan. For thousands of chunks that takes milliseconds, and the result is exact
  (no approximation, unlike HNSW indexes). If the corpus grows a lot, DuckDB's `vss` extension
  adds HNSW without switching databases.
- **Character chunking that never crosses pages.** Each PDF page is chunked on its own, so the
  `[file.pdf, p. N]` citation is always exact. The cut looks, in the second half of the window,
  for a paragraph end, then a sentence end, then a line break, and finally a space. That avoids
  tiny chunks. The overlap restarts at the beginning of a word.
- **`search_document:` / `search_query:` prefixes.** `nomic-embed-text` was trained with them;
  without the prefixes, retrieval gets worse. They're configurable, for other models.
- **Hybrid search with RRF.** Cosine ranges from 0 to 1 and BM25 has no upper bound, so the
  scores aren't comparable. RRF only uses the ranks (`Σ 1/(60 + rank)`), so no normalization is
  needed. Each search returns `candidates` items before fusion.
- **Portuguese BM25.** The index uses the `portuguese` stemmer, accent stripping, a custom
  stopword list (without "não", which changes the meaning) and keeps digits. DuckDB's default
  would drop numbers like "001/2026" and "R$ 110".
- **The FTS index is rebuilt on every ingestion that changes something.** DuckDB doesn't update
  this index on its own when the table changes.
- **No FOREIGN KEY between `chunks` and `files`.** In DuckDB, a FK gets in the way of deleting and
  re-inserting. Consistency comes from one transaction per file (delete chunks, insert new ones,
  update `files`).
- **`meta` table.** It stores the index's model, dimension and chunking parameters. That prevents
  mixing vectors from different models, or querying an inconsistent index.
- **Streamed answers, with explicit `num_ctx`.** Ollama's default context can silently truncate
  the prompt. If no chunk is retrieved, the model isn't even called.
- **Deliberately out of scope:** OCR, cross-encoder reranking, conversation history and
  metadata-filtered search. These are natural next steps.

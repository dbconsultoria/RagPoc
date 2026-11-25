# rag-local

RAG (Retrieval-Augmented Generation) local, simples e sem frameworks. O projeto indexa os documentos
da pasta `docs/` e responde perguntas em português **só com base neles**, sempre citando a fonte.
Tudo roda na sua máquina: Ollama para embeddings e geração, DuckDB para armazenamento e busca.

Não usa LangChain, LlamaIndex nem banco vetorial dedicado. Cada peça é um módulo pequeno, que dá
para ler de uma vez só.

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes e ambientes)
- [Ollama](https://ollama.com/download)

## Instalação

### 1. uv

O `requirements.txt` serve só para instalar o uv. As dependências do projeto ficam no
`pyproject.toml`.

```powershell
python -m pip install -r requirements.txt
```

Alternativas: `winget install astral-sh.uv` (Windows) ou o instalador oficial
(`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`).

> **`uv` não reconhecido?** O pip instala o executável na pasta `Scripts` do seu Python
> (ex.: `C:\Users\<você>\AppData\Local\Programs\Python\Python312\Scripts`), que nem sempre está no
> PATH. Adicione essa pasta ao PATH ou use `python -m uv ...` com esse mesmo Python.
> Não instale o uv **dentro** da `.venv` do projeto: o `uv sync` remove o que não está no
> `pyproject.toml`.

### 2. Dependências do projeto

```powershell
uv sync          # cria/atualiza .venv com as dependências e as de desenvolvimento (pytest)
uv run rag --help
```

### 3. Ollama e modelos

Instale o Ollama e confira se o servidor está no ar. No Windows e no macOS o app já sobe o
servidor; no Linux, rode `ollama serve`.

```powershell
ollama --version
curl http://localhost:11434/api/tags      # deve listar os modelos instalados

ollama pull nomic-embed-text              # embeddings, 768 dimensões (obrigatório)
ollama pull qwen2.5:7b                    # geração (veja a seção "Modelo de geração")
```

## Onde colocar os documentos

Coloque arquivos `.md`, `.txt`, `.pdf` e `.docx` na pasta **`docs/`**, na raiz do projeto.
Subpastas são lidas recursivamente, e a fonte é gravada como caminho relativo
(ex.: `contratos/2026/aditivo.pdf`). Arquivos ocultos e temporários do Office (`~$...`) são
ignorados.

Se `docs/` não existir ou estiver vazia na primeira ingestão, o `rag ingest` cria três arquivos
de exemplo para você testar na hora. Para usar outra pasta, mude `docs_dir` no `config.toml` ou
defina `RAG_DOCS_DIR`.

> PDFs escaneados (só imagem) não têm texto extraível. O `ingest` avisa, mas não faz OCR.

## Comandos

```powershell
uv run rag ingest                         # indexa docs/ (incremental)
uv run rag stats                          # arquivos, chunks e parâmetros do índice
uv run rag ask "Qual o valor da taxa de inscrição?"
uv run rag ask "..." --verbose            # + texto dos chunks e rank de cada busca
uv run rag ask "..." -k 6 --mode vector   # top k e tipo de busca (hybrid | vector | bm25)
uv run rag eval                           # hit rate e MRR dos três modos de busca
uv run rag eval -k 8 -v                   # outro k, com o rank por pergunta
uv run rag eval --mode hybrid -f outro.yaml
```

Com a `.venv` ativada (`.venv\Scripts\activate`), dá para chamar só `rag ...`.

### `rag ingest`

- Calcula o SHA-256 de cada arquivo e só reprocessa os novos ou alterados.
- Remove do índice os arquivos que saíram de `docs/`.
- Se você mudar `embed_model`, `embed_dim`, `embed_doc_prefix`, `chunk_size` ou `chunk_overlap`,
  o índice antigo fica incompatível e o `ingest` reindexa tudo automaticamente (e avisa).
  O `ask` e o `eval` se recusam a rodar até você reindexar.
- Um arquivo corrompido não derruba a ingestão: ele aparece como falha e é tentado de novo na
  próxima execução.

### `rag ask`

Mostra a resposta (em streaming) e uma tabela com os chunks usados e o score de cada um. Com
`--verbose`, mostra também o texto de cada chunk e, no modo híbrido, a posição e o score que o
chunk teve na busca vetorial e no BM25. Isso ajuda a entender por que um trecho entrou no
contexto.

Se não houver resposta no contexto, o modelo é instruído a responder exatamente
`Não encontrei nos documentos.`

## Configuração

Ordem de prioridade: **padrões no código < `config.toml` < variáveis de ambiente `RAG_*`**
(que também podem vir de um arquivo `.env`; veja `.env.example`).

| Chave | Padrão | O que é |
|---|---|---|
| `docs_dir` | `docs` | pasta dos documentos |
| `db_path` | `rag.duckdb` | arquivo do banco (fora de `docs/`, ignorado pelo git) |
| `eval_file` | `eval/perguntas.yaml` | perguntas do `rag eval` |
| `ollama_url` | `http://localhost:11434` | servidor Ollama |
| `embed_model` / `embed_dim` | `nomic-embed-text` / `768` | modelo de embedding e dimensão |
| `embed_doc_prefix` / `embed_query_prefix` | `search_document: ` / `search_query: ` | prefixos de tarefa do nomic |
| `gen_model` | `llama3.1` (o `config.toml` usa `qwen2.5-coder:7b`) | modelo de geração |
| `gen_temperature` / `gen_num_ctx` | `0.1` / `8192` | parâmetros da geração |
| `chunk_size` / `chunk_overlap` | `800` / `100` | chunking, em caracteres |
| `top_k` | `4` | chunks no contexto |
| `search_mode` | `hybrid` | `hybrid`, `vector` ou `bm25` |
| `candidates` | `20` | candidatos de cada busca antes da fusão |
| `rrf_k` | `60` | constante do RRF |

Exemplo, para trocar o modelo de geração só nesta sessão do PowerShell:

```powershell
$env:RAG_GEN_MODEL = "qwen3:8b"; uv run rag ask "..."
```

### Modelo de geração

O `config.toml` vem com `qwen2.5-coder:7b`, que serve **só para validar o pipeline**. Ele é
especializado em código. Para uso real, prefira um **modelo de instrução geral**, que segue
melhor as regras do prompt e escreve um português mais natural:

- **`qwen3:8b`**: o mais indicado. Ele "pensa" antes de responder: fica mais lento, mas o
  raciocínio não aparece na saída, porque o código exibe só `message.content`.
- **`qwen2.5:7b`**: mais rápido, bom em português.

```powershell
ollama pull qwen3:8b
# e em config.toml: gen_model = "qwen3:8b"
```

Trocar o modelo de geração **não** exige reindexar. Só o modelo de embedding exige.

## Avaliação (`rag eval`)

O `eval/perguntas.yaml` traz perguntas com gabarito. Um chunk recuperado conta como acerto quando
atende **a todos** os critérios informados:

```yaml
perguntas:
  - pergunta: Quanto custa a inscrição?
    fonte: edital-dataprev_supe-versaofinal.pdf   # caminho relativo em docs/
    pagina: 5                                     # opcional (PDF)
    trecho: R$ 110,00                             # opcional; comparado sem acento/caixa
```

- **Hit rate@k**: fração das perguntas em que algum chunk relevante veio no top k.
- **MRR@k**: média de `1/posição` do primeiro chunk relevante (0 quando não vem). Ela premia
  trazer o trecho certo em primeiro lugar.

Sem `--mode`, o comando compara os três modos lado a lado. Ele também avisa quando o gabarito de
uma pergunta não bate com **nenhum** chunk do índice (fonte errada ou trecho partido entre dois
chunks). Sem esse aviso, um erro no gabarito apareceria como métrica baixa.

**Como comparar ajustes:** mude um parâmetro por vez, rode `rag ingest` se for de indexação e
depois `rag eval -v`. Exemplos:

```powershell
$env:RAG_CHUNK_SIZE = "500"; $env:RAG_CHUNK_OVERLAP = "80"; uv run rag ingest; uv run rag eval -v
$env:RAG_CANDIDATES = "50"; uv run rag eval      # não exige reindexar
Remove-Item Env:RAG_CHUNK_SIZE, Env:RAG_CHUNK_OVERLAP, Env:RAG_CANDIDATES; uv run rag ingest
```

Resultado inicial (16 perguntas, chunks de 800/100, `nomic-embed-text`):

| Modo | Hit rate@4 | MRR@4 | Hit rate@8 | MRR@8 |
|---|---|---|---|---|
| hybrid | 56,2% | 0,448 | 75,0% | 0,477 |
| vector | 43,8% | 0,406 | 50,0% | 0,419 |
| bm25 | 68,8% | 0,516 | 75,0% | 0,523 |

Neste corpus, um edital cheio de termos exatos, o BM25 sozinho supera a busca vetorial, e a
fusão fica no meio. Isso é ponto de partida para ajustes, não conclusão.

## Testes

```powershell
uv run pytest -q
```

Os testes cobrem chunking, ingestão incremental, busca vetorial, BM25 (acentos, stemming e
stopwords), RRF, prompt e métricas. Eles **não precisam do Ollama**: um `FakeEmbedder`
(bag-of-words com hashing) substitui os embeddings, e o cliente HTTP é testado com
`httpx.MockTransport`. A primeira execução precisa de internet para o DuckDB baixar a extensão
`fts`, se ela ainda não estiver no cache.

## Estrutura

```
├── config.toml              # configuração
├── docs/                    # seus documentos
├── eval/perguntas.yaml      # conjunto de avaliação
├── src/rag_local/
│   ├── config.py            # Settings tipado: padrões < config.toml < RAG_*
│   ├── loaders.py           # md/txt/pdf/docx -> seções (texto + página)
│   ├── chunking.py          # janela de caracteres com quebra em parágrafo/frase
│   ├── ollama.py            # cliente HTTP mínimo com erros acionáveis
│   ├── embeddings.py        # /api/embed em lotes, com prefixos de tarefa
│   ├── store.py             # DuckDB: schema, transações, busca vetorial e FTS
│   ├── ingest.py            # hash -> diff -> carregar -> chunk -> embed -> gravar
│   ├── retrieval.py         # vetorial, BM25 e RRF
│   ├── generation.py        # prompt e /api/chat em streaming
│   ├── evaluation.py        # hit rate e MRR
│   ├── samples.py           # documentos de exemplo
│   └── cli.py               # typer: ingest, ask, stats, eval
└── tests/
```

## Schema no DuckDB

```sql
files  (source VARCHAR PRIMARY KEY, sha256 VARCHAR, size_bytes BIGINT, n_chunks INTEGER, ingested_at TIMESTAMP)
chunks (id BIGINT PRIMARY KEY DEFAULT nextval('chunks_id_seq'), source VARCHAR, page INTEGER,
        chunk_index INTEGER, text VARCHAR, embedding FLOAT[768])
meta   (key VARCHAR PRIMARY KEY, value VARCHAR)   -- parâmetros com que o índice foi gerado
stopwords_pt (sw VARCHAR)                         -- stopwords do índice FTS
fts_main_chunks.*                                 -- criado pela extensão fts
```

Dá para inspecionar o banco direto: `uv run python -c "import duckdb; print(duckdb.connect('rag.duckdb').sql('from files'))"`.

## Decisões de arquitetura

- **Sem frameworks.** O pipeline inteiro cabe em ~1.300 linhas, CLI incluída. O objetivo é entender cada etapa,
  não esconder as etapas atrás de abstrações.
- **DuckDB como banco vetorial.** Arrays `FLOAT[768]` com `array_cosine_similarity` fazem busca
  exata, uma varredura completa. Para milhares de chunks isso leva milissegundos, e o resultado
  é exato (sem a aproximação de índices HNSW). Se o corpus crescer muito, a extensão `vss` do
  DuckDB adiciona HNSW sem trocar de banco.
- **Chunking por caracteres, nunca atravessando páginas.** Cada página do PDF é chunkada
  separadamente, então a citação `[arquivo.pdf, p. N]` é sempre exata. O corte procura, na
  segunda metade da janela, um fim de parágrafo, depois um fim de frase, depois uma quebra de
  linha e por fim um espaço. Isso evita chunks minúsculos. O overlap recomeça no início de uma
  palavra.
- **Prefixos `search_document:` / `search_query:`.** O `nomic-embed-text` foi treinado com eles;
  sem os prefixos, o retrieval piora. Eles são configuráveis, para outros modelos.
- **Busca híbrida com RRF.** O cosseno vai de 0 a 1 e o BM25 não tem teto, então os scores não
  são comparáveis. O RRF usa só as posições (`Σ 1/(60 + rank)`) e dispensa normalização. Cada
  busca traz `candidates` itens antes da fusão.
- **BM25 em português.** O índice usa stemmer `portuguese`, remoção de acentos, uma lista própria
  de stopwords (sem "não", que muda o sentido) e preserva dígitos. O padrão do DuckDB descartaria
  números como "001/2026" e "R$ 110".
- **O índice FTS é recriado a cada ingestão que muda algo.** O DuckDB não atualiza esse índice
  sozinho quando a tabela muda.
- **Sem FOREIGN KEY entre `chunks` e `files`.** No DuckDB, FK atrapalha apagar e reinserir. A
  consistência vem de uma transação por arquivo (apaga chunks, insere novos, atualiza `files`).
- **Tabela `meta`.** Ela guarda modelo, dimensão e parâmetros de chunking do índice. Isso impede
  misturar vetores de modelos diferentes ou consultar com um índice inconsistente.
- **Respostas em streaming, com `num_ctx` explícito.** O contexto padrão do Ollama pode truncar o
  prompt em silêncio. Sem nenhum chunk recuperado, o modelo nem é chamado.
- **Fora do escopo, de propósito:** OCR, reranking com cross-encoder, histórico de conversa e
  busca com filtros por metadados. São próximos passos naturais.

"""Documentos de exemplo, criados quando a pasta de documentos não existe ou está vazia."""

SAMPLE_DOCS: dict[str, str] = {
    "exemplo-rag.md": """# O que é RAG

RAG (Retrieval-Augmented Generation) é uma técnica em que um modelo de linguagem responde
perguntas usando trechos recuperados de uma base de documentos, em vez de depender só do
que aprendeu no treinamento.

## Etapas

O pipeline tem duas fases. Na ingestão, os documentos são divididos em chunks e cada chunk
vira um embedding. Na consulta, a pergunta também vira um embedding, os chunks mais
parecidos são recuperados e entram no prompt como contexto.

## Busca híbrida

A busca vetorial encontra trechos com significado parecido, mesmo com palavras diferentes.
O BM25 encontra trechos com as mesmas palavras-chave. O Reciprocal Rank Fusion combina as
duas listas somando 1/(60 + posição) de cada documento em cada ranking.
""",
    "exemplo-duckdb.txt": """DuckDB é um banco de dados analítico embutido, que roda dentro do próprio processo,
sem servidor. O banco inteiro fica em um único arquivo.

Para busca vetorial, o DuckDB oferece arrays de tamanho fixo, como FLOAT[768], e a função
array_cosine_similarity, que calcula a similaridade de cosseno entre dois vetores.

A extensão fts adiciona busca textual completa com ranking BM25. O índice é criado com
PRAGMA create_fts_index e consultado com a função match_bm25. O índice não é atualizado
automaticamente quando a tabela muda, então precisa ser recriado.
""",
    "exemplo-ollama.md": """# Ollama

O Ollama roda modelos de linguagem localmente e expõe uma API HTTP na porta 11434.

Para baixar um modelo, use `ollama pull nomic-embed-text`. O nomic-embed-text gera
embeddings de 768 dimensões e espera os prefixos "search_document:" para documentos e
"search_query:" para perguntas.

Para geração de texto, modelos de instrução como qwen2.5:7b ou llama3.1 funcionam bem em
português. A rota /api/embed gera embeddings e a rota /api/chat gera respostas.
""",
}

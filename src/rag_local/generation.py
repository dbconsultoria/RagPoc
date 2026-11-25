"""Geração de resposta via Ollama (/api/chat), restrita ao contexto recuperado."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from rag_local.config import Settings
from rag_local.ollama import OllamaClient
from rag_local.retrieval import SearchHit

NOT_FOUND = "Não encontrei nos documentos."

SYSTEM_PROMPT = f"""Você é um assistente que responde perguntas usando EXCLUSIVAMENTE os trechos de \
documentos fornecidos no contexto.

Regras obrigatórias:
1. Use somente informações presentes no contexto. Não use conhecimento próprio nem suposições.
2. Depois de cada afirmação, cite a fonte entre colchetes exatamente como aparece no cabeçalho \
do trecho, por exemplo [edital.pdf, p. 3] ou [notas.md].
3. Se o contexto não contiver a resposta, responda apenas: "{NOT_FOUND}"
4. Responda em português do Brasil, de forma direta e objetiva."""


def build_context(hits: Sequence[SearchHit]) -> str:
    return "\n\n---\n\n".join(f"[{hit.citation}]\n{hit.chunk.text}" for hit in hits)


def build_messages(question: str, hits: Sequence[SearchHit]) -> list[dict[str, str]]:
    user = (
        f"Contexto:\n\n{build_context(hits)}\n\n"
        f"Pergunta: {question}\n\n"
        f"Responda usando só o contexto acima e cite as fontes entre colchetes. "
        f'Se a resposta não estiver no contexto, responda apenas "{NOT_FOUND}"'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class Generator:
    def __init__(self, client: OllamaClient, model: str, temperature: float = 0.1, num_ctx: int = 8192) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx

    @classmethod
    def from_settings(cls, settings: Settings) -> Generator:
        return cls(
            OllamaClient(settings.ollama_url),
            model=settings.gen_model,
            temperature=settings.gen_temperature,
            num_ctx=settings.gen_num_ctx,
        )

    def stream_answer(self, question: str, hits: Sequence[SearchHit]) -> Iterator[str]:
        if not hits:
            yield NOT_FOUND  # nada recuperado: nem chama o modelo
            return
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(question, hits),
            "stream": True,
            # num_ctx explícito: o padrão do Ollama pode truncar o contexto em silêncio.
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        # Modelos com "thinking" (ex.: qwen3) mandam o raciocínio em message.thinking;
        # só message.content é exibido.
        for event in self.client.stream("/api/chat", payload):
            piece = event.get("message", {}).get("content", "")
            if piece:
                yield piece

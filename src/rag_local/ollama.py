"""Cliente HTTP mínimo para a API do Ollama, com mensagens de erro acionáveis."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 300.0, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(self.base_url + path, json=payload)
        except httpx.ConnectError as exc:
            raise self._connection_error() from exc
        self._raise_for_status(response, payload)
        return response.json()

    def stream(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            with self._client.stream("POST", self.base_url + path, json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    self._raise_for_status(response, payload)
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
        except httpx.ConnectError as exc:
            raise self._connection_error() from exc

    def _connection_error(self) -> OllamaError:
        return OllamaError(
            f"Não consegui conectar ao Ollama em {self.base_url}. Ele está rodando? (ollama serve)"
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response, payload: dict[str, Any]) -> None:
        if response.status_code == 404:
            model = payload.get("model", "?")
            raise OllamaError(f"Modelo '{model}' não encontrado no Ollama. Rode: ollama pull {model}")
        if response.status_code >= 400:
            raise OllamaError(f"Ollama respondeu {response.status_code}: {response.text}")

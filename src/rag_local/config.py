"""Configuração: padrões < config.toml < variáveis de ambiente (RAG_*, também via .env)."""

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

CONFIG_FILE = "config.toml"
ENV_PREFIX = "RAG_"
SEARCH_MODES = ("hybrid", "vector", "bm25")


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = Path("docs")
    db_path: Path = Path("rag.duckdb")
    eval_file: Path = Path("eval/perguntas.yaml")

    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    gen_model: str = "llama3.1"
    gen_temperature: float = 0.1
    gen_num_ctx: int = 8192

    chunk_size: int = 800
    chunk_overlap: int = 100

    top_k: int = 8
    search_mode: str = "hybrid"
    candidates: int = 20
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.search_mode not in SEARCH_MODES:
            raise ValueError(f"search_mode inválido: {self.search_mode!r} (use {', '.join(SEARCH_MODES)})")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap precisa ser >= 0 e menor que chunk_size")


def find_project_root(start: Path | None = None) -> Path:
    """Sobe a partir do diretório atual até achar config.toml ou pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / CONFIG_FILE).exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return current


def _convert(name: str, field_type: Any, value: Any) -> Any:
    try:
        if field_type is Path:
            return Path(value)
        if field_type is int:
            return int(value)
        if field_type is float:
            return float(value)
        return str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valor inválido para {name}: {value!r}") from exc


def load_settings(root: Path | None = None) -> Settings:
    root = root or find_project_root()
    load_dotenv(root / ".env")
    types = {f.name: f.type for f in fields(Settings)}

    raw: dict[str, Any] = {}
    config_path = root / CONFIG_FILE
    if config_path.exists():
        with config_path.open("rb") as fh:
            data = tomllib.load(fh)
        # As seções do TOML servem só para organizar a leitura; as chaves são achatadas.
        for key, value in data.items():
            if isinstance(value, dict):
                raw.update(value)
            else:
                raw[key] = value

    unknown = set(raw) - set(types)
    if unknown:
        raise ValueError(f"Chaves desconhecidas em {CONFIG_FILE}: {', '.join(sorted(unknown))}")

    for name in types:
        env_value = os.environ.get(ENV_PREFIX + name.upper())
        if env_value is not None:
            raw[name] = env_value

    values = {name: _convert(name, types[name], value) for name, value in raw.items()}
    settings = Settings(**values)

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else (root / path).resolve()

    return replace(
        settings,
        docs_dir=absolute(settings.docs_dir),
        db_path=absolute(settings.db_path),
        eval_file=absolute(settings.eval_file),
    )

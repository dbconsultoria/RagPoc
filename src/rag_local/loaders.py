"""Leitura de arquivos: cada arquivo vira uma lista de seções (texto + página opcional)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = frozenset({".md", ".txt", ".pdf", ".docx"})


@dataclass(frozen=True)
class Section:
    text: str
    page: int | None = None  # 1-based; só PDF tem página


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    # Junta palavras hifenizadas na quebra de linha (comum em PDF): "inscri-\nção" -> "inscrição".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: Path) -> list[Section]:
    reader = PdfReader(path)
    return [
        Section(clean_text(page.extract_text() or ""), page=number)
        for number, page in enumerate(reader.pages, start=1)
    ]


def _load_docx(path: Path) -> list[Section]:
    document = Document(str(path))
    blocks = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return [Section(clean_text("\n\n".join(blocks)))]


def load_file(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return [Section(clean_text(path.read_text(encoding="utf-8", errors="replace")))]
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    raise ValueError(f"Extensão não suportada: {path.suffix}")


def iter_documents(docs_dir: Path) -> Iterator[Path]:
    """Arquivos suportados, em ordem estável. Ignora ocultos e temporários do Office (~$)."""
    if not docs_dir.exists():
        return
    for path in sorted(docs_dir.rglob("*")):
        relative_parts = path.relative_to(docs_dir).parts
        if any(part.startswith(".") for part in relative_parts) or path.name.startswith("~$"):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def relative_source(path: Path, docs_dir: Path) -> str:
    return path.relative_to(docs_dir).as_posix()

"""Локальный индекс текстовой библиотеки для контекстных ответов Ollama."""

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIBRARY_PATH = ROOT / 'data' / 'classics_library.json'
TOKEN_RE = re.compile(r'[a-zа-яё0-9]+', re.IGNORECASE)


def _tokens(text):
    return TOKEN_RE.findall((text or '').lower().replace('ё', 'е'))


def _read_document(path):
    if path.suffix.lower() != '.pdf':
        return path.read_text(encoding='utf-8')
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            'Для PDF установите зависимость: .venv/bin/python -m pip install -r requirements.txt'
        ) from error
    reader = PdfReader(str(path))
    pages = [page.extract_text() or '' for page in reader.pages]
    return '\n'.join(pages)


def build_library(source, output=DEFAULT_LIBRARY_PATH, chunk_words=220, overlap=40):
    """Build a compact JSON index from user-provided TXT, Markdown, or PDF files."""
    source = Path(source)
    if source.is_file():
        files = [source] if source.suffix.lower() in ('.txt', '.md', '.pdf') else []
    else:
        files = sorted(path for path in source.rglob('*')
                       if path.suffix.lower() in ('.txt', '.md', '.pdf'))
    chunks = []
    for path in files:
        words = _read_document(path).split()
        step = max(1, chunk_words - overlap)
        for start in range(0, len(words), step):
            text = ' '.join(words[start:start + chunk_words]).strip()
            if text:
                chunks.append({'title': path.stem, 'text': text})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'version': 1, 'chunks': chunks},
                                 ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return len(chunks)


class LibraryIndex:
    def __init__(self, path=DEFAULT_LIBRARY_PATH):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding='utf-8'))
        self.chunks = data.get('chunks', [])

    def search(self, query, limit=3):
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return []
        ranked = []
        for chunk in self.chunks:
            words = Counter(_tokens(chunk.get('text', '')))
            score = sum(words[token] for token in query_tokens)
            if score:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked[:limit]]

    def context(self, query, limit=3):
        matches = self.search(query, limit=limit)
        if not matches:
            return ''
        fragments = '\n\n'.join(
            '[' + item.get('title', 'Книга') + ']\n' + item.get('text', '')
            for item in matches
        )
        return (
            'Релевантные фрагменты из локальной библиотеки. Используй их только '
            'если они действительно относятся к вопросу и не выдумывай продолжение:\n' + fragments
        )
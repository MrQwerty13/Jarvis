"""Локальная JSON-память результатов обработки Jarvis."""

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMORY_PATH = ROOT / 'data' / 'jarvis_memory.json'


def _normalize(text):
    return ' '.join((text or '').lower().replace('ё', 'е').split())


class JsonMemory:
    """Stores experiences and safely reuses exact dialog results."""

    def __init__(self, path=None):
        self.path = Path(path or DEFAULT_MEMORY_PATH)
        self._lock = threading.RLock()

    def _read(self):
        if not self.path.exists():
            return {'version': 1, 'entries': {}}
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return {'version': 1, 'entries': {}}
        if not isinstance(data, dict) or not isinstance(data.get('entries'), dict):
            return {'version': 1, 'entries': {}}
        return data

    def remember(self, text, result, backend='unknown'):
        key = _normalize(text)
        if not key:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            data = self._read()
            entry = data['entries'].get(key, {})
            entry.update({
                'text': key,
                'backend': backend,
                'tag': result.get('tag'),
                'reply': result.get('reply', ''),
                'done': bool(result.get('done', False)),
                'last_seen': now,
            })
            entry['uses'] = int(entry.get('uses', 0)) + 1
            data['entries'][key] = entry
            self._write(data)

    def recall(self, text, backend):
        key = _normalize(text)
        with self._lock:
            entry = self._read()['entries'].get(key)
        if not entry or entry.get('backend') != backend:
            return None
        return dict(entry)

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.jarvis-memory-', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
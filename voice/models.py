"""Загрузка локальной модели Vosk для офлайн-распознавания речи."""

import zipfile
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models'

# Маленькие офлайн-модели: https://alphacephei.com/vosk/models
MODELS = {
    'ru': {
        'name': 'vosk-model-small-ru-0.22',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip',
    },
    'en': {
        'name': 'vosk-model-small-en-us-0.15',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
    },
}


def model_path(language='ru'):
    if language not in MODELS:
        raise ValueError('Поддерживаемые языки: ' + ', '.join(sorted(MODELS)))
    return MODELS_DIR / MODELS[language]['name']


def _looks_ready(path):
    return (path / 'am' / 'final.mdl').exists() or (path / 'conf' / 'model.conf').exists()


def ensure_model(language='ru'):
    path = model_path(language)
    if path.exists() and _looks_ready(path):
        return path

    info = MODELS[language]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archive = MODELS_DIR / (info['name'] + '.zip')
    print('Загрузка модели ' + info['name'] + ' (один раз, ~40–50 МБ)…', flush=True)
    with urlopen(info['url'], timeout=120) as response:
        content = response.read()
    archive.write_bytes(content)
    print('Распаковка…', flush=True)
    with zipfile.ZipFile(archive, 'r') as zipped:
        zipped.extractall(MODELS_DIR)
    archive.unlink(missing_ok=True)
    if not path.exists() or not _looks_ready(path):
        raise FileNotFoundError('После распаковки не найдена модель: ' + str(path))
    return path

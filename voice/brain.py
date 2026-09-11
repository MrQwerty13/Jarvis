"""Мини-нейросеть диалога: bag-of-words → MLP → категория ответа."""

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from numai.model import MLP

from .dialogue_data import intents_for


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / 'models'
TOKEN_RE = re.compile(r'[a-zа-яё0-9]+', re.IGNORECASE)


def default_model_path(language='ru'):
    return DEFAULT_MODEL_DIR / f'talk-{language}.npz'


def tokenize(text):
    return TOKEN_RE.findall(text.lower().replace('ё', 'е'))


def build_vocab(intents):
    words = sorted({token for intent in intents for pattern in intent['patterns'] for token in tokenize(pattern)})
    return {word: index for index, word in enumerate(words)}


def bag_of_words(tokens, vocab):
    vector = np.zeros(len(vocab), dtype=np.float32)
    for token in tokens:
        index = vocab.get(token)
        if index is not None:
            vector[index] = 1.0
    return vector


def _augment(pattern, language='ru'):
    """Небольшие вариации фраз, чтобы сеть была устойчивее к речи."""
    if language == 'en':
        variants = {pattern, 'please ' + pattern, pattern + ' please', pattern + ' jarvis'}
    else:
        variants = {pattern, pattern + ' пожалуйста', 'слушай ' + pattern, pattern + ' джарвис'}
    return sorted(variants)


class TalkBrain:
    def __init__(self, language='ru', model=None, vocab=None, labels=None, intents=None, fallback=None):
        self.language = language
        self.model = model
        self.vocab = vocab or {}
        self.labels = labels or []
        self.intents, self.fallback = intents_for(language) if intents is None else (intents, fallback)
        self._by_name = {intent['name']: intent for intent in self.intents}
        self.rng = np.random.default_rng()

    @property
    def ready(self):
        return self.model is not None and bool(self.vocab) and bool(self.labels)

    def encode(self, text):
        return bag_of_words(tokenize(text), self.vocab)[None, :]

    def train(self, epochs=400, lr=0.01, seed=42):
        self.vocab = build_vocab(self.intents)
        self.labels = [intent['name'] for intent in self.intents]
        label_index = {name: i for i, name in enumerate(self.labels)}
        samples = []
        targets = []
        for intent in self.intents:
            for pattern in intent['patterns']:
                for phrase in _augment(pattern, self.language):
                    samples.append(bag_of_words(tokenize(phrase), self.vocab))
                    targets.append(label_index[intent['name']])
        x = np.stack(samples).astype(np.float32)
        y = np.array(targets, dtype=np.int64)
        hidden = max(16, min(64, len(self.vocab) * 2))
        self.model = MLP(sizes=(len(self.vocab), hidden, len(self.labels)), seed=seed)
        order = np.arange(len(x))
        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            rng.shuffle(order)
            losses = []
            for index in order:
                losses.append(self.model.train_batch(x[index:index + 1], y[index:index + 1], lr=lr))
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f'  эпоха {epoch + 1}/{epochs}, loss={np.mean(losses):.4f}', flush=True)
        return self

    def save(self, path=None):
        path = Path(path or default_model_path(self.language))
        if not self.ready:
            raise RuntimeError('Нечего сохранять: сначала обучите модель.')
        metadata = {
            'language': self.language,
            'vocab': self.vocab,
            'labels': self.labels,
        }
        self.model.save(path, metadata=metadata)
        side = path.with_suffix('.json')
        side.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        return path

    @classmethod
    def load(cls, path=None, language='ru'):
        path = Path(path or default_model_path(language))
        meta_path = path.with_suffix('.json')
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError('Нет обученной talk-модели: ' + str(path))
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        model = MLP.load(path)
        brain = cls(
            language=metadata.get('language', language),
            model=model,
            vocab=metadata['vocab'],
            labels=metadata['labels'],
        )
        return brain

    def predict(self, text, threshold=0.45):
        if not self.ready:
            raise RuntimeError('Модель не загружена.')
        tokens = tokenize(text)
        if not tokens:
            return None, 0.0
        features = self.encode(text)
        if float(features.sum()) == 0.0:
            return None, 0.0
        probs = self.model.predict_proba(features)[0]
        index = int(probs.argmax())
        confidence = float(probs[index])
        if confidence < threshold:
            return None, confidence
        return self.labels[index], confidence

    def answer(self, text, threshold=0.45):
        label, confidence = self.predict(text, threshold=threshold)
        if label is None:
            reply = str(self.rng.choice(self.fallback))
            return reply, None, confidence
        intent = self._by_name[label]
        reply = str(self.rng.choice(intent['responses']))
        if reply == '__TIME__':
            now = datetime.now().strftime('%H:%M')
            if self.language == 'en':
                reply = f'The time is {now}.'
            else:
                reply = f'Сейчас {now}.'
        return reply, label, confidence


def ensure_brain(language='ru', model_path=None, force_train=False):
    path = Path(model_path or default_model_path(language))
    side = path.with_suffix('.json')
    if not force_train and path.exists() and side.exists():
        metadata = json.loads(side.read_text(encoding='utf-8'))
        if metadata.get('language', language) == language:
            return TalkBrain.load(path, language=language)
    print('Обучение мини-сети диалога (' + language + ')…', flush=True)
    brain = TalkBrain(language=language).train()
    brain.save(path)
    print('Сохранено: ' + str(path), flush=True)
    return brain

"""MLP, backpropagation и Adam, без библиотек машинного обучения."""

import json
from pathlib import Path

import numpy as np


class MLP:
    def __init__(self, sizes=(784, 128, 64, 10), seed=42):
        self.sizes = tuple(int(n) for n in sizes)
        if len(self.sizes) < 2 or any(n < 1 for n in self.sizes):
            raise ValueError('Некорректные размеры сети.')
        rng = np.random.default_rng(seed)
        self.params = {}
        for i, (inputs, outputs) in enumerate(zip(self.sizes, self.sizes[1:])):
            self.params[f'W{i}'] = (rng.standard_normal((inputs, outputs)) * np.sqrt(2 / inputs)).astype(np.float32)
            self.params[f'b{i}'] = np.zeros(outputs, dtype=np.float32)
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0

    def _forward(self, x):
        activations = [x]
        for i in range(len(self.sizes) - 1):
            z = activations[-1] @ self.params[f'W{i}'] + self.params[f'b{i}']
            if i < len(self.sizes) - 2:
                activations.append(np.maximum(z, 0))
            else:
                # log-softmax остаётся устойчивым даже при больших логитах.
                shifted = z - z.max(axis=1, keepdims=True)
                log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
                activations.append(np.exp(log_probs))
        return activations, log_probs

    def predict_proba(self, x):
        return self._forward(x)[0][-1]

    def loss_and_gradients(self, x, y):
        activations, log_probs = self._forward(x)
        loss = -log_probs[np.arange(len(y)), y].mean()
        delta = activations[-1].copy()
        delta[np.arange(len(y)), y] -= 1
        delta /= len(y)
        gradients = {}
        for i in reversed(range(len(self.sizes) - 1)):
            gradients[f'W{i}'] = activations[i].T @ delta
            gradients[f'b{i}'] = delta.sum(axis=0)
            if i:
                delta = (delta @ self.params[f'W{i}'].T) * (activations[i] > 0)
        return float(loss), gradients

    def train_batch(self, x, y, lr=0.001):
        loss, gradients = self.loss_and_gradients(x, y)
        self.step += 1
        for name, gradient in gradients.items():
            self.m[name] = 0.9 * self.m[name] + 0.1 * gradient
            self.v[name] = 0.999 * self.v[name] + 0.001 * gradient ** 2
            m_hat = self.m[name] / (1 - 0.9 ** self.step)
            v_hat = self.v[name] / (1 - 0.999 ** self.step)
            self.params[name] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        return loss

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('wb') as output:
            np.savez_compressed(output, sizes=np.array(self.sizes),
                                metadata=json.dumps(metadata or {}), **self.params)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as checkpoint:
            sizes = checkpoint['sizes']
            if sizes.ndim != 1 or not 2 <= len(sizes) <= 8 or np.any(sizes > 4096):
                raise ValueError('Некорректная архитектура в файле модели.')
            model = cls(sizes)
            for name, initial in model.params.items():
                value = checkpoint[name]
                if value.shape != initial.shape or not np.isfinite(value).all():
                    raise ValueError('Повреждённые веса модели: ' + name)
                model.params[name] = value.astype(np.float32)
        return model

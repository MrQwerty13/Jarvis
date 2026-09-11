"""Воспроизводимое обучение со случайных весов и отдельная оценка качества."""

import json
import time
from pathlib import Path

import numpy as np

from .data import download, load_split
from .model import MLP
from .vision import augment


def evaluate(model, images, labels):
    predictions = []
    for start in range(0, len(images), 512):
        batch = images[start:start+512].reshape(-1, 784)
        predictions.append(model.predict_proba(batch).argmax(axis=1))
    predicted = np.concatenate(predictions)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    totals = confusion.sum(axis=1)
    return {
        'accuracy': float(np.mean(predicted == labels)),
        'samples': len(labels),
        'per_digit_accuracy': {
            str(digit): float(confusion[digit, digit] / totals[digit]) if totals[digit] else None
            for digit in range(10)
        },
        'confusion_matrix': confusion.tolist(),
    }


def train(args):
    if args.model.exists() and not args.overwrite:
        raise ValueError(f'Модель уже существует: {args.model}. Для нового обучения укажите --overwrite или другой --model.')
    download(args.data)
    print('Подготовка изображений…', flush=True)
    images, labels = load_split(args.data, train=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(images))
    validation_indices, training_indices = order[:5000], order[5000:]
    if args.limit:
        training_indices = training_indices[:args.limit]
    training_images, training_labels = images[training_indices], labels[training_indices]
    validation_images, validation_labels = images[validation_indices], labels[validation_indices]
    model = MLP(seed=args.seed)
    initial = evaluate(model, validation_images, validation_labels)['accuracy']
    print(f'До обучения: {initial:.2%}. Обучение: {len(training_indices)}; валидация: 5000.', flush=True)
    best = -1
    best_epoch = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs+1):
        permutation = rng.permutation(len(training_images))
        total_loss = 0
        lr = args.lr * (0.5 if epoch > args.epochs * 0.7 else 1)
        for offset in range(0, len(permutation), args.batch_size):
            indices = permutation[offset:offset+args.batch_size]
            batch = augment(training_images[indices], rng)
            loss = model.train_batch(batch.reshape(-1, 784), training_labels[indices], lr=lr)
            if not np.isfinite(loss):
                raise ValueError('Обучение стало неустойчивым. Попробуйте меньший --lr.')
            total_loss += loss * len(indices)
        accuracy = evaluate(model, validation_images, validation_labels)['accuracy']
        row = {'epoch': epoch, 'loss': total_loss/len(training_images), 'validation_accuracy': accuracy}
        history.append(row)
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            model.save(args.model, {'alphabet': 'arabic-0-9', 'seed': args.seed,
                                   'epoch': epoch, 'validation_accuracy': best,
                                   'preprocessing': 'center-mass-20-in-28-v1'})
        print(f'Эпоха {epoch:02}/{args.epochs}: ошибка={row["loss"]:.4f}, валидация={accuracy:.2%}', flush=True)
    # Тестовая выборка открывается после выбора лучших весов по валидации.
    test_images, test_labels = load_split(args.data, train=False)
    test_metrics = evaluate(MLP.load(args.model), test_images, test_labels)
    report = {
        'architecture': list(model.sizes), 'seed': args.seed,
        'trained_from_random_weights': True, 'dataset': 'MNIST',
        'training_samples': len(training_images), 'validation_samples': 5000,
        'initial_validation_accuracy': initial, 'best_epoch': best_epoch,
        'best_validation_accuracy': best, 'epochs': history,
        'test': test_metrics, 'elapsed_seconds': time.monotonic()-started,
        'note': 'MNIST accuracy does not measure live camera accuracy.',
    }
    report_path = args.model.with_suffix('.metrics.json')
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Тест MNIST: {test_metrics["accuracy"]:.2%} на {test_metrics["samples"]} изображениях.', flush=True)
    print(f'Веса: {args.model}\nОтчёт: {report_path}', flush=True)

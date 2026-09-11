"""Reproducible comparison of saved weights on held-out handwriting examples.

Run: python -m numai.benchmark --baseline models/arabic-baseline.npz
Synthetic stress cases measure specific distortions, not personal handwriting.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import cv2

from .camera import recognize_frame
from .data import load_split
from .handwriting import STRESS_VARIANTS, stress_images
from .model import MLP


def measure(model, images, labels, threshold=0.85):
    probabilities = np.concatenate([model.predict_proba(images[i:i+512].reshape(-1, 784))
                                    for i in range(0, len(images), 512)])
    predicted = probabilities.argmax(axis=1)
    ordered = np.sort(probabilities, axis=1)
    accepted = (ordered[:, -1] >= threshold) & (ordered[:, -1]-ordered[:, -2] >= 0.25)
    correct = predicted == labels
    return {'samples': len(labels), 'accuracy': float(correct.mean()),
            'accepted': int(accepted.sum()), 'correct_accepted': int((accepted & correct).sum()),
            'wrong_accepted': int((accepted & ~correct).sum()),
            'unanswered': int((~accepted).sum())}


def measure_scenes(model, images, labels):
    """Three digits per gradient-lit frame, deterministic sizes and positions."""
    rng = np.random.default_rng(2026)
    metrics = {'samples': 300, 'localized': 0, 'correct_accepted': 0,
               'wrong_accepted': 0, 'unanswered': 0, 'extra_candidates': 0}
    for start in range(0, 300, 3):
        gray = np.tile(np.linspace(150, 255, 640, dtype=np.float32), (480, 1))
        targets = []
        for column in range(3):
            size = int(rng.integers(84, 169))
            x = column*210 + int(rng.integers(5, 210-size))
            y = int(rng.integers(5, 475-size))
            ink = cv2.resize(images[start+column], (size, size))
            gray[y:y+size, x:x+size] *= 1-ink
            targets.append((x, y, size, int(labels[start+column])))
        readings = recognize_frame(model, cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR))
        used = set()
        for x, y, size, expected in targets:
            matches = [(i, reading) for i, reading in enumerate(readings)
                       if x <= reading.box[0]+reading.box[2]/2 < x+size
                       and y <= reading.box[1]+reading.box[3]/2 < y+size]
            if len(matches) != 1:
                metrics['unanswered'] += 1
                continue
            index, reading = matches[0]
            used.add(index)
            metrics['localized'] += 1
            if reading.value is None:
                metrics['unanswered'] += 1
            else:
                metrics['correct_accepted' if reading.value == expected else 'wrong_accepted'] += 1
        metrics['extra_candidates'] += len(readings)-len(used)
    return metrics


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=root/'models/arabic.npz')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--data', type=Path, default=root/'data/mnist')
    parser.add_argument('--output', type=Path, default=root/'models/handwriting-comparison.json')
    args = parser.parse_args()
    images, labels = load_split(args.data, train=False)
    stressed = stress_images(images)
    report = {'note': 'Held-out MNIST and synthetic stress; not real-camera accuracy.', 'models': {}}
    for name, path in [('current', args.model), ('baseline', args.baseline)]:
        if path is None:
            continue
        model = MLP.load(path)
        metrics = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'clean': measure(model, images, labels),
                   'handwriting': measure(model, stressed, labels),
                   'synthetic_camera': measure_scenes(model, stressed, labels),
                   'variants': {variant: measure(model, stressed[i::len(STRESS_VARIANTS)],
                                                 labels[i::len(STRESS_VARIANTS)])
                                for i, variant in enumerate(STRESS_VARIANTS)}}
        report['models'][name] = metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

import argparse
import sys
from pathlib import Path
from urllib.error import URLError

import cv2

from .camera import classify, run_camera
from .data import download, load_split
from .model import MLP
from .train import evaluate, train
from .vision import extract_digit


ROOT = Path(__file__).resolve().parent.parent


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Нужно положительное целое число.')
    return number


def probability(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError('Значение должно быть в диапазоне (0, 1].')
    return number


def main():
    parser = argparse.ArgumentParser(description='NumAI: собственная модель для арабских цифр 0–9')
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train', help='Обучить модель со случайных весов на MNIST')
    training.add_argument('--epochs', type=positive_int, default=15)
    training.add_argument('--batch-size', type=positive_int, default=128)
    training.add_argument('--lr', type=probability, default=0.001)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--limit', type=positive_int, help='Ограничение обучающей выборки для быстрой проверки')
    training.add_argument('--overwrite', action='store_true', help='Заново обучить и заменить существующие веса')
    testing = commands.add_parser('evaluate', help='Проверить веса на тестовой выборке MNIST')
    camera = commands.add_parser('camera', help='Распознавать цифры с камеры')
    camera.add_argument('--camera', type=int, default=0, help='Индекс камеры (по умолчанию 0)')
    camera.add_argument('--stable-frames', type=positive_int, default=5)
    camera.add_argument('--no-preview', action='store_true', help='Только терминал, без окна изображения')
    camera.add_argument('--max-frames', type=positive_int, default=0, help='Остановиться после N кадров')
    image = commands.add_parser('image', help='Распознать цифру на светлом изображении без камеры')
    image.add_argument('path', type=Path)
    for command in (training, testing, camera, image):
        command.add_argument('--model', type=Path, default=ROOT/'models'/'arabic.npz')
    for command in (training, testing):
        command.add_argument('--data', type=Path, default=ROOT/'data'/'mnist')
    for command in (camera, image):
        command.add_argument('--threshold', type=probability, default=0.85)
    args = parser.parse_args()
    try:
        if args.command == 'train':
            train(args)
            return 0
        if not args.model.exists():
            raise ValueError('Веса модели не найдены. Сначала выполните: python -m numai train')
        model = MLP.load(args.model)
        if model.sizes[0] != 784 or model.sizes[-1] != 10:
            raise ValueError('Нужна модель с входом 28×28 и выходом для цифр 0–9.')
        if args.command == 'camera':
            if args.camera < 0:
                raise ValueError('Индекс камеры должен быть неотрицательным.')
            run_camera(model, args.camera, args.threshold, args.stable_frames,
                       not args.no_preview, args.max_frames)
        elif args.command == 'image':
            frame = cv2.imread(str(args.path))
            if frame is None:
                raise ValueError('Не удалось прочитать изображение: ' + str(args.path))
            number, score = classify(model, extract_digit(frame), args.threshold)
            print('Цифра не найдена или модель не уверена.' if number is None
                  else f'Вижу цифру: {number} (оценка модели: {score:.0%})')
        else:
            download(args.data)
            images, labels = load_split(args.data, train=False)
            metrics = evaluate(model, images, labels)
            print(f'Точность MNIST: {metrics["accuracy"]:.2%}; примеров: {metrics["samples"]}')
            for digit, accuracy in metrics['per_digit_accuracy'].items():
                print(f'  {digit}: {accuracy:.2%}')
        return 0
    except KeyboardInterrupt:
        print('\nОстановлено.')
        return 130
    except (OSError, ValueError, RuntimeError, URLError, cv2.error, KeyError) as exc:
        print('Ошибка: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

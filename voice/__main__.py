import argparse
import sys

from .listen import run_listener
from .models import MODELS


def main():
    parser = argparse.ArgumentParser(
        description='Voice agent: слушает микрофон и печатает сказанные фразы в терминал',
    )
    parser.add_argument(
        '--lang',
        choices=sorted(MODELS),
        default='ru',
        help='Язык распознавания (по умолчанию ru)',
    )
    parser.add_argument(
        '--device',
        type=int,
        default=None,
        help='Индекс устройства ввода sounddevice (по умолчанию системный)',
    )
    args = parser.parse_args()
    try:
        run_listener(language=args.lang, device=args.device)
    except Exception as error:
        print('Ошибка: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

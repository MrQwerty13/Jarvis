"""Command line entry point for camAI."""

import argparse
from pathlib import Path

from numai.model import MLP

from .camera import run_camera


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description='camAI: числа, лица и пальцы')
    parser.add_argument('command', nargs='?', default='camera', choices=('camera',))
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.65)
    parser.add_argument('--stable-frames', type=int, default=4)
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--model', type=Path, default=ROOT / 'models' / 'arabic.npz')
    args = parser.parse_args()
    if args.camera < 0 or not 0 < args.threshold <= 1:
        parser.error('Некорректные параметры камеры или threshold.')
    model = MLP.load(args.model)
    run_camera(model, index=args.camera, threshold=args.threshold,
               stable_frames=max(1, args.stable_frames), preview=not args.no_preview,
               max_frames=args.max_frames)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

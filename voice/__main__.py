import argparse
import sys

from .brain import ensure_brain
from .listen import run_listener
from .models import MODELS
from .talk import run_talk


def main():
    parser = argparse.ArgumentParser(
        description='Jarvis voice agent: слушает, отвечает мини-сетью и говорит вслух',
    )
    commands = parser.add_subparsers(dest='command')

    talk = commands.add_parser('talk', help='Диалог: речь → ответ сети → голос (по умолчанию)')
    talk.add_argument('--lang', choices=sorted(MODELS), default='ru')
    talk.add_argument('--device', type=int, default=None)
    talk.add_argument('--threshold', type=float, default=0.45)
    talk.add_argument('--mute', action='store_true', help='Только текст, без озвучки')

    listen = commands.add_parser('listen', help='Только печатать распознанные фразы')
    listen.add_argument('--lang', choices=sorted(MODELS), default='ru')
    listen.add_argument('--device', type=int, default=None)

    train = commands.add_parser('train', help='Обучить мини-сеть диалога заново')
    train.add_argument('--lang', choices=sorted(MODELS), default='ru')

    parser.add_argument('--lang', choices=sorted(MODELS), default='ru')
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.45)
    parser.add_argument('--mute', action='store_true')

    args = parser.parse_args()
    command = args.command or 'talk'

    try:
        if command == 'listen':
            run_listener(language=args.lang, device=args.device)
        elif command == 'train':
            ensure_brain(language=args.lang, force_train=True)
        else:
            run_talk(
                language=args.lang,
                device=getattr(args, 'device', None),
                threshold=getattr(args, 'threshold', 0.45),
                mute_tts=getattr(args, 'mute', False),
            )
    except Exception as error:
        print('Ошибка: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

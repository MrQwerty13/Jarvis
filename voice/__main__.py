import argparse
import sys

from .brain import ensure_brain
from .bridge import run_bridge
from .listen import run_listener
from .library import DEFAULT_LIBRARY_PATH, build_library
from .models import MODELS
from .ollama_chat import DEFAULT_HOST, DEFAULT_MODEL
from .talk import handle_text, run_talk


def add_common_flags(parser):
    parser.add_argument('--lang', choices=sorted(MODELS), default='ru')
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.45, help='Порог для backend=mini')
    parser.add_argument('--mute', action='store_true', help='Только текст, без озвучки')
    parser.add_argument(
        '--backend',
        choices=('ollama', 'mini'),
        default='ollama',
        help='Движок ответа: ollama (по умолчанию) или mini MLP',
    )
    parser.add_argument('--ollama-model', default=DEFAULT_MODEL, help='Модель Ollama (по умолчанию qwen:14b)')
    parser.add_argument('--ollama-host', default=DEFAULT_HOST, help='URL Ollama API')
    parser.add_argument('--camera', type=int, default=0, help='Индекс камеры для распознавания чисел')


def main():
    parser = argparse.ArgumentParser(
        description='Jarvis voice agent: Ollama/qwen, веб-поиск, NumAI и wake-word через Go',
    )
    commands = parser.add_subparsers(dest='command')

    talk = commands.add_parser('talk', help='Диалог напрямую в Python')
    add_common_flags(talk)
    talk.add_argument('--wake', action='store_true', help='Реагировать только на «Джарвис, …»')

    listen = commands.add_parser('listen', help='Только печатать распознанные фразы')
    listen.add_argument('--lang', choices=sorted(MODELS), default='ru')
    listen.add_argument('--device', type=int, default=None)

    train = commands.add_parser('train', help='Обучить мини-сеть диалога заново')
    train.add_argument('--lang', choices=sorted(MODELS), default='ru')

    library = commands.add_parser('library', help='Индексировать локальные документы (.txt/.md/.pdf)')
    library.add_argument('--path', required=True, help='Папка с книгами, на которые у вас есть права')
    library.add_argument('--output', default=str(DEFAULT_LIBRARY_PATH), help='Путь JSON-индекса')

    once = commands.add_parser('once', help='Один текстовый запрос → ответ (+ голос)')
    add_common_flags(once)
    once.add_argument('--text', required=True, help='Текст команды без имени')

    bridge = commands.add_parser('bridge', help='JSONL-мост для Go-демона')
    add_common_flags(bridge)

    add_common_flags(parser)
    parser.add_argument('--wake', action='store_true', help='Реагировать только на «Джарвис, …»')

    args = parser.parse_args()
    command = args.command or 'talk'

    try:
        if command == 'listen':
            run_listener(language=args.lang, device=args.device)
        elif command == 'train':
            ensure_brain(language=args.lang, force_train=True)
        elif command == 'library':
            count = build_library(args.path, args.output)
            print(f'Индекс создан: {count} фрагментов, {args.output}')
        elif command == 'once':
            result = handle_text(
                args.text,
                language=args.lang,
                backend=args.backend,
                ollama_model=args.ollama_model,
                ollama_host=args.ollama_host,
                mute_tts=args.mute,
                camera_index=args.camera,
            )
            return 0 if not result.get('done') else 0
        elif command == 'bridge':
            run_bridge(
                language=args.lang,
                device=args.device,
                backend=args.backend,
                ollama_model=args.ollama_model,
                ollama_host=args.ollama_host,
                mute_tts=args.mute,
                camera_index=args.camera,
            )
        else:
            run_talk(
                language=args.lang,
                device=args.device,
                threshold=args.threshold,
                mute_tts=args.mute,
                backend=args.backend,
                ollama_model=args.ollama_model,
                ollama_host=args.ollama_host,
                camera_index=args.camera,
                require_wake=bool(getattr(args, 'wake', False)),
            )
    except Exception as error:
        print('Ошибка: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

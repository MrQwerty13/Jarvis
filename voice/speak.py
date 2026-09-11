"""Синтез речи через бесплатный Edge TTS (интернет)."""

import asyncio
import tempfile
from pathlib import Path
import subprocess
import sys

import edge_tts


VOICES = {
    'ru': 'ru-RU-DmitryNeural',
    'en': 'en-US-GuyNeural',
}


async def _synthesize(text, voice, path):
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(str(path))


def speak(text, language='ru', voice=None):
    """Сказать фразу вслух. Нужен интернет; на macOS играет через afplay."""
    text = (text or '').strip()
    if not text:
        return
    chosen = voice or VOICES.get(language, VOICES['ru'])
    with tempfile.TemporaryDirectory(prefix='jarvis-tts-') as folder:
        audio_path = Path(folder) / 'reply.mp3'
        try:
            asyncio.run(_synthesize(text, chosen, audio_path))
        except Exception as error:
            print('TTS ошибка: ' + str(error), file=sys.stderr, flush=True)
            return
        if sys.platform == 'darwin':
            subprocess.run(['afplay', str(audio_path)], check=False)
        else:
            # Linux fallback: пытаемся через ffplay, иначе пропускаем звук.
            result = subprocess.run(
                ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', str(audio_path)],
                check=False,
            )
            if result.returncode != 0:
                print('Не удалось воспроизвести речь (нужен afplay или ffplay).', file=sys.stderr, flush=True)

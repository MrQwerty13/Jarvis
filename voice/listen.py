"""Слушает микрофон и печатает распознанные фразы в терминал."""

import json
import queue
import sys

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

from .models import ensure_model


SAMPLE_RATE = 16000
BLOCK_SIZE = 4000


def run_listener(language='ru', device=None):
    SetLogLevel(-1)
    model_dir = ensure_model(language)
    model = Model(str(model_dir))
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(False)

    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr, flush=True)
        audio_queue.put(bytes(indata))

    device_name = 'по умолчанию'
    if device is not None:
        device_name = str(device)
    print('Voice agent запущен. Язык: ' + language + ', микрофон: ' + device_name, flush=True)
    print('Говорите — фразы появятся ниже. Ctrl+C — выход.', flush=True)
    print('-' * 40, flush=True)

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype='int16',
            channels=1,
            device=device,
            callback=callback,
        ):
            while True:
                data = audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    text = json.loads(recognizer.Result()).get('text', '').strip()
                    if text:
                        print('Вы сказали: ' + text, flush=True)
    except KeyboardInterrupt:
        print('\nОстановка.', flush=True)

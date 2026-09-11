"""Диалоговый цикл: слушает → мини-сеть отвечает → говорит вслух."""

import json
import queue
import sys
import threading
import time

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

from .brain import ensure_brain
from .models import ensure_model
from .speak import speak


SAMPLE_RATE = 16000
BLOCK_SIZE = 4000


def run_talk(language='ru', device=None, threshold=0.45, mute_tts=False):
    SetLogLevel(-1)
    brain = ensure_brain(language=language)
    model = Model(str(ensure_model(language)))
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(False)

    audio_queue = queue.Queue()
    pause = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr, flush=True)
        if not pause.is_set():
            audio_queue.put(bytes(indata))

    def drain():
        while True:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
        # Сброс фразы, чтобы эхо динамика не ушло в следующий ответ.
        recognizer.Reset()

    print('Jarvis talk mode. Язык: ' + language, flush=True)
    print('Говорите. Ответ появится в терминале и голосом. Ctrl+C — выход.', flush=True)
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
                if not recognizer.AcceptWaveform(data):
                    continue
                text = json.loads(recognizer.Result()).get('text', '').strip()
                if not text:
                    continue
                print('Вы: ' + text, flush=True)
                reply, label, confidence = brain.answer(text, threshold=threshold)
                tag = label or 'fallback'
                print(f'Джарвис ({tag}, {confidence:.0%}): {reply}', flush=True)
                if mute_tts:
                    continue
                pause.set()
                drain()
                try:
                    speak(reply, language=language)
                finally:
                    time.sleep(0.15)
                    drain()
                    pause.clear()
                if label == 'bye':
                    print('Диалог завершён.', flush=True)
                    break
    except KeyboardInterrupt:
        print('\nОстановка.', flush=True)

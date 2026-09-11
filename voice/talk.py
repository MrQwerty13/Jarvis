"""Диалоговый цикл и одноразовый ответ."""

import json
import queue
import sys
import threading
import time

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

from .actions import ActionRouter
from .models import ensure_model
from .ollama_chat import DEFAULT_HOST, DEFAULT_MODEL
from .speak import speak
from .wake import extract_command


SAMPLE_RATE = 16000
BLOCK_SIZE = 4000


def handle_text(
    text,
    language='ru',
    backend='ollama',
    ollama_model=DEFAULT_MODEL,
    ollama_host=DEFAULT_HOST,
    mute_tts=False,
    camera_index=0,
    router=None,
):
    speak_fn = None if mute_tts else (lambda phrase: speak(phrase, language=language))
    router = router or ActionRouter(
        language=language,
        backend=backend,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        camera_index=camera_index,
        speak_fn=speak_fn,
        mute_tts=mute_tts,
    )
    result = router.handle(text)
    reply = result.get('reply') or ''
    print(f'Джарвис ({result.get("tag")}): {reply}', flush=True)
    if reply and not mute_tts and not result.get('skip_tts'):
        speak(reply, language=language)
    return result


def run_talk(
    language='ru',
    device=None,
    threshold=0.45,
    mute_tts=False,
    backend='ollama',
    ollama_model=DEFAULT_MODEL,
    ollama_host=DEFAULT_HOST,
    camera_index=0,
    require_wake=False,
):
    del threshold  # используется только backend=mini внутри ActionRouter/brain
    SetLogLevel(-1)
    speak_fn = None if mute_tts else (lambda phrase: speak(phrase, language=language))
    router = ActionRouter(
        language=language,
        backend=backend,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        camera_index=camera_index,
        speak_fn=speak_fn,
        mute_tts=mute_tts,
    )
    source_name = 'ollama/' + ollama_model if backend == 'ollama' else 'mini-mlp'

    stt_model = Model(str(ensure_model(language)))
    recognizer = KaldiRecognizer(stt_model, SAMPLE_RATE)
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
        recognizer.Reset()

    print('Jarvis talk mode. Backend: ' + source_name + ', язык: ' + language, flush=True)
    if require_wake:
        print('Нужно обращение: «Джарвис, …»', flush=True)
    print('Команды: «Ты свободен» — выход; «Распознавание чисел» — камера NumAI.', flush=True)
    print('Говорите. Ctrl+C — выход.', flush=True)
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
                command = text
                if require_wake:
                    extracted = extract_command(text)
                    if extracted is None:
                        print('(без имени — пропуск)', flush=True)
                        continue
                    command = extracted or 'привет'

                pause.set()
                drain()
                try:
                    result = router.handle(command)
                    reply = result.get('reply') or ''
                    print(f'Джарвис ({result.get("tag")}): {reply}', flush=True)
                    if reply and not mute_tts:
                        speak(reply, language=language)
                finally:
                    time.sleep(0.15)
                    drain()
                    pause.clear()

                if result.get('done'):
                    print('Диалог завершён.', flush=True)
                    break
    except KeyboardInterrupt:
        print('\nОстановка.', flush=True)

"""JSONL-мост для Go-демона: слушает фразы и выполняет команды по запросу."""

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


SAMPLE_RATE = 16000
BLOCK_SIZE = 4000


def _emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def run_bridge(
    language='ru',
    device=None,
    backend='ollama',
    ollama_model=DEFAULT_MODEL,
    ollama_host=DEFAULT_HOST,
    mute_tts=False,
    camera_index=0,
):
    SetLogLevel(-1)
    router = ActionRouter(
        language=language,
        backend=backend,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        camera_index=camera_index,
    )
    stt_model = Model(str(ensure_model(language)))
    recognizer = KaldiRecognizer(stt_model, SAMPLE_RATE)
    recognizer.SetWords(False)

    audio_queue = queue.Queue()
    pause = threading.Event()
    stop = threading.Event()

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

    def listen_loop():
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype='int16',
            channels=1,
            device=device,
            callback=callback,
        ):
            while not stop.is_set():
                try:
                    data = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if pause.is_set():
                    continue
                if not recognizer.AcceptWaveform(data):
                    continue
                text = json.loads(recognizer.Result()).get('text', '').strip()
                if text:
                    _emit({'type': 'heard', 'text': text})

    def handle_act(text):
        pause.set()
        drain()
        try:
            result = router.handle(text)
            reply = result.get('reply') or ''
            _emit({
                'type': 'reply',
                'text': text,
                'reply': reply,
                'tag': result.get('tag'),
                'done': bool(result.get('done')),
            })
            if reply and not mute_tts:
                speak(reply, language=language)
            return bool(result.get('done'))
        finally:
            time.sleep(0.15)
            drain()
            pause.clear()

    listener = threading.Thread(target=listen_loop, daemon=True)
    listener.start()
    _emit({
        'type': 'ready',
        'backend': backend,
        'model': ollama_model if backend == 'ollama' else 'mini',
        'language': language,
    })

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                _emit({'type': 'error', 'error': 'invalid json'})
                continue
            action = message.get('type') or message.get('action')
            if action in ('shutdown', 'quit', 'exit'):
                _emit({'type': 'bye'})
                break
            if action == 'act':
                text = (message.get('text') or '').strip()
                if not text:
                    continue
                if handle_act(text):
                    _emit({'type': 'bye'})
                    break
            else:
                _emit({'type': 'error', 'error': 'unknown action'})
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()

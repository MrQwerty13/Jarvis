"""Клиент Ollama для голосового диалога Jarvis (по умолчанию qwen:14b + веб-поиск)."""

import json
import subprocess
import urllib.error
import urllib.request

from .search import format_search_context, web_search


DEFAULT_HOST = 'http://127.0.0.1:11434'
DEFAULT_MODEL = 'qwen:14b'

SYSTEM_PROMPTS = {
    'ru': (
        'Ты Джарвис — голосовой помощник на компьютере пользователя. '
        'Отвечай по-русски коротко: 1–2 предложения, без markdown и списков. '
        'Если нужны свежие или фактические данные из интернета — вызови инструмент web_search. '
        'Не выдумывай новости и факты, которых нет в результатах поиска.'
    ),
    'en': (
        'You are Jarvis, a local voice assistant. '
        'Reply in English in 1–2 short sentences without markdown. '
        'Use the web_search tool for fresh or factual internet information. '
        'Do not invent facts missing from search results.'
    ),
}

SEARCH_HINTS = (
    'найди', 'погугли', 'поиск', 'поискай', 'что такое', 'кто такой', 'кто такая',
    'когда', 'где находится', 'новости', 'курс', 'погода', 'сколько стоит',
    'search', 'google', 'who is', 'what is', 'news', 'weather', 'price',
)

BYE_WORDS = {
    'ru': ('ты свободен', 'ты свободна', 'отбой', 'выключайся'),
    'en': ('you are free', 'dismissed', 'shut down'),
}

WEB_SEARCH_TOOL = {
    'type': 'function',
    'function': {
        'name': 'web_search',
        'description': 'Search the public internet from the user device via DuckDuckGo.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Search query in the user language',
                },
            },
            'required': ['query'],
        },
    },
}


class OllamaError(RuntimeError):
    pass


def _request(host, path, payload=None, timeout=180):
    url = host.rstrip('/') + path
    data = None
    headers = {'Accept': 'application/json'}
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(url, data=data, headers=headers, method='GET' if data is None else 'POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='ignore')
        raise OllamaError('HTTP ' + str(error.code) + ': ' + detail) from error
    except urllib.error.URLError as error:
        raise OllamaError(
            'Ollama недоступна на ' + host + '. Запустите: ollama serve. Детали: ' + str(error)
        ) from error


def is_alive(host=DEFAULT_HOST):
    try:
        _request(host, '/api/tags', timeout=3)
        return True
    except OllamaError:
        return False


def list_models(host=DEFAULT_HOST):
    payload = _request(host, '/api/tags', timeout=10)
    return [item.get('name', '') for item in payload.get('models', [])]


def has_model(model, host=DEFAULT_HOST):
    names = list_models(host)
    return any(name == model or name.startswith(model + ':') or name == model + ':latest' for name in names)


def ensure_model(model=DEFAULT_MODEL, host=DEFAULT_HOST):
    if not is_alive(host):
        raise OllamaError('Ollama не запущена. Выполните в другом терминале: ollama serve')
    if has_model(model, host=host):
        return model
    print('Модель ' + model + ' не найдена. Скачиваю через ollama pull…', flush=True)
    result = subprocess.run(['ollama', 'pull', model], check=False)
    if result.returncode != 0 or not has_model(model, host=host):
        raise OllamaError('Не удалось скачать модель: ' + model)
    return model


def is_bye(text, language='ru'):
    lowered = ' '.join((text or '').lower().replace('ё', 'е').split())
    return any(word in lowered for word in BYE_WORDS.get(language, BYE_WORDS['ru']))


def needs_search(text):
    lowered = ' '.join((text or '').lower().replace('ё', 'е').split())
    return any(hint in lowered for hint in SEARCH_HINTS)


def clean_reply(text, language='ru'):
    text = ' '.join((text or '').split()).strip()
    if not text:
        return 'Не получилось сформулировать ответ. Повтори, пожалуйста.' if language == 'ru' else 'I could not form an answer. Please repeat.'
    # До двух коротких предложений для голосового ответа.
    parts = []
    rest = text
    for _ in range(2):
        cut = None
        for separator in ('. ', '! ', '? '):
            index = rest.find(separator)
            if index != -1 and (cut is None or index < cut[0]):
                cut = (index, separator)
        if cut is None:
            parts.append(rest)
            rest = ''
            break
        index, separator = cut
        parts.append(rest[:index + 1].strip())
        rest = rest[index + len(separator):].strip()
        if not rest:
            break
    text = ' '.join(part for part in parts if part)
    if len(text) > 320:
        text = text[:317].rstrip() + '…'
    return text


def _tool_call_name(call):
    if not isinstance(call, dict):
        return '', {}
    if 'function' in call and isinstance(call['function'], dict):
        function = call['function']
        name = function.get('name') or ''
        arguments = function.get('arguments') or {}
    else:
        name = call.get('name') or ''
        arguments = call.get('arguments') or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {'query': arguments}
    if not isinstance(arguments, dict):
        arguments = {}
    return name, arguments


class OllamaChat:
    def __init__(self, model=DEFAULT_MODEL, language='ru', host=DEFAULT_HOST):
        self.model = model
        self.language = language
        self.host = host.rstrip('/')
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS['ru'])}]
        self.tools_supported = True

    def answer(self, text):
        text = (text or '').strip()
        if not text:
            return '', False
        self.messages.append({'role': 'user', 'content': text})

        reply_message = {}
        if self.tools_supported:
            try:
                reply_message = self._chat(self._history(), tools=True)
            except OllamaError as error:
                if 'does not support tools' in str(error) or 'HTTP 400' in str(error):
                    self.tools_supported = False
                    reply_message = {}
                else:
                    raise

        tool_calls = reply_message.get('tool_calls') or []
        if tool_calls:
            self.messages.append(reply_message)
            for call in tool_calls:
                name, arguments = _tool_call_name(call)
                if name != 'web_search':
                    continue
                query = (arguments.get('query') or text).strip()
                print('Поиск в интернете: ' + query, flush=True)
                results = web_search(query)
                context = format_search_context(query, results)
                self.messages.append({'role': 'tool', 'content': context})
            reply_message = self._chat(self._history(limit=12), tools=False)
        else:
            if needs_search(text):
                self._inject_search(text)
            reply_message = self._chat(self._history(limit=12), tools=False)

        reply = clean_reply(reply_message.get('content', ''), language=self.language)
        self.messages.append({'role': 'assistant', 'content': reply})
        self._compact_history()
        return reply, is_bye(text, self.language)

    def _history(self, limit=10):
        return [self.messages[0]] + self.messages[1:][-limit:]

    def _compact_history(self):
        compacted = [self.messages[0]]
        for message in self.messages[1:]:
            role = message.get('role')
            content = message.get('content')
            if role in ('user', 'assistant') and content:
                compacted.append({'role': role, 'content': content})
        self.messages = [compacted[0]] + compacted[1:][-10:]

    def _inject_search(self, query):
        print('Поиск в интернете: ' + query, flush=True)
        results = web_search(query)
        context = format_search_context(query, results)
        self.messages.append({
            'role': 'user',
            'content': (
                'Ниже справка из интернета. Ответь на мой предыдущий вопрос '
                'коротко по-русски, опираясь на эти факты:\n' + context
            ),
        })

    def _chat(self, messages, tools=False):
        payload = {
            'model': self.model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': 0.5,
                'top_p': 0.9,
                'num_predict': 160,
            },
        }
        if tools:
            payload['tools'] = [WEB_SEARCH_TOOL]
        result = _request(self.host, '/api/chat', payload=payload, timeout=300)
        return result.get('message') or {}
def ensure_ollama_chat(model=DEFAULT_MODEL, language='ru', host=DEFAULT_HOST):
    ensure_model(model=model, host=host)
    return OllamaChat(model=model, language=language, host=host)

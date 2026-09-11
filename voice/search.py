"""Поиск в интернете с устройства пользователя (DuckDuckGo)."""

import json
import re
import urllib.parse
import urllib.request


USER_AGENT = 'JarvisVoiceAgent/1.0 (+local; DuckDuckGo)'


def web_search(query, max_results=5):
    query = ' '.join((query or '').split())
    if not query:
        return []
    results = _instant_answer(query)
    if len(results) < max_results:
        for item in _html_results(query):
            if item not in results:
                results.append(item)
            if len(results) >= max_results:
                break
    return results[:max_results]


def format_search_context(query, results):
    if not results:
        return f'По запросу «{query}» ничего полезного не найдено.'
    lines = [f'Результаты поиска по запросу «{query}»:']
    for index, item in enumerate(results, start=1):
        title = item.get('title') or 'Без названия'
        snippet = item.get('snippet') or ''
        url = item.get('url') or ''
        lines.append(f'{index}. {title}. {snippet} {url}'.strip())
    return '\n'.join(lines)


def _instant_answer(query):
    url = 'https://api.duckduckgo.com/?' + urllib.parse.urlencode({
        'q': query,
        'format': 'json',
        'no_html': 1,
        'skip_disambig': 1,
    })
    try:
        payload = _get_json(url)
    except Exception:
        return []
    results = []
    abstract = (payload.get('AbstractText') or '').strip()
    heading = (payload.get('Heading') or '').strip()
    abstract_url = (payload.get('AbstractURL') or '').strip()
    if abstract:
        results.append({
            'title': heading or query,
            'snippet': abstract,
            'url': abstract_url,
        })
    for topic in payload.get('RelatedTopics') or []:
        if not isinstance(topic, dict):
            continue
        if 'Text' in topic:
            results.append({
                'title': (topic.get('Text') or '')[:80],
                'snippet': topic.get('Text') or '',
                'url': topic.get('FirstURL') or '',
            })
        for nested in topic.get('Topics') or []:
            results.append({
                'title': (nested.get('Text') or '')[:80],
                'snippet': nested.get('Text') or '',
                'url': nested.get('FirstURL') or '',
            })
    return results


def _html_results(query):
    url = 'https://html.duckduckgo.com/html/?' + urllib.parse.urlencode({'q': query})
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            html = response.read().decode('utf-8', errors='ignore')
    except Exception:
        return []
    results = []
    # Простой разбор выдачи html.duckduckgo.com без внешних парсеров.
    blocks = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for href, title, snippet in blocks:
        results.append({
            'title': _strip_tags(title),
            'snippet': _strip_tags(snippet),
            'url': _unwrap_ddg_link(href),
        })
    return results


def _unwrap_ddg_link(href):
    href = urllib.parse.unquote(href)
    match = re.search(r'[?&]uddg=([^&]+)', href)
    if match:
        return urllib.parse.unquote(match.group(1))
    return href


def _strip_tags(value):
    value = re.sub(r'<[^>]+>', ' ', value or '')
    return ' '.join(value.split())


def _get_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode('utf-8'))

"""Учебные фразы и ответы для мини-сети диалога."""

# Каждая категория: примеры входа (как их может распознать Vosk) и варианты ответа.
INTENTS_RU = [
    {
        'name': 'greeting',
        'patterns': [
            'привет', 'здравствуй', 'здравствуйте', 'добрый день', 'доброе утро',
            'добрый вечер', 'хай', 'хелло', 'салют', 'привет джарвис',
        ],
        'responses': [
            'Привет! Я Джарвис. Чем могу помочь?',
            'Здравствуй! Слушаю тебя.',
            'Привет! Готов к разговору.',
        ],
    },
    {
        'name': 'how_are_you',
        'patterns': [
            'как дела', 'как ты', 'как жизнь', 'что нового', 'как настроение',
            'как сам', 'ты как',
        ],
        'responses': [
            'Отлично, схемы в порядке. А у тебя?',
            'Работаю штатно. Спасибо, что спросил.',
            'Всё хорошо. Готов помогать.',
        ],
    },
    {
        'name': 'name',
        'patterns': [
            'как тебя зовут', 'кто ты', 'твоё имя', 'представься', 'ты кто',
            'как твоё имя', 'скажи своё имя',
        ],
        'responses': [
            'Меня зовут Джарвис. Я голосовой помощник в этом проекте.',
            'Я Джарвис — мини-нейросеть для разговора.',
        ],
    },
    {
        'name': 'thanks',
        'patterns': [
            'спасибо', 'благодарю', 'спс', 'отлично спасибо', 'спасибо большое',
        ],
        'responses': [
            'Пожалуйста!',
            'Всегда рад помочь.',
            'Обращайся.',
        ],
    },
    {
        'name': 'bye',
        'patterns': [
            'пока', 'до свидания', 'увидимся', 'спокойной ночи', 'бывай',
            'всего доброго', 'конец',
        ],
        'responses': [
            'Пока! Буду на связи.',
            'До встречи!',
            'Удачи! Выключаюсь из разговора.',
        ],
    },
    {
        'name': 'help',
        'patterns': [
            'помощь', 'что ты умеешь', 'что ты можешь', 'помоги', 'команды',
            'расскажи что умеешь',
        ],
        'responses': [
            'Я слушаю голос, понимаю простые фразы мини-сетью и отвечаю голосом.',
            'Могу поздороваться, представиться, ответить как дела и попрощаться. Говори коротко и ясно.',
        ],
    },
    {
        'name': 'time',
        'patterns': [
            'который час', 'сколько время', 'скажи время', 'какое сейчас время',
            'сколько сейчас времени',
        ],
        'responses': [
            '__TIME__',
        ],
    },
    {
        'name': 'joke',
        'patterns': [
            'расскажи анекдот', 'пошути', 'шутка', 'рассмеши', 'анекдот',
        ],
        'responses': [
            'Почему нейросеть не спорит? Потому что у неё всегда есть последний слой.',
            'Программист заходит в лифт, нажимает один, потом ноль. Лифт отвечает: понял, булева логика.',
        ],
    },
    {
        'name': 'mood_good',
        'patterns': [
            'у меня всё хорошо', 'отлично', 'замечательно', 'супер', 'я в порядке',
            'всё отлично',
        ],
        'responses': [
            'Рад это слышать!',
            'Отлично. Тогда продолжаем.',
        ],
    },
    {
        'name': 'mood_bad',
        'patterns': [
            'мне плохо', 'устала', 'устал', 'грустно', 'плохое настроение',
        ],
        'responses': [
            'Жаль это слышать. Я рядом, если нужно просто поговорить.',
            'Держись. Могу рассказать шутку или просто послушать.',
        ],
    },
]

FALLBACK_RU = [
    'Не совсем понял. Повтори, пожалуйста, проще.',
    'Пока не распознал фразу. Скажи ещё раз.',
    'Хм, сложно. Попробуй другими словами.',
]

INTENTS_EN = [
    {
        'name': 'greeting',
        'patterns': ['hello', 'hi', 'hey', 'good morning', 'good evening', 'hi jarvis'],
        'responses': [
            'Hello! I am Jarvis. How can I help?',
            'Hi! I am listening.',
        ],
    },
    {
        'name': 'how_are_you',
        'patterns': ['how are you', 'how do you do', 'what is up', 'whats up'],
        'responses': [
            'Running fine. Thanks for asking.',
            'All systems nominal. And you?',
        ],
    },
    {
        'name': 'name',
        'patterns': ['what is your name', 'who are you', 'your name', 'introduce yourself'],
        'responses': [
            'I am Jarvis, a tiny talk network in this project.',
        ],
    },
    {
        'name': 'thanks',
        'patterns': ['thanks', 'thank you', 'thx'],
        'responses': ['You are welcome!', 'Anytime.'],
    },
    {
        'name': 'bye',
        'patterns': ['bye', 'goodbye', 'see you', 'good night'],
        'responses': ['Goodbye!', 'See you later!'],
    },
    {
        'name': 'help',
        'patterns': ['help', 'what can you do', 'commands'],
        'responses': [
            'I listen, classify simple phrases with a mini network, and speak back.',
        ],
    },
    {
        'name': 'time',
        'patterns': ['what time is it', 'tell me the time', 'current time'],
        'responses': ['__TIME__'],
    },
    {
        'name': 'joke',
        'patterns': ['tell a joke', 'joke', 'make me laugh'],
        'responses': [
            'Why did the neural net go to therapy? Too many unresolved layers.',
        ],
    },
]

FALLBACK_EN = [
    'I did not catch that. Please say it again.',
    'Not sure I understood. Try simpler words.',
]


def intents_for(language):
    if language == 'en':
        return INTENTS_EN, FALLBACK_EN
    return INTENTS_RU, FALLBACK_RU

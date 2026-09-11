import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voice.library import LibraryIndex, build_library
from voice.ollama_chat import OllamaChat


class LibraryTests(unittest.TestCase):
    def test_builds_and_searches_local_text_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'books'
            source.mkdir()
            (source / 'pushkin.txt').write_text(
                'Мороз и солнце; день чудесный! Еще ты дремлешь, друг прелестный.',
                encoding='utf-8',
            )
            output = Path(directory) / 'library.json'
            self.assertEqual(build_library(source, output, chunk_words=20, overlap=0), 1)
            index = LibraryIndex(output)
            self.assertEqual(index.search('мороз солнце')[0]['title'], 'pushkin')
            self.assertIn('Мороз и солнце', index.context('мороз солнце'))

    def test_indexes_pdf_text(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'book.pdf'
            source.touch()
            output = Path(directory) / 'library.json'
            with patch('voice.library._read_document', return_value='PDF текст о дороге и звездах') as read:
                self.assertEqual(build_library(source, output, chunk_words=20, overlap=0), 1)
            read.assert_called_once_with(source)
            self.assertEqual(LibraryIndex(output).search('звездах')[0]['title'], 'book')

    def test_new_ollama_chat_starts_with_only_system_context(self):
        first = OllamaChat()
        first.messages.append({'role': 'user', 'content': 'старый вопрос'})
        second = OllamaChat()
        self.assertEqual(len(second.messages), 1)
        self.assertEqual(second.messages[0]['role'], 'system')


if __name__ == '__main__':
    unittest.main()
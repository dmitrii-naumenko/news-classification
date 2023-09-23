import nltk
from typing import List, Dict
import re
import string


class Preprocessor:
    """Class for preprocessing data."""

    def __init__(self):
        self.handle_punctuation_expr = re.compile(f'[{string.punctuation}]')
        self.re_arr = [
            re.compile(
                r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'),
            re.compile(r'<.*?>'),
            re.compile(r'[%s]' % re.escape(string.punctuation)),
            re.compile(r'\s+'),
            re.compile(r'\[[0-9]*\]'),
            re.compile(r'[^\w\s]'),
            re.compile(r'\d'),
        ]
        #nltk.download('punkt')

    def handle_punctuation(self, inp_str: str) -> str:
        return self.handle_punctuation_expr.sub(' ', inp_str)

    def simple_preproc(self, inp_str: str) -> List[str]:
        text = self.handle_punctuation(inp_str)
        text = text.lower().replace('\xa0', ' ').strip()
        for r in self.re_arr:
            text = r.sub(' ', text)

        return nltk.tokenize.word_tokenize(text, language="russian")

    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> List[str]:
        return [k for k, v in vocab.items() if v >= min_occurancies]

from spacy.lang.de import German
from typing import List
from pprint import pprint

from sentences.sentence_segmenter import SentenceSegmenter 

class Transformer:

    def __init__(self):
        self.nlp = German()
        self.nlp.tokenizer.rules = {}
        self.nlp.add_pipe(SentenceSegmenter())

    def extract(self, raw_text: List[str]) -> List[str]:
        doc = self.nlp.pipe(raw_text)
        return doc

if __name__ == "__main__":
    t = Transformer()
    text = ["Hallo. Das war der erste Satz. Mal schauen was noch kommen mag.",
            "Der zweite Versuch: 1. Wir wollen auch sowas testen, 2. Was denn sonst?",
            "Das letzte Dokument ist ganz kurz. Nur 2 Sätze.",
            "War ein Witz"]
    for t in t.extract(text):
        print("--------")
        for sent in t.sents:
            print(sent)
import os
from pathlib import Path

import torch.nn as nn
import torch
from gensim.models import Word2Vec as W2V, KeyedVectors

from .preprocessing import Tokenizer

TOKENIZER_PATH = Path("model")
DATA_PATH = Path("data")/"dataset"

class Word2Vec(nn.Module):

    def __init__(self,
                 embedding_path: Path=Path("embedding")/"word2vec"/"100.wv",
                 n_tokens: int=-1, 
                 embedding_size: int=100):
        super().__init__()
        word_model = KeyedVectors.load(embedding_path, mmap='r')
        self.id2tok, self.tok2id = {}, {}
        for word in word_model.vocab:
            index = word_model.vocab[word].index
            self.id2tok[index] = word
            self.tok2id[word] = index

        word_model = torch.Tensor(word_model.vectors)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding.from_pretrained(word_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.embedding(X)

    def get_word_mapping(self):
        return {"id2tok": self.id2tok, "tok2id": self.tok2id}

def train_word2vec(embedding_path: Path=Path("embedding")/"word2vec"/"100.wv"):
    tok = Tokenizer(TOKENIZER_PATH, normalize=True, mapping=None)
    sentences = []
    for verdict in os.listdir(DATA_PATH):
        ver_dict = tok.tokenize_verdict_without_id(DATA_PATH/verdict)
        for r in ["facts", "reasoning"]:
            for sentence in ver_dict[r]:
                sentences.append(sentence)
        
        for sentence in ver_dict["guiding_principle"][0]:
            sentences.append(sentence)
        for sentence in ver_dict["guiding_principle"][1]:
            sentences.append(sentence)

    model = W2V(sentences=sentences, vector_size=100, window=5, workers=8)
    model.save(embedding_path)



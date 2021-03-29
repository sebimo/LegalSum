import os
import io
from pathlib import Path
import pickle

from tqdm import tqdm
import torch.nn as nn
import torch
from gensim.models import Word2Vec as W2V, KeyedVectors
import numpy as np

from .preprocessing import Tokenizer

TOKENIZER_PATH = Path("model")
DATA_PATH = Path("data")/"dataset"

class Word2Vec(nn.Module):

    def __init__(self,
                 n_tokens: int=-1, 
                 embedding_size: int=100,
                 embedding_path: Path=Path("embedding")/"word2vec"/"100.wv",
                 abstractive=False):
        super().__init__()
        word_model = KeyedVectors.load(str(embedding_path), mmap='r')
        self.id2tok, self.tok2id = {}, {}
        for word in word_model.wv.vocab:
            index = word_model.wv.vocab[word].index
            self.id2tok[index] = word
            self.tok2id[word] = index

        word_model = torch.tensor(word_model.wv.vectors)
        self.embedding_size = embedding_size
        assert self.embedding_size == 100, "Pretrained word2vec embeddings have size 100"
        if abstractive:
            # We need to add one token ending a sentence
            word_model = torch.cat([word_model, torch.zeros([1,self.embedding_size], dtype=torch.float32)])
            assert word_model.shape[0]-1 not in self.id2tok
            self.id2tok[word_model.shape[0]-1] = "<end>"
            self.tok2id["<end>"] = word_model.shape[0]-1
        self.embedding = nn.Embedding.from_pretrained(word_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.embedding(X)

    def get_word_mapping(self):
        return {"id2tok": self.id2tok, "tok2id": self.tok2id}

    def get_name(self):
        return "word2vec"


class GloVe(nn.Module):

    def __init__(self,
                 n_tokens: int=-1,
                 embedding_size: int=100,
                 embedding_path: Path=Path("embedding")/"glove"/"glove.pkl",
                 abstractive=False):
        super().__init__()
        with open(embedding_path, "rb") as f:
            d = pickle.load(f)
        self.id2tok = d["id2tok"]
        self.tok2id = d["tok2id"]
        self.embedding_size = embedding_size
        assert self.embedding_size == 100, "Pretrained GloVe embeddings have size 100"
        glove = torch.tensor(d["wv"], dtype=torch.float32)
        if abstractive:
            # We need to add one token ending a sentence
            glove = torch.cat([glove, torch.zeros([1,self.embedding_size], dtype=torch.float32)])
            assert glove.shape[0]-1 not in self.id2tok
            self.id2tok[glove.shape[0]-1] = "<end>"
            self.tok2id["<end>"] = glove.shape[0]-1
        self.embedding = nn.Embedding.from_pretrained(glove)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.embedding(X)

    def get_word_mapping(self):
        return {"id2tok": self.id2tok, "tok2id": self.tok2id}

    def get_name(self):
        return "glove"

def train_word2vec(embedding_path: Path=Path("embedding")/"word2vec"/"100.wv"):
    """ Will automatically create a word2vec model from the verdicts found under data/dataset and save it to embedding_path"""
    tok = Tokenizer(TOKENIZER_PATH, normalize=True, mapping=None)
    sentences = []
    print("Train word2vec")
    for verdict in tqdm(os.listdir(DATA_PATH), desc="Tokenize"):
        ver_dict = tok.tokenize_verdict_without_id(DATA_PATH/verdict)
        for r in ["facts", "reasoning", "guiding_principle"]:
            for sentence in ver_dict[r]:
                sentences.append(sentence)

    print("Start training word2vec model")
    model = W2V(sentences=sentences, size=100, window=5, min_count=100, iter=15, workers=8)
    model.save(str(embedding_path))

def convert_glove(embedding_path: Path=Path("embedding")/"glove"/"100.wv", create_corpus: bool=False):
    """ Read the result from the glove implementation -> glove.pkl 
        Code taken from the python evalutation script from https://github.com/stanfordnlp/GloVe + adapted to our use case 
    """

    print("Reading GloVe embeddings")
    with open(Path("embedding")/"glove"/"vocab.txt", "r") as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(Path("embedding")/"glove"/"vectors.txt", "r") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    
    with open(Path("embedding")/"glove"/"glove.pkl", "wb") as f:
        pickle.dump({
            "id2tok": ivocab,
            "tok2id": vocab,
            "wv": W_norm
        }, f)

def create_glove_corpus():
    """ We have to write all the document texts into a single file, each line corresponds to one document.
        Tokens should be seperated by spaces.
    """
    tok = Tokenizer(TOKENIZER_PATH, normalize=True, mapping=None)
    lines = []
    print("Creating GloVe corpus")
    for verdict in tqdm(os.listdir(DATA_PATH), desc="Tokenize"):
        sentences = []
        ver_dict = tok.tokenize_verdict_without_id(DATA_PATH/verdict)
        for r in ["facts", "reasoning", "guiding_principle"]:
            for sentence in ver_dict[r]:
                sentences.append(" ".join(sentence))

        lines.append(" ".join(sentences)+"\n")

    with io.open(Path("data")/"dataset.txt", "w+", encoding="utf-8") as f:
        f.writelines(lines)

if __name__ == "__main__":
    """ create_glove_corpus() will convert the dataset to the needed structure for the c implementation of GloVe.
        The repository was cloned and changed to use the newly created file as input.
        convert_glove() will take the vectors.txt and vocab.txt from the result (need to be in embedding/glove) and
        convert them to a unified file with the index mappings to words + the word vectors as a numpy array.
        -> GloVe will read this pkl file 
    """
    #create_glove_corpus()
    #convert_glove()
    """ word2vec setup """
    #train_word2vec()

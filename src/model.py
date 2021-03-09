# Contains all the models and its components for the summarization task
# We might want to compile the models to get faster training, is that possible?
# TODO Attention-layer for Encoding

from pathlib import Path
from enum import Enum
from typing import List

import torch
import torch.nn as nn

class AttentionType(Enum):
    DOT = 1
    BILINEAR = 2
    ADDITIVE = 3

class HierarchicalEncoder(nn.Module):

    def __init__(self,
                 embedding_size: int = 200,
                 n_tokens: int = 50000,
                 activation: nn.Module = nn.ReLU(),
                 embedding_layer: nn.Module=nn.Embedding,
                 attention: str="DOT"):
        super(HierarchicalEncoder, self).__init__()
        self.embedding_size = embedding_size
        
        assert attention in ["DOT", "BILINEAR", "ADDITIVE"]
        if attention == "DOT":
            self.attention = Attention(self.embedding_size, AttentionType.DOT)
        elif attention == "BILINEAR":
            self.attention = Attention(self.embedding_size, AttentionType.BILINEAR, attention_sizes=[100])
        elif attention == "ADDITIVE":
            self.attention = Attention(self.embedding_size, AttentionType.ADDITIVE, attention_sizes=[100, 100])
        else:
            raise ValueError("Attention type unknown: "+attention+"; Choose: DOT, BILINEAR, ADDITIVE")
        self.dropout = dropout
        self._activation = activation
        
        # We might want to replace this with something different
        self._embedding = embedding_layer
        self._emb_layers = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            self._activation,
        )
        self._classification = nn.Sequential(
            nn.Linear(self.embedding_size, 1)
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        X = self._embedding(X)
        # Use the mask to exlude any embeddings of  padded vectors
        X = torch.mul(mask.unsqueeze(-1), X)

        X = self._emb_layers(X)

        X = self.attention(X)

        X = self._activation(X)
        X = self._classification(X)
        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        return self.sig(E)     

class Attention(nn.Module):

    def __init__(self,
                embedding_size: int = 200,
                attention_type: AttentionType=AttentionType.DOT,
                attention_sizes: List[int]=[]):
        """ Arguments:
                - embedding_size  : size of the incoming token embeddings
                - attention_type  : the way attention shall be calculated in the model
                - attention_sizes : additional sizes for the embedding layer
                    -> AttentionType.DOT : no entries
                    -> AttentionType.BILINEAR : one additional dimension for the matrix
                    -> AttentionType.ADDITIVE : one for the s vector, one for the matrices
        """
        super(Attention, self).__init__()
        self.embedding_size = embedding_size
        self.attention_type = attention_type
        self.attention_sizes = attention_sizes
        self.setup_attention()
        self.__attention_softmax__ = nn.Softmax(dim=-1)
        assert self.attention in [self.dot_attention, self.bilinear_attention, self.additive_attention]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        weights = self.attention(X)
        X = torch.mul(X, weights)
        # Reduce them by summing of all weighted token embeddings (-2 as we want to keep the last embedding dimension)
        X = torch.sum(X, dim=-2)
        return X

    def setup_attention(self):
        """ Will create all the matrices etc. for the  wanted attention type + sets self.attention to the appropriate function """
        if self.attention_type == AttentionType.DOT:
            assert len(self.attention_sizes) == 0
            self.attention = self.dot_attention
            self.s = nn.Linear(self.embedding_size, 1, bias=False)
        elif self.attention_type == AttentionType.BILINEAR:
            assert len(self.attention_sizes) == 1
            self.attention = self.bilinear_attention
            self.s = nn.Linear(self.attention_sizes[0], 1, bias=False)
            self.W = nn.Linear(self.embedding_size, self.attention_sizes[0], bias=False)
        elif self.attention_type == AttentionType.ADDITIVE:
            assert len(self.attention_sizes) == 2
            self.attention = self.additive_attention
            s = torch.empty((1, self.attention_sizes[0]), dtype=torch.float32, requires_grad=True)
            # We might want to change the initialization for s
            nn.init.xavier_normal_(s)
            self.s = nn.Parameter(s)
            print(self.s.shape)
            self.W1 = nn.Linear(self.embedding_size, self.attention_sizes[1], bias=False)
            self.W2 = nn.Linear(self.attention_sizes[0], self.attention_sizes[1], bias=False)
            self.v = nn.Linear(self.attention_sizes[1], 1, bias=False)
            self.tanh = nn.Tanh()

    def dot_attention(self, X: torch.Tensor) -> torch.Tensor:
        """ e_i = s^T h_i """   
        weights = self.s(X)
        # We need to normalize them:
        weights = self.__attention_softmax__(weights)
        return weights

    def bilinear_attention(self, X: torch.Tensor) -> torch.Tensor:
        """ e_i = s^T W h_i """
        weights = self.s(self.W(X))
        # We need to normalize them:
        weights = self.__attention_softmax__(weights)
        return weights

    def additive_attention(self, X: torch.Tensor) -> torch.Tensor:
        """ e_i = v^T tanh(W_1 h_i + W_2 s) """
        weights = self.W1(X) + self.W2(self.s).unsqueeze_(0)
        weights = self.tanh(weights)
        weights = self.v(weights)
        weights = self.__attention_softmax__(weights)
        return weights

class RNNEncoder(nn.Module):

    def __init__(self,
                 embedding_size: int = 200,
                 n_tokens: int = 50000,
                 layers = 1,
                 bidirectional = False,
                 activation: nn.Module = nn.ReLU(),
                 embedding_layer: nn.Module=nn.Embedding):
        super(RNNEncoder, self).__init__()
        self.embedding_size = embedding_size
        self._activation = activation
        
        # We might want to replace this embedding layer with something different
        self._embedding = embedding_layer
        self.layers = layers
        self.bidirectional = bidirectional

        self.gru1 = nn.GRU(self.embedding_size, self.embedding_size, num_layers=layers, bidirectional=self.bidirectional)
        self.gru_size = self.embedding_size*(2 if self.bidirectional else 1)
        self.reduction1 = nn.Linear(self.gru_size, self.embedding_size)
        self.reduction2 = nn.Linear(self.embedding_size, self.embedding_size)

        self._classification = nn.Sequential(
            nn.Linear(self.embedding_size, 1)
        )
        self.sig = nn.Sigmoid()

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        X = self._embedding(X)
        # Use the mask to exlude any embeddings of  padded vectors
        X = torch.mul(mask.unsqueeze(-1), X)

        # RNN over the tokens in a sentence
        X, _ = self.gru1(X)
        # Take last element as embedding for sentence
        X = self.reduction1(X[:,-1,:])
        X = self._activation(X)

        X = self.reduction2(X)
        X = self._activation(X)
        X = self._classification(X)

        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        return self.sig(E)

class CNNEncoder(nn.Module):

    def __init__(self,
                 embedding_size: int = 200,
                 n_tokens: int = 50000,
                 activation: nn.Module = nn.ReLU(),
                 embedding_layer: nn.Module=nn.Embedding):
        super(CNNEncoder, self).__init__()
        self.embedding_size = embedding_size
        
        self._activation = activation
        
        # We might want to replace this with something different
        self._embedding = embedding_layer
        self._emb_layers = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            self._activation,
        )

        self._conv = nn.Sequential(
            nn.Conv1d(self.embedding_size, int(self.embedding_size/2), kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(int(self.embedding_size/2), int(self.embedding_size/4), kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(int(self.embedding_size/4), 10, kernel_size=7, stride=1, padding=3)
        )

        self._classification = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10,1)
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        X = self._embedding(X)
        # Use the mask to exlude any embeddings of padded vectors
        X = torch.mul(mask.unsqueeze(-1), X)

        X = self._emb_layers(X)

        X = torch.transpose(X, -1, -2)

        X = self._conv(X)

        X = torch.amax(X, dim=-1)

        X = self._activation(X)
        X = self._classification(X)
        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        return self.sig(E) 

def save_model(model: nn.Module, path: Path):
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: Path, device: torch.device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
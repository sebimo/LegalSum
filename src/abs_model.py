# Model definitions for abstractive summarization
# I seperated the files to make the model implementations independent. This script will just define some nn.Module; all the utilities are still found in model.py
# This file might introduce some similar models to the ones defined in model.py, but as we want backward compability + replication we do not want to change the nn.Module in model.py

from enum import Enum
from typing import List, Dict

import torch
import torch.nn as nn

from .embedding import GloVe, Word2Vec

class AttentionType(Enum):
    DOT = 1
    BILINEAR = 2
    ADDITIVE = 3

class HierarchicalCrossEncoder(nn.Module):

    def __init__(self,
                 embedding_layer: nn.Module,
                 cross_sentence_layer: nn.Module,
                 embedding_size: int = 100,
                 cross_sentence_size: List[int] = [50, 50],
                 activation: nn.Module = nn.ReLU(),
                 attention: str="DOT"):
        super(HierarchicalCrossEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.cross_sentence_size = cross_sentence_size
        assert self.cross_sentence_size[0] == self.embedding_size
        
        assert attention in ["DOT", "BILINEAR", "ADDITIVE"]
        if attention == "DOT":
            self.attention = Attention(self.embedding_size, AttentionType.DOT)
        elif attention == "BILINEAR":
            self.attention = Attention(self.embedding_size, AttentionType.BILINEAR, attention_sizes=[100])
        elif attention == "ADDITIVE":
            self.attention = Attention(self.embedding_size, AttentionType.ADDITIVE, attention_sizes=[100, 100])
        else:
            raise ValueError("Attention type unknown: "+attention+"; Choose: DOT, BILINEAR, ADDITIVE")
        self.activation = activation
        
        self.embedding = embedding_layer
        self.emb_layers = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.embedding_size, self.embedding_size),
            self.activation,
        )

        self._cross_sentence_layer = cross_sentence_layer

        self._lin_layers = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.cross_sentence_size[1], self.cross_sentence_size[1]),
            nn.ReLU(),
            nn.Linear(self.cross_sentence_size[1], self.cross_sentence_size[1])
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        X = self._embedding(X)
        # Use the mask to exlude any embeddings of  padded vectors
        X = torch.mul(mask.unsqueeze(-1), X)

        X = self.emb_layers(X)

        X = self.attention(X)

        X = self.cross_sentence_layer(X)
        X = self.activation(X)

        X = self.lin_layers(X)

        return X

    # TODO add functionality to dynamically change the vectors + matrices used for attention based on the previously created tokens

    def get_name(self):
        return "HIER"+self._cross_sentence_layer.get_name()

class CNNCrossEncoder(nn.Module):

    def __init__(self,
                 embedding_layer: nn.Module,
                 cross_sentence_layer: nn.Module,
                 embedding_size: int = 100,
                 cross_sentence_size: List[int] = [50, 50],
                 n_tokens: int = 50000,
                 activation: nn.Module = nn.ReLU()):
        super(CNNCrossEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.cross_sentence_size = cross_sentence_size
        
        self.activation = activation
        
        # We might want to replace this with something different
        self.embedding = embedding_layer
        self.emb_layers = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            self.activation,
        )

        self.conv = nn.Sequential(
            nn.Conv1d(self.embedding_size, int(self.embedding_size/2), kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(int(self.embedding_size/2), int(self.embedding_size/4), kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(int(self.embedding_size/4), self.cross_sentence_size[0], kernel_size=7, stride=1, padding=3)
        )

        self.cross_sentence_layer = cross_sentence_layer

        self.lin_layers = nn.Sequential(
            nn.Linear(self.cross_sentence_size[1], self.cross_sentence_size[1]),
            nn.ReLU(),
            nn.Linear(self.cross_sentence_size[1],self.cross_sentence_size[1])
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        X = self.embedding(X)
        # Use the mask to exlude any embeddings of padded vectors
        X = torch.mul(mask.unsqueeze(-1), X)

        X = self.emb_layers(X)

        X = torch.transpose(X, -1, -2)

        X = self.conv(X)

        X = torch.amax(X, dim=-1)

        X = self.cross_sentence_layer(X)

        X = self.activation(X)
        X = self.lin_layers(X)
        
        return X

    def get_name(self):
        return "CNN"+self._cross_sentence_layer.get_name()


class CrossSentenceCNN(nn.Module):
    """ Module which will transfer information between nearby sentences -> decision about extraction should not be only based on sentence """
    
    def __init__(self,
                 cross_sentence_size: List[int]= [100, 100]):
        super(CrossSentenceCNN, self).__init__()
        self.cross_sentence_size = cross_sentence_size

        self.conv = nn.Sequential(
            nn.Conv1d(self.cross_sentence_size[0], self.cross_sentence_size[0], kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(self.cross_sentence_size[0], self.cross_sentence_size[1], kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(self.cross_sentence_size[1], self.cross_sentence_size[1], kernel_size=7, stride=1, padding=3)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.transpose(X[None,:,:], -1, -2)
        X = self.conv(X)
        X = torch.squeeze(X, dim=0)
        return torch.transpose(X, -1, -2)

    def get_name(self):
        return "_CNN"


class CrossSentenceRNN(nn.Module):
    """ Module which will transfer information between sentences via a bidirectional RNN + some linear layer to reduce the resulting dimensionality """

    def __init__(self,
                 cross_sentence_size: List[int]= [100, 100]):
        super(CrossSentenceRNN, self).__init__()
        self.cross_sentence_size = cross_sentence_size
        self.bidirectional = True
        self.directions = 2 if self.bidirectional else 1
        self.layers = 1

        self.gru = nn.GRU(self.cross_sentence_size[0], 
                           self.cross_sentence_size[1], 
                           num_layers=self.layers,
                           batch_first=True,
                           bidirectional=self.bidirectional)

        self.linear = nn.Linear(self.cross_sentence_size[1]*self.directions, self.cross_sentence_size[1])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X, _ = self.gru(X[None,:,:])
        X = self.linear(X)
        X = torch.squeeze(X, dim=0)
        return X

    def get_name(self):
        return "_RNN"

class Decoder(nn.Module):

    def __init__(self,
                 input_sizes: List[int]=[100, 100, 100],
                 output_size: int=50000):
        super(Decoder, self).__init__()
        self.embed_size = sum(input_sizes)
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.output_size)
        )

    def forward(facts: torch.Tensor, reasoning: torch.Tensor, previous: torch.Tensor):
        # For every tensor in previous: create one output!
        # We also might want to do this easier first and only produce word by word
        pass

class AbstractiveModel(nn.Module):

    def __init__(self,
                 fact_encoder: nn.Module,
                 reason_encoder: nn.Module,
                 prev_encoder: nn.Module,
                 decoder: nn.Module,
                 num_tokens: int=50000                 
                ):
        super(AbstractiveModel, self).__init__()
        self.num_tokens = num_tokens
        self.fact_encoder = fact_encoder
        self.reason_encoder = reason_encoder
        self.decoder = decoder
        self.prev_encoder = prev_encoder

    def forward(self, target, length, facts, facts_mask, reason, reason_mask):
        p_tensor = self.prev_encoder(target)

        f_tensor = self.fact_encoder(facts, facts_mask)
        r_tensor = self.reason_encoder(reason, reason_mask)

        return self.decoder(f_tensor, r_tensor, p_tensor)
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

        self.cross_sentence_layer = cross_sentence_layer

        self.lin_layers = nn.Sequential(
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
        # TODO we do not want sentence by sentence prediction
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
        # We do not want sentence by sentence prediction
        X, _ = self.gru(torch.unsqueeze(X, -3))
        X = self.linear(X[:,-1,:])
        return X

    def get_name(self):
        return "_RNN"

class RNNPrevEncoder(nn.Module):

    def __init__(self,
                 embedding_layer: nn.Module,
                 embedding_size: int = 200,
                 layers = 1,
                 bidirectional = False):
        super(RNNPrevEncoder, self).__init__()
        self.embedding_size = embedding_size
        
        # We might want to replace this embedding layer with something different
        self.embedding = embedding_layer
        self.layers = layers
        self.bidirectional = bidirectional

        self.gru1 = nn.GRU(self.embedding_size, self.embedding_size, num_layers=layers, bidirectional=self.bidirectional)
        self.gru_size = self.embedding_size*(2 if self.bidirectional else 1)
        self.reduction1 = nn.Linear(self.gru_size, self.embedding_size)
        self.reduction2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.activation = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.embedding(X)

        # RNN over the tokens in a sentence
        X, _ = self.gru1(X)
        # Take last element as embedding for the sequence
        X = self.reduction1(X[:,-1,:])
        X = self.activation(X)

        X = self.reduction2(X)
        X = self.activation(X)

        return X

    def get_name(self):
        return "RNN"

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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, facts: torch.Tensor, reasoning: torch.Tensor, previous: torch.Tensor):
        # We might need to use different stacking functions
        X = torch.hstack([facts, reasoning, previous])
        X = self.layers(X.squeeze(0))

        return self.softmax(X)

    def get_name(self):
        return "LIN"

class AbstractiveModel(nn.Module):

    def __init__(self,
                 body_encoder: nn.Module,
                 prev_encoder: nn.Module,
                 decoder: nn.Module,
                 prev_size: List[int]=[100]         
                ):
        super(AbstractiveModel, self).__init__()
        self.body_encoder = body_encoder
        self.decoder = decoder
        self.prev_encoder = prev_encoder
        self.prev_size = prev_size

    def forward(self, previous, facts, facts_mask, reason, reason_mask):
        # Will only produce one word
        if previous.shape[0] == 0:
            previous = torch.zeros([1,self.prev_size[0]], dtype=torch.long).to(previous.device)
        p_tensor = self.prev_encoder(previous)

        f_tensor = self.body_encoder(facts, facts_mask)
        r_tensor = self.body_encoder(reason, reason_mask)

        return self.decoder(f_tensor, r_tensor, p_tensor)

    def forward_sentence(self, target, length, facts, facts_mask, reason, reason_mask):
        # We get one target sentence and want to predict the masked words, i.e. the models can only see the past words
        pass

    def get_name(self):
        return self.body_encoder._get_name()+"_PRE_"+self.prev_encoder._get_name()+"_DEC_"+self.decoder._get_name()
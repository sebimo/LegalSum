# Model definitions for abstractive summarization
# I seperated the files to make the model implementations independent. This script will just define some nn.Module; all the utilities are still found in model.py
# This file might introduce some similar models to the ones defined in model.py, but as we want backward compability + replication we do not want to change the nn.Module in model.py
from pathlib import Path
from enum import Enum
from typing import List, Dict

import torch
import torch.nn as nn

from .embedding import GloVe, Word2Vec
from .model import load_model, AttentionType

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
            nn.Linear(self.embedding_size, self.embedding_size),
            self.activation,
        )

        self.cross_sentence_layer = cross_sentence_layer

        self.lin_layers = nn.Sequential(
            nn.Linear(self.cross_sentence_size[1], self.cross_sentence_size[1]),
            nn.ReLU(),
            nn.Linear(self.cross_sentence_size[1], self.cross_sentence_size[1])
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        X = self.embedding(X)
        # Use the mask to exlude any embeddings of  padded vectors
        X = torch.mul(mask.unsqueeze(-1), X)

        X = self.emb_layers(X)

        X = self.attention(X)

        X = self.cross_sentence_layer(X)
        X = self.activation(X)

        X = self.lin_layers(X)

        return X

    def get_name(self):
        return "HIER"+self.cross_sentence_layer.get_name()

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
        self.attention_softmax__ = nn.Softmax(dim=-1)
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
            self.W1 = nn.Linear(self.embedding_size, self.attention_sizes[1], bias=False)
            self.W2 = nn.Linear(self.attention_sizes[0], self.attention_sizes[1], bias=False)
            self.v = nn.Linear(self.attention_sizes[1], 1, bias=False)
            self.tanh = nn.Tanh()

    def dot_attention(self, X: torch.Tensor) -> torch.Tensor:
        """ e_i = s^T h_i """   
        weights = self.s(X)
        # We need to normalize them:
        weights = self.attention_softmax__(weights)
        return weights

    def bilinear_attention(self, X: torch.Tensor) -> torch.Tensor:
        """ e_i = s^T W h_i """
        weights = self.s(self.W(X))
        # We need to normalize them:
        weights = self.attention_softmax__(weights)
        return weights

    def additive_attention(self, X: torch.Tensor) -> torch.Tensor:
        """ e_i = v^T tanh(W_1 h_i + W_2 s) """
        weights = self.W1(X) + self.W2(self.s).unsqueeze_(0)
        weights = self.tanh(weights)
        weights = self.v(weights)
        weights = self.attention_softmax__(weights)
        return weights

    def get_name(self):
        return "ATT"

class GuidedHierarchicalCrossEncoder(nn.Module):

    def __init__(self,
                 embedding_layer: nn.Module,
                 cross_sentence_layer: nn.Module,
                 embedding_size: int = 100,
                 cross_sentence_size: List[int] = [50, 50],
                 activation: nn.Module = nn.ReLU(),
                 attention: str="DOT"):
        super(GuidedHierarchicalCrossEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.cross_sentence_size = cross_sentence_size
        assert self.cross_sentence_size[0] == self.embedding_size
        
        assert attention in ["DOT", "BILINEAR", "ADDITIVE"]
        if attention == "DOT":
            self.attention = GuidedAttention(self.embedding_size, AttentionType.DOT)
        elif attention == "BILINEAR":
            self.attention = GuidedAttention(self.embedding_size, AttentionType.BILINEAR, attention_sizes=[100])
        elif attention == "ADDITIVE":
            self.attention = GuidedAttention(self.embedding_size, AttentionType.ADDITIVE, attention_sizes=[100, 100])
        else:
            raise ValueError("Attention type unknown: "+attention+"; Choose: DOT, BILINEAR, ADDITIVE")
        self.activation = activation
        
        self.embedding = embedding_layer
        self.emb_layers = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            self.activation,
        )

        self.cross_sentence_layer = cross_sentence_layer

        self.lin_layers = nn.Sequential(
            nn.Linear(self.cross_sentence_size[1], self.cross_sentence_size[1]),
            nn.ReLU(),
            nn.Linear(self.cross_sentence_size[1], self.cross_sentence_size[1])
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor, prev: torch.Tensor, is_facts: bool) -> torch.Tensor:
        X = self.embedding(X)
        # Use the mask to exlude any embeddings of  padded vectors
        X = torch.mul(mask.unsqueeze(-1), X)

        X = self.emb_layers(X)

        X = self.attention(X, prev, is_facts)

        X = self.cross_sentence_layer(X)
        X = self.activation(X)

        X = self.lin_layers(X)

        return X

    def get_name(self):
        return "GHIER"+self.cross_sentence_layer.get_name()

class GuidedAttention(nn.Module):

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
        super(GuidedAttention, self).__init__()
        self.embedding_size = embedding_size
        self.attention_type = attention_type
        self.attention_sizes = attention_sizes
        self.setup_attention()
        self.attention_softmax__ = nn.Softmax(dim=-1)
        assert self.attention in [self.dot_attention, self.bilinear_attention, self.additive_attention]

    def forward(self, X: torch.Tensor, prev: torch.Tensor, is_facts: bool) -> torch.Tensor:
        weights = self.attention(X, prev, is_facts)
        X = torch.mul(X, weights)
        # Reduce them by summing of all weighted token embeddings (-2 as we want to keep the last embedding dimension)
        X = torch.sum(X, dim=-2)
        return X

    def setup_attention(self):
        """ Will create all the matrices etc. for the  wanted attention type + sets self.attention to the appropriate function """
        if self.attention_type == AttentionType.DOT:
            assert len(self.attention_sizes) == 0
            self.attention = self.dot_attention
            self.f_lin = nn.Linear(self.embedding_size, self.embedding_size)
            self.r_lin = nn.Linear(self.embedding_size, self.embedding_size)
        elif self.attention_type == AttentionType.BILINEAR:
            assert len(self.attention_sizes) == 1
            self.attention = self.bilinear_attention
            self.f_lin = nn.Linear(self.embedding_size, self.attention_sizes[0])
            self.r_lin = nn.Linear(self.embedding_size, self.attention_sizes[0])
            self.W = nn.Linear(self.embedding_size, self.attention_sizes[0], bias=False)
        elif self.attention_type == AttentionType.ADDITIVE:
            assert len(self.attention_sizes) == 2
            self.attention = self.additive_attention
            s = torch.empty((1, self.attention_sizes[0]), dtype=torch.float32, requires_grad=True)
            # We might want to change the initialization for s
            nn.init.xavier_normal_(s)
            self.s = nn.Parameter(s)
            self.W1 = nn.Linear(self.embedding_size, self.attention_sizes[1], bias=False)
            self.W2 = nn.Linear(self.attention_sizes[0], self.attention_sizes[1], bias=False)
            self.f_lin = nn.Linear(self.embedding_size, self.attention_sizes[1])
            self.r_lin = nn.Linear(self.embedding_size, self.attention_sizes[1])
            self.tanh = nn.Tanh()

    def dot_attention(self, X: torch.Tensor, prev: torch.Tensor, is_facts: bool) -> torch.Tensor:
        """ Depending on prev we want to change the search vector, i.e. s = l(prev)
            l in this case a a one layer linear function with bias; switched between f_lin and r_lin depending on is_facts (first one for facts, second one for reasoning) 
            In total:
                e_i = x_lin(prev)^T h_i 
        """   
        if is_facts:
            s = self.f_lin(prev)
        else:
            s = self.r_lin(prev)
        weights = (X @ s.T)
        # We need to normalize them:
        weights = self.attention_softmax__(weights)
        return weights

    def bilinear_attention(self, X: torch.Tensor, prev: torch.Tensor, is_facts: bool) -> torch.Tensor:
        """ e_i = x_lin(prev)^T W h_i """
        if is_facts:
            s = self.f_lin(prev)
        else:
            s = self.r_lin(prev)
        weights = (self.W(X) @ s.T)
        # We need to normalize them:
        weights = self.attention_softmax__(weights)
        return weights

    def additive_attention(self, X: torch.Tensor, prev: torch.Tensor, is_facts: bool) -> torch.Tensor:
        """ e_i = x_lin(prev)^T tanh(W_1 h_i + W_2 s) """
        weights = self.W1(X) + self.W2(self.s).unsqueeze_(0)
        weights = self.tanh(weights)

        if is_facts:
            v = self.f_lin(prev)
        else:
            v = self.r_lin(prev)
        weights = (weights @ v.T)
        weights = self.attention_softmax__(weights)
        return weights

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
        return "CNN"+self.cross_sentence_layer.get_name()


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
        X = torch.amax(torch.transpose(X, -1, -2), dim=-2)
        return X

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

class LRNNPrevEncoder(nn.Module):

    def __init__(self,
                 embedding_layer: nn.Module,
                 embedding_size: int = 200,
                 layers = 1,
                 bidirectional = False):
        super(LRNNPrevEncoder, self).__init__()
        self.embedding_size = embedding_size
        
        # We might want to replace this embedding layer with something different
        self.embedding = embedding_layer
        self.layers = layers
        self.bidirectional = bidirectional

        self.lin = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )

        self.gru1 = nn.GRU(self.embedding_size, self.embedding_size, num_layers=layers, bidirectional=self.bidirectional)
        self.gru_size = self.embedding_size*(2 if self.bidirectional else 1)
        self.reduction1 = nn.Linear(self.gru_size, self.embedding_size)
        self.reduction2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.activation = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.embedding(X)
        X = self.lin(X)
        # RNN over the tokens in a sentence
        X, _ = self.gru1(X)
        # Take last element as embedding for the sequence
        X = self.reduction1(X[:,-1,:])
        X = self.activation(X)

        X = self.reduction2(X)
        X = self.activation(X)

        return X

    def get_name(self):
        return "LRNN"

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

class TemplateDecoder(nn.Module):

    def __init__(self,
                 input_sizes: List[int]=[100, 100, 100, 100],
                 output_size: int=50000):
        super(TemplateDecoder, self).__init__()
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

    def forward(self, facts: torch.Tensor, reasoning: torch.Tensor, previous: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        # We might need to use different stacking functions
        X = torch.hstack([facts, reasoning, previous, template])
        X = self.layers(X.squeeze(0))

        return self.softmax(X)

    def get_name(self):
        return "TLIN"

class LDecoder(nn.Module):

    def __init__(self,
                 input_sizes: List[int]=[100, 100, 100],
                 output_size: int=50000):
        super(LDecoder, self).__init__()
        self.embed_size = sum(input_sizes)
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size),
            nn.ReLU(),
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
        return "LLIN"

class AttentionDecoder(nn.Module):

    def __init__(self,
                 input_sizes: List[int]=[100, 100, 100],
                 output_size: int=50000,
                 attention: str="DOT"):
        super(AttentionDecoder, self).__init__()
        self.embed_size = input_sizes[0]
        self.output_size = output_size
        # All the input_sizes need to be equal
        assert all(i == input_sizes[0] for i in input_sizes)
        assert attention in ["DOT", "BILINEAR", "ADDITIVE"]
        if attention == "DOT":
            self.attention = Attention(self.embed_size, AttentionType.DOT)
        elif attention == "BILINEAR":
            self.attention = Attention(self.embed_size, AttentionType.BILINEAR, attention_sizes=[100])
        elif attention == "ADDITIVE":
            self.attention = Attention(self.embed_size, AttentionType.ADDITIVE, attention_sizes=[100, 100])
        else:
            raise ValueError("Attention type unknown: "+attention+"; Choose: DOT, BILINEAR, ADDITIVE")

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
        X = torch.vstack([facts, reasoning, previous])
        X = self.attention(X)
        X = self.layers(X).squeeze(0)

        return self.softmax(X)

    def get_name(self):
        return "ADEC"

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
        p_tensor = self.prev_encoder(previous)

        f_tensor = self.body_encoder(facts, facts_mask)
        r_tensor = self.body_encoder(reason, reason_mask)

        return self.decoder(f_tensor, r_tensor, p_tensor)

    def forward_sentence(self, target, length, facts, facts_mask, reason, reason_mask):
        # We get one target sentence and want to predict the masked words, i.e. the models can only see the past words
        pass

    def get_name(self):
        return "NORM_ABS_"+self.body_encoder.get_name()+"_PRE_"+self.prev_encoder.get_name()+"_DEC_"+self.decoder.get_name()

class TemplateAbstractiveModel(nn.Module):

    def __init__(self,
                 body_encoder: nn.Module,
                 prev_encoder: nn.Module,
                 decoder: nn.Module,
                 template_encoder: nn.Module,
                 prev_size: List[int]=[100]         
                ):
        super(TemplateAbstractiveModel, self).__init__()
        self.body_encoder = body_encoder
        self.decoder = decoder
        self.prev_encoder = prev_encoder
        self.template_encoder = template_encoder
        self.prev_size = prev_size

    def forward(self, previous, template, facts, facts_mask, reason, reason_mask):
        # Will only produce one word
        t_tensor = self.template_encoder(template)
        p_tensor = self.prev_encoder(previous)

        # Here we give the template embedding instead!!!
        f_tensor = self.body_encoder(facts, facts_mask, t_tensor, is_facts=True)
        r_tensor = self.body_encoder(reason, reason_mask, t_tensor, is_facts=False)

        return self.decoder(f_tensor, r_tensor, p_tensor, t_tensor)

    def forward_sentence(self, target, length, facts, facts_mask, reason, reason_mask):
        # We get one target sentence and want to predict the masked words, i.e. the models can only see the past words
        pass

    def get_name(self):
        return "TEMPLATE_ABS_"+self.body_encoder.get_name()+"_PRE_"+self.prev_encoder.get_name()+"_TEMP_"+self.template_encoder.get_name()+"_DEC_"+self.decoder.get_name()

class AttentionTemplate(nn.Module):

    def __init__(self,
                embedding: nn.Module,
                embedding_size: int = 100):
        """ Arguments:
                - embedding
                - embedding_size  : size of the incoming token embeddings
        """
        super(AttentionTemplate, self).__init__()
        self.embedding = embedding
        self.embedding_size = embedding_size
        self.attention_type = AttentionType.DOT
        self.attention_sizes = []
        self.setup_attention()
        self.attention_softmax__ = nn.Softmax(dim=-1)
        assert self.attention in [self.dot_attention]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.embedding(X)
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

    def dot_attention(self, X: torch.Tensor) -> torch.Tensor:
        """ e_i = s^T h_i """   
        weights = self.s(X)
        # We need to normalize them:
        weights = self.attention_softmax__(weights)
        return weights

    def get_name(self):
        return "ATT"

class GuidedAbstractiveModel(nn.Module):
    # Similar to the model above, but the body encoder will get information about the region + the previous state

    def __init__(self,
                 body_encoder: nn.Module,
                 prev_encoder: nn.Module,
                 decoder: nn.Module,
                 prev_size: List[int]=[100]         
                ):
        super(GuidedAbstractiveModel, self).__init__()
        self.body_encoder = body_encoder
        self.decoder = decoder
        self.prev_encoder = prev_encoder
        self.prev_size = prev_size

    def forward(self, previous, facts, facts_mask, reason, reason_mask):
        # Will only produce one word
        p_tensor = self.prev_encoder(previous)

        f_tensor = self.body_encoder(facts, facts_mask, p_tensor, is_facts=True)
        r_tensor = self.body_encoder(reason, reason_mask, p_tensor, is_facts=False)

        return self.decoder(f_tensor, r_tensor, p_tensor)

    def forward_sentence(self, target, length, facts, facts_mask, reason, reason_mask):
        # We get one target sentence and want to predict the masked words, i.e. the models can only see the past words
        pass

    def get_name(self):
        return "GUIDED_ABS_"+self.body_encoder.get_name()+"_PRE_"+self.prev_encoder.get_name()+"_DEC_"+self.decoder.get_name()

def parse_abs_run_parameters(filename: str) -> Dict:
    split = filename.split("_")
    parameters = {}
    parameters["modelfile"] = "model/"+"_".join(split[:5])+".model"
    poss_params = set(["model", "lr", "abstractive", "embedding", "attention", "loss", "target"])
    param_name = []
    for s in split[5:]:
        if s in poss_params:
            param_name = [s]
        elif s == "type":
            param_name.append(s)
        elif param_name[0] == "model":
            if "model" in parameters:
                parameters["model"] += "_"+s
            else:
                parameters["model"] = s
        elif len(param_name) > 0:
            parameters["_".join(param_name)] = s
            param_name = []
        else:
            raise ValueError("Filename does not follow known format: "+filename)

    # Change the types accordingly
    if "lr" in parameters:
        parameters["lr"] = float(parameters["lr"])
    if "abstractive" in parameters:
        parameters["abstractive"] = True if parameters["abstractive"] == "1" else False
    if "attention" in parameters:
        if parameters["attention"] is None:
            del parameters["attention"]
    assert parameters["loss_type"] == "ABS"
    del parameters["loss_type"]
    assert parameters["abstractive"]
    del parameters["abstractive"]
    assert parameters["target"] == "NONE"
    del parameters["target"]
    return parameters

def reload_abs_model(parameters: Dict) -> nn.Module:
    """ Based on the parameters from above recreate the model """
    # Create the embeddings:
    embedding_size = 100
    cross_sentence_size = [100, 100]
    if parameters["embedding"] == "glove":
        embedding = GloVe(embedding_size=embedding_size, abstractive=True)
    elif parameters["embedding"] == "word2vec":
        embedding = Word2Vec(embedding_size=embedding_size, abstractive=True)
    
    abs_param, split = parameters["model"].split("_ABS_")
    body_param, split = split.split("_PRE_")
    prev_param, dec_param = split.split("_DEC_")

    # Create the body encoder:
    if body_param == "CNN_CNN":
        cross_sentence = CrossSentenceCNN(cross_sentence_size=cross_sentence_size)
        body_encoder = CNNCrossEncoder(embedding, 
                                       cross_sentence, 
                                       embedding_size=embedding_size,
                                       cross_sentence_size=cross_sentence_size)
    elif body_param == "HIER_RNN":
        cross_sentence = CrossSentenceRNN(cross_sentence_size=cross_sentence_size)
        body_encoder = HierarchicalCrossEncoder(embedding, 
                                                cross_sentence, 
                                                embedding_size=embedding_size,
                                                cross_sentence_size=cross_sentence_size,
                                                attention=parameters["attention"])
    elif body_param == "GHIER_RNN":
        cross_sentence = CrossSentenceRNN(cross_sentence_size=cross_sentence_size)
        body_encoder = GuidedHierarchicalCrossEncoder(embedding,
                                                    cross_sentence,
                                                    embedding_size=embedding_size,
                                                    cross_sentence_size=cross_sentence_size,
                                                    attention=parameters["attention"])
    else:
        raise ValueError("'"+body_param+"' is not a valid body encoder!")

    if prev_param == "RNN":
        prev_encoder = RNNPrevEncoder(embedding, embedding_size=embedding_size)
    else:
        raise ValueError("'"+prev_param+"' is not a valid previous encoder!")

    if dec_param == "LIN":
        decoder = Decoder(input_sizes=[embedding_size]+([cross_sentence_size[1]]*2),
                          output_size=len(embedding.get_word_mapping()["tok2id"]))
    elif dec_param == "ADEC":
        decoder = AttentionDecoder(input_sizes=[embedding_size]+([cross_sentence_size[1]]*2),
                                   output_size=len(embedding.get_word_mapping()["tok2id"]))
    elif dec_param == "TLIN":
        decoder = TemplateDecoder(input_sizes=[embedding_size]*4,output_size=len(embedding.get_word_mapping()["tok2id"]))
    else:
        raise ValueError("'"+dec_param+"' is not a valid decoder!")
    
    if abs_param == "NORM":
        model = AbstractiveModel(
            body_encoder,
            prev_encoder,
            decoder,
            prev_size=[embedding_size]
        )
    elif abs_param == "GUIDED":
        model = GuidedAbstractiveModel(
            body_encoder,
            prev_encoder,
            decoder,
            prev_size=[embedding_size]
        )
    elif abs_param == "TEMPLATE":
        temp_encoder = AttentionTemplate(embedding, embedding_size=embedding_size)
        model = TemplateAbstractiveModel(
            body_encoder,
            prev_encoder,
            decoder,
            temp_encoder,
            prev_size=[embedding_size]
        )
    else:
        raise ValueError("'"+abs_param+"' is not a valid abstractive model!")

    model = load_model(model, Path(parameters["modelfile"]), torch.device("cuda:0"))
    return model, embedding

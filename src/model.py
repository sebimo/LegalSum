# Contains all the models and its components for the summarization task
# We might want to compile the models to get faster training, is that possible?
# TODO Attention-layer for Encoding
import sys
from pathlib import Path
from enum import Enum
from typing import List, Dict
from collections import defaultdict

import torch
import torch.nn as nn

from .embedding import GloVe, Word2Vec

class AttentionType(Enum):
    DOT = 1
    BILINEAR = 2
    ADDITIVE = 3

# General difference between models: Encoder are just defined on a sentence per sentence basis, i.e. predictions are done without knowledge about other sentences
# -> in the CrossEncoder other sentences are to some extend taken into account for the sentence embedding

class HierarchicalEncoder(nn.Module):

    def __init__(self,
                 embedding_layer: nn.Module,
                 embedding_size: int = 200,
                 n_tokens: int = 50000,
                 activation: nn.Module = nn.ReLU(),
                 dropout: float=0.0,
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
        self.activation = activation
        
        # We might want to replace this with something different
        self.embedding = embedding_layer
        self.emb_layers = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            self.activation,
        )
        self.classification = nn.Sequential(
            nn.Linear(self.embedding_size, 1)
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        X = self.embedding(X)
        # Use the mask to exlude any embeddings of  padded vectors
        X = torch.mul(mask.unsqueeze(-1), X)

        X = self.emb_layers(X)

        X = self.attention(X)

        X = self.activation(X)
        X = self.classification(X)
        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        return self.sig(E)  

    def get_name(self):
        return "HIER"   

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
            print(self.s.shape)
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

# This model is not feasible/does not work as the optimization eventually will reach an errorous state
# This problem is a bit hard to debug as it can be related to the GPU or operating system
class RNNEncoder(nn.Module):

    def __init__(self,
                 embedding_layer: nn.Module,
                 embedding_size: int = 200,
                 n_tokens: int = 50000,
                 layers = 1,
                 bidirectional = False,
                 activation: nn.Module = nn.ReLU()):
        super(RNNEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.activation = activation
        
        # We might want to replace this embedding layer with something different
        self.embedding = embedding_layer
        self.layers = layers
        self.bidirectional = bidirectional

        self.gru1 = nn.GRU(self.embedding_size, self.embedding_size, num_layers=layers, bidirectional=self.bidirectional)
        self.gru_size = self.embedding_size*(2 if self.bidirectional else 1)
        self.reduction1 = nn.Linear(self.gru_size, self.embedding_size)
        self.reduction2 = nn.Linear(self.embedding_size, self.embedding_size)

        self.classification = nn.Sequential(
            nn.Linear(self.embedding_size, 1)
        )
        self.sig = nn.Sigmoid()

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        X = self.embedding(X)
        # Use the mask to exlude any embeddings of  padded vectors
        X = torch.mul(mask.unsqueeze(-1), X)

        # RNN over the tokens in a sentence
        X, _ = self.gru1(X)
        # Take last element as embedding for sentence
        X = self.reduction1(X[:,-1,:])
        X = self.activation(X)

        X = self.reduction2(X)
        X = self.activation(X)
        X = self.classification(X)

        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        return self.sig(E)

    def get_name(self):
        return "RNN"

class CNNEncoder(nn.Module):

    def __init__(self,
                 embedding_layer: nn.Module,
                 embedding_size: int = 200,
                 n_tokens: int = 50000,
                 activation: nn.Module = nn.ReLU()):
        super(CNNEncoder, self).__init__()
        self.embedding_size = embedding_size
        
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
            nn.Conv1d(int(self.embedding_size/4), 10, kernel_size=7, stride=1, padding=3)
        )

        self.classification = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10,1)
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

        X = self.activation(X)
        X = self.classification(X)
        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        return self.sig(E) 

    def get_name(self):
        return "CNN"

# The following models are supersets of the previously defined ones, as they aggregate

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

        self.classification = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.cross_sentence_size[1], self.cross_sentence_size[1]),
            nn.ReLU(),
            nn.Linear(self.cross_sentence_size[1], 1)
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

        X = self.classification(X)

        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        E = self.sig(E)
        return E 

    def get_name(self):
        return "HIER"+self.cross_sentence_layer.get_name()

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

        self.classification = nn.Sequential(
            nn.Linear(self.cross_sentence_size[1], 10),
            nn.ReLU(),
            nn.Linear(10,1)
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
        X = self.classification(X)
        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        return self.sig(E) 

    def get_name(self):
        return "CNN"+self.cross_sentence_layer.get_name()


class RNNCrossEncoder(nn.Module):

    def __init__(self,
                 embedding_layer: nn.Module,
                 cross_sentence_layer: nn.Module,
                 embedding_size: int = 100,
                 layers = 1,
                 cross_sentence_size: List[int] = [100, 100],
                 n_tokens: int = 50000,
                 activation: nn.Module = nn.ReLU()):
        super(RNNCrossEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.cross_sentence_size = cross_sentence_size
        
        self.activation = activation
        
        # We might want to replace this with something different
        self.embedding = embedding_layer
        self.emb_layers = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            self.activation,
        )

        self.layers = layers

        self.gru = nn.GRU(self.embedding_size, self.embedding_size, num_layers=layers, bidirectional=self.bidirectional)
        self.gru_size = self.embedding_size*(2 if self.bidirectional else 1)
        self.sentence_enc = nn.Sequential(
            nn.Linear(self.gru_size, self.embedding_size),
            self.activation,
            nn.Linear(self.embedding_size, self.cross_sentence_size[0]),
            self.activation
        )

        self.cross_sentence_layer = cross_sentence_layer

        self.classification = nn.Sequential(
            nn.Linear(self.cross_sentence_size[1], self.cross_sentence_size[1]),
            self.activation,
            nn.Linear(self.cross_sentence_size[1],1)
        )
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        X = self.embedding(X)
        # Use the mask to exlude any embeddings of padded vectors
        X = torch.mul(mask.unsqueeze(-1), X)

        X = self.emb_layers(X)

        X, _ = self.gru(X)
        # Take last element as embedding for sentence
        X = self.sentence_enc(X[:,-1,:])

        X = self.cross_sentence_layer(X)

        X = self.activation(X)
        X = self.classification(X)
        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        return self.sig(E) 

    def get_name(self):
        return "RNN"+self.cross_sentence_layer.get_name()


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

def save_model(model: nn.Module, path: Path):
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: Path, device: torch.device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def parse_run_parameters(filename: str) -> Dict:
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
        parameters["abstractive"] = True if parameters["abstractive"] == 0 else False
    if "attention" in parameters:
        if parameters["attention"] is None:
            del parameters["attention"]
    return parameters

def reload_model(parameters: Dict) -> nn.Module:
    """ Based on the parameters from above recreate the model """
    # Create the embeddings:
    embedding_size = 100
    if parameters["embedding"] == "glove":
        embeddings = GloVe(embedding_size=embedding_size)
    elif parameters["embedding"] == "word2vec":
        embeddings = Word2Vec(embedding_size=embedding_size)
    
    # Create the model
    if parameters["model"] == "HIER":
        model = HierarchicalEncoder(embedding_layer=embeddings, embedding_size=embedding_size, attention=parameters["attention"])
    elif parameters["model"] == "RNN":
        model = RNNEncoder(embedding_layer=embeddings, embedding_size=embedding_size)
    elif parameters["model"] == "CNN":
        model = CNNEncoder(embedding_layer=embeddings, embedding_size=embedding_size)
    elif parameters["model"] == "HIER_CNN":
        # Fixed dimensions, i.e. we do not change them here
        cross_sentence_size = [100, 100]
        cross_sentence = CrossSentenceCNN(cross_sentence_size)
        model = HierarchicalCrossEncoder(
                    embedding_layer=embeddings,
                    cross_sentence_layer=cross_sentence,
                    cross_sentence_size=cross_sentence_size
                ) 
    elif parameters["model"] == "HIER_RNN":
        # Fixed dimensions, i.e. we do not change them here
        cross_sentence_size = [100, 100]
        cross_sentence = CrossSentenceRNN(cross_sentence_size)
        model = HierarchicalCrossEncoder(
                    embedding_layer=embeddings,
                    cross_sentence_layer=cross_sentence,
                    cross_sentence_size=cross_sentence_size
                ) 
    elif parameters["model"] == "CNN_CNN":
        # Fixed dimensions, i.e. we do not change them here 
        cross_sentence_size = [100, 100]
        cross_sentence = CrossSentenceCNN(cross_sentence_size)
        model = CNNCrossEncoder(
                    embedding_layer=embeddings,
                    cross_sentence_layer=cross_sentence,
                    cross_sentence_size=cross_sentence_size
                ) 
    elif parameters["model"] == "CNN_RNN":
        # Fixed dimensions, i.e. we do not change them here
        cross_sentence_size = [100, 100]
        cross_sentence = CrossSentenceRNN(cross_sentence_size)
        model = CNNCrossEncoder(
                    embedding_layer=embeddings,
                    cross_sentence_layer=cross_sentence,
                    cross_sentence_size=cross_sentence_size
                ) 
    elif parameters["model"] == "RNN_RNN":
        cross_sentence_size = [100, 100]
        cross_sentence = CrossSentenceRNN(cross_sentence_size)
        model = RNNCrossEncoder(
                    embedding_layer=embeddings,
                    cross_sentence_layer=cross_sentence,
                    cross_sentence_size=cross_sentence_size
                ) 
    elif parameters["model"] == "RNN_CNN":
        cross_sentence_size = [100, 100]
        cross_sentence = CrossSentenceCNN(cross_sentence_size)
        model = RNNCrossEncoder(
                    embedding_layer=embeddings,
                    cross_sentence_layer=cross_sentence,
                    cross_sentence_size=cross_sentence_size
                ) 
    try:
        m = load_model(model, Path(parameters["modelfile"]), torch.device("cuda:0"))
    except RuntimeError:
        # Some models were trained before a model refactor, i.e. they have slightly different attribute names
        # To subvert this issue, we will parse the error message for the relevant attributes that are missing or are unexpected and then compare those two lists
        # If the unexpected list contains all the attributes from the missing list with "_" before them, we just set every attribute in the model object to that name
        # This only works as the refactor was aestatic, and no logic was changed. (i.e. model._embedding -> model.embedding)
        _, ex_obj, _ = sys.exc_info()
        values = str(ex_obj).split("Missing key(s) in state_dict: ")
        assert len(values) == 2
        values = values[1].strip()
        unexpected = values.split("Unexpected key(s) in state_dict: ")
        assert len(unexpected) == 2
        
        missing = list(map(lambda x: x.strip(".").strip("\""), unexpected[0].strip().split(", ")))
        unexpected = list(map(lambda x: x.strip(".").strip("\""), unexpected[1].strip().split(", ")))
        missing_set = set()
        for s in missing:
            missing_set.add(s.split(".")[0])
        unexpected_set = set()
        # If we have hierarchical models, we might need to change the sentence encoder as well
        sub_unexpected = defaultdict(set)
        for s in unexpected:
            attr = s.split(".")[0]
            unexpected_set.add(attr)
            sub_split = s.split("._")
            if len(sub_split) == 1:
                pass
            elif len(sub_split) == 2:
                # [0] as this is the attribute, "_" will already be removed by the split command
                sub_unexpected[attr].add(sub_split[1].split(".")[0])
            else:
                raise ValueError("Unknown attribute format:"+s)

        # We now check that all the changes are purely aestatic + rename the attributes
        assert len(missing_set) == len(unexpected_set)
        for x in missing_set:
            assert "_"+x in unexpected_set
            setattr(model, "_"+x, getattr(model, x))
            delattr(model, x)
        # In a hierarchical model the sentence encoders also have some aestatic changes
        for x in sub_unexpected:
            for s_x in sub_unexpected[x]:
                setattr(getattr(model, x), "_"+s_x, getattr(getattr(model, x), s_x))
                delattr(getattr(model, x), s_x)

        m = load_model(model, Path(parameters["modelfile"]), torch.device("cuda:0"))

        # We now have to revert all renames, as otherwise the methods will not work...
        
        # Change in inverted order, as otherwise the name references in sub_unexpected would not match
        for x in sub_unexpected:
            for s_x in sub_unexpected[x]:
                setattr(getattr(model, x), s_x, getattr(getattr(model, x), "_"+s_x))
                delattr(getattr(model, x), "_"+s_x)
        for x in missing_set:
            setattr(model, x, getattr(model, "_"+x))
            delattr(model, "_"+x)        

    model = m
    return model, embeddings

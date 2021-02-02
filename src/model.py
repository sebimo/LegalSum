# Contains all the models and its components for the summarization task
# We might want to compile the models to get faster training, is that possible?
# TODO Attention-layer for Encoding

import torch
import torch.nn as nn

class HierarchicalEncoder(nn.Module):

    def __init__(self,
                 embedding_size: int = 200,
                 n_tokens: int = 50000,
                 activation: nn.Module = nn.ReLU(),
                 dropout: float = 0.2,
                 embedding_layer: nn.Module=nn.Embedding):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = dropout
        self._activation = activation
        
        # We might want to replace this with something different
        self._embedding = embedding_layer(n_tokens, embedding_size)
        self._emb_layers = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),
            self._activation,
        )
        self._classification = nn.Sequential(
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )

        self._token_query = nn.Linear(self.embedding_size, 1, bias=False)
        self._sent_layers = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.embedding_size, self.embedding_size),
            self._activation,
        )
        self._sentence_query = nn.Linear(self.embedding_size, 1, bias=False)
        self._attention_softmax = nn.Softmax(dim=-1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._embedding(X)
        X = self._emb_layers(X)

        X = self.__get_token_attention__(X)
        X = self.__get_sentence_attention__(X)

        X = self._activation(X)
        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        y = self._classification(E)
        return y

    def __get_token_weights__(self, X: torch.Tensor) -> torch.Tensor:
        """ Returns the attention weights for calculating the sentence encoding 
        Arguments:
            X : torch.Tensor(batch_size, num_sentences = 200, num_tokens = 100, embedding_size)
        Returns:
            A : torch.Tensor(batch_size, num_sentences = 200, num_tokens = 100, 1) -- weight for each token indicating its importance
        """
        weights = self._token_query(X)
        # We need to normalize them:
        weights = self._attention_softmax(weights)
        return weights

    def __get_token_attention__(self, X: torch.Tensor) -> torch.Tensor:
        """ Calculates the weighted sum over the token embeddings 
        Arguments:
            X : torch.Tensor(batch_size, num_sentences = 200, num_tokens = 100, embedding_size)
        Returns:
            torch.Tensor(batch_size, num_sentences = 200, embedding_size)
        """
        # Get the token weights + apply them to all token embeddings
        weights = self.__get_token_weights__(X)
        X = torch.mul(X, weights)
        # Reduce them by summing of all weighted token embeddings (-2 as we want to keep the last embedding dimension)
        X = torch.sum(X, dim=-2)
        return X

    def __get_sentence_weights__(self, X: torch.Tensor) -> torch.Tensor:
        """ Returns the attention weights for calculating the sentence encoding 
        Arguments:
            X : torch.Tensor(batch_size, num_sentences = 200, embedding_size)
        Returns:
            A : torch.Tensor(batch_size, num_sentences = 200, 1) -- weight for each sentence indicating its importance
        """
        weights = self._sentence_query(X)
        # We need to normalize them
        weights = self._attention_softmax(weights)
        return weights

    def __get_sentence_attention__(self, X: torch.Tensor) -> torch.Tensor:
        """ Calculates the weighted sum over the sentence embeddings
        Arguments:
            X : torch.Tensor(batch_size, num_sentences = 200, embedding_size)
        Returns:
            torch.Tensor(batch_size, embedding_size)
        """
        # Get the sentence weights + apply them to all sentence embeddings
        weights = self.__get_sentence_weights__(X)
        X = torch.mul(X, weights)
        # Reduce the sentences by summing over all weighted sentence embeddings
        X = torch.sum(X, dim=-2)
        return X

class RNNEncoder(nn.Module):

    def __init__(self,
                 embedding_size: int = 200,
                 n_tokens: int = 50000,
                 layers = 2,
                 bidirectional = True,
                 activation: nn.Module = nn.ReLU(),
                 dropout: float = 0.2,
                 embedding_layer: nn.Module=nn.Embedding):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = dropout
        self._activation = activation
        
        # We might want to replace this embedding layer with something different
        self._embedding = embedding_layer(n_tokens, embedding_size)
        self.layers = layers
        self.bidirectional = bidirectional

        self.gru1 = nn.GRU(self.embedding_size, self.embedding_size, num_layers=layers, bidirectional=self.bidirectional)
        self.gru_size = self.embedding_size*self.layers*(2 if self.bidirectional else 1)
        self.reduction1 = nn.Linear(self.gru_size, self.embedding_size)
        self.reduction2 = nn.Linear(self.embedding_size, self.embedding_size)

        self._classification = nn.Sequential(
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._embedding(X)

        initial_shape = X.shape
        #X = X.reshape((-1, initial_shape[-2], self.embedding_size))

        # RNN over the tokens in a sentence
        X, _ = self.gru1(X)
        X = X.reshape(initial_shape[0], initial_shape[1], self.gru_size)
        # Max over -2 as, we want to keep the embedding dimensionality
        X = torch.mean(X, dim=-2, keepdim=True)
        X = torch.squeeze(X, dim=-2)

        X = self.reduction1(X)
        X = self._activation(X)

        X = self.reduction2(X)
        X = self._activation(X)

        return X

    def classify(self, E: torch.Tensor) -> torch.Tensor:
        y = self._classification(E)
        return y
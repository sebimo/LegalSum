import pytest
import torchtest
import torch
from torch.nn import Embedding

from ..model import HierarchicalEncoder, RNNEncoder, Attention, AttentionType, parse_run_parameters, reload_model, CNNCrossEncoder, CrossSentenceCNN
from ..embedding import GloVe

class TestHierarchicalEncoder:

    def setup(self, max_sent=200, max_token=100, embedding_size=200, n_tokens=500):
        self.max_sent = max_sent
        self.max_token = max_token
        self.embedding_size = embedding_size
        self.n_tokens = n_tokens
        self.embeddings = Embedding(num_embeddings=n_tokens, embedding_dim=embedding_size)
        self.model = HierarchicalEncoder(embedding_size=self.embedding_size, n_tokens=self.n_tokens, embedding_layer=self.embeddings)

    def setup_batch(self):
        self.setup()
        batch = torch.rand((self.max_sent, self.max_token, self.embedding_size))
        return batch

    def setup_indice_batch(self):
        self.setup(max_sent=10, max_token=10, embedding_size=200, n_tokens=100)
        batch = torch.zeros((self.max_sent, self.max_token), dtype=torch.long)
        for s in range(self.max_sent):
            for t in range(self.max_token):
                batch[s][t] = torch.randint(low=0, high=self.n_tokens, size=(1,))

        return batch

    def test_model(self):
        batch = self.setup_indice_batch()
        pred = self.model(batch, torch.ones(batch.shape[:-1], dtype=torch.bool))
        # We just want to check, if there is any error in the model (dimensions, ...)
        assert True


class TestRNNcuda:

    def setup(self, max_sent=200, max_token=100, embedding_size=200, n_tokens=500):
        self.max_sent = max_sent
        self.max_token = max_token
        self.embedding_size = embedding_size
        self.n_tokens = n_tokens
        self.embeddings = Embedding(num_embeddings=n_tokens, embedding_dim=embedding_size)
        self.model = HierarchicalEncoder(embedding_size=self.embedding_size, n_tokens=self.n_tokens, embedding_layer=self.embeddings).cuda()

    def setup_batch(self):
        self.setup()
        batch = torch.rand((self.max_sent, self.max_token, self.embedding_size))
        return batch

    def setup_indice_batch(self):
        self.setup(max_sent=10, max_token=10, embedding_size=200, n_tokens=100)
        batch = torch.zeros((self.max_sent, self.max_token), dtype=torch.long)
        for s in range(self.max_sent):
            for t in range(self.max_token):
                batch[s][t] = torch.randint(low=0, high=self.n_tokens, size=(1,))

        return batch

    def test_cuda(self):
        batch = self.setup_indice_batch().cuda()
        pred = self.model(batch, torch.ones(batch.shape[:-1], dtype=torch.bool).cuda())
        assert True


class TestRNNEncoder:

    def setup(self, max_sent=200, max_token=100, embedding_size=100, n_tokens=500):
        self.max_sent = max_sent
        self.max_token = max_token
        self.embedding_size = embedding_size
        self.n_tokens = n_tokens
        self.embedding = GloVe(embedding_size=self.embedding_size)
        self.model = RNNEncoder(embedding_size=self.embedding_size, n_tokens=self.n_tokens, embedding_layer=self.embedding, layers=1)

    def setup_batch(self):
        self.setup()
        batch = torch.rand((self.max_sent, self.max_token, self.embedding_size))
        return batch

    def setup_indice_batch(self):
        self.setup(max_sent=10, max_token=10, embedding_size=100, n_tokens=100)
        batch = torch.zeros((self.max_sent, self.max_token), dtype=torch.long)
        for s in range(self.max_sent):
            for t in range(self.max_token):
                batch[s][t] = torch.randint(low=0, high=self.n_tokens, size=(1,))

        return batch

    def test_forward(self):
        batch = self.setup_indice_batch()
        pred = self.model(batch, torch.ones(batch.shape[:-1], dtype=torch.bool))
        # Just checks for any bugs in dimensions etc.
        assert True


class TestHIERcuda:

    def setup(self, max_sent=200, max_token=100, embedding_size=100, n_tokens=500):
        self.max_sent = max_sent
        self.max_token = max_token
        self.embedding_size = embedding_size
        self.n_tokens = n_tokens
        self.embedding = GloVe(embedding_size=self.embedding_size)
        self.model = HierarchicalEncoder(embedding_size=self.embedding_size, n_tokens=self.n_tokens, embedding_layer=self.embedding).cuda()

    def setup_batch(self):
        self.setup()
        batch = torch.rand((self.max_sent, self.max_token, self.embedding_size))
        return batch

    def setup_indice_batch(self):
        self.setup(max_sent=10, max_token=10, embedding_size=100, n_tokens=100)
        batch = torch.zeros((self.max_sent, self.max_token), dtype=torch.long)
        for s in range(self.max_sent):
            for t in range(self.max_token):
                batch[s][t] = torch.randint(low=0, high=self.n_tokens, size=(1,))

        return batch

    def test_cuda(self):
        batch = self.setup_indice_batch().cuda()
        pred = self.model(batch, torch.ones(batch.shape[:-1], dtype=torch.bool).cuda())
        assert True


class TestAttention:

    def setup_batch(self):
        self.batch = torch.rand(1, 25, 200)

    def test_basics(self):
        self.setup_batch()
        # Tests, if we can instantiate the class
        a = Attention()
        r = a(self.batch)
        assert len(r.shape) == 2 and r.shape[1] == 200 and r.shape[0] == 1
        a = Attention(attention_type=AttentionType.BILINEAR, attention_sizes=[100])
        r = a(self.batch)
        assert len(r.shape) == 2 and r.shape[1] == 200 and r.shape[0] == 1
        a = Attention(attention_type=AttentionType.ADDITIVE, attention_sizes=[200, 100])
        r = a(self.batch)
        assert len(r.shape) == 2 and r.shape[1] == 200 and r.shape[0] == 1
        assert True

class TestModelLoading:

    def test_fileparsing(self):
        filename = "01_03_2021__112503_model_HIER_lr_0.003184952816752174_abstractive_0_embedding_glove_attention_BILINEAR"
        parameters = parse_run_parameters(filename)
        res = {
            "modelfile": "model/01_03_2021__112503.model",
            "model": "HIER",
            "lr": 0.003184952816752174,
            "abstractive": False,
            "embedding": "glove",
            "attention": "BILINEAR"
        }
        assert parameters == res

        filename = "09_03_2021__100342_model_RNN_lr_0.035803844608484355_abstractive_0_embedding_glove_attention_None_loss_type_BCE"
        parameters = parse_run_parameters(filename)
        res = {
            "modelfile": "model/09_03_2021__100342.model",
            "model": "RNN",
            "lr": 0.035803844608484355,
            "abstractive": False,
            "embedding": "glove",
            "loss_type": "BCE"
        }

        # Bigger models

        filename = "09_03_2021__100342_model_CNN_RNN_lr_0.035803844608484355_abstractive_0_embedding_glove_attention_None_loss_type_BCE"
        parameters = parse_run_parameters(filename)
        res = {
            "modelfile": "model/09_03_2021__100342.model",
            "model": "CNN_RNN",
            "lr": 0.035803844608484355,
            "abstractive": False,
            "embedding": "glove",
            "loss_type": "BCE"
        }

        filename = "09_03_2021__100342_model_HIER_RNN_lr_0.035803844608484355_abstractive_0_embedding_glove_attention_None_loss_type_BCE"
        parameters = parse_run_parameters(filename)
        res = {
            "modelfile": "model/09_03_2021__100342.model",
            "model": "HIER_RNN",
            "lr": 0.035803844608484355,
            "abstractive": False,
            "embedding": "glove",
            "loss_type": "BCE"
        }

    def model_loading(self):
        filename = "19_03_2021__204939_model_CNN_CNN_lr_0.006123480173958231_abstractive_0_embedding_glove_attention_DOT_loss_type_BCE_target_GREEDY"
        parameters = parse_run_parameters(filename)
        model, embedding = reload_model(parameters)
        assert isinstance(model, CNNCrossEncoder)
        assert isinstance(model.cross_sentence_layer, CrossSentenceCNN)
        assert isinstance(embedding, GloVe)
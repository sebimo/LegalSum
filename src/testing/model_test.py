import pytest
import torchtest
import torch

from ..model import HierarchicalEncoder, RNNEncoder

class TestHierarchicalEncoder:

    def setup(self, max_sent=200, max_token=100, embedding_size=200, n_tokens=500):
        self.max_sent = max_sent
        self.max_token = max_token
        self.embedding_size = embedding_size
        self.n_tokens = n_tokens
        self.model = HierarchicalEncoder(embedding_size=self.embedding_size, n_tokens=self.n_tokens)

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

    def test_token_weights_shape(self):
        batch = self.setup_batch()
        output = self.model.__get_token_weights__(batch)
        assert len(output.shape) == 3
        o_s, o_t, o_e = output.shape
        assert o_s == self.max_sent
        assert o_t == self.max_token
        assert o_e == 1

    def test_token_weights_softmax(self):
        batch = self.setup_batch()
        output = self.model.__get_token_weights__(batch)
        weight_sum = torch.sum(output, dim=-1)
        for i in range(self.max_token):
            assert weight_sum[0][i] == 1.0

    def test_sentence_weights_shape(self):
        self.setup()
        batch = torch.rand((self.max_sent, self.embedding_size))
        output = self.model.__get_sentence_weights__(batch)
        assert len(output.shape) == 2
        o_s, o_e = output.shape
        assert o_s == self.max_sent
        assert o_e == 1

    def test_sentence_weights_softmax(self):
        self.setup()
        batch = torch.rand((self.max_sent, self.embedding_size))
        output = self.model.__get_sentence_weights__(batch)
        weight_sum = torch.sum(output, dim=-1)
        print(weight_sum.shape)
        for i in range(self.max_sent):
            assert weight_sum[i] == 1.0

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
        self.model = HierarchicalEncoder(embedding_size=self.embedding_size, n_tokens=self.n_tokens).cuda()

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

    def setup(self, max_sent=200, max_token=100, embedding_size=200, n_tokens=500):
        self.max_sent = max_sent
        self.max_token = max_token
        self.embedding_size = embedding_size
        self.n_tokens = n_tokens
        self.model = RNNEncoder(embedding_size=self.embedding_size, n_tokens=self.n_tokens, layers=1)

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

    def test_forward(self):
        batch = self.setup_indice_batch()
        pred = self.model(batch, torch.ones(batch.shape[:-1], dtype=torch.bool))
        # Just checks for any bugs in dimensions etc.
        assert True


class TestHIERcuda:

    def setup(self, max_sent=200, max_token=100, embedding_size=200, n_tokens=500):
        self.max_sent = max_sent
        self.max_token = max_token
        self.embedding_size = embedding_size
        self.n_tokens = n_tokens
        self.model = RNNEncoder(embedding_size=self.embedding_size, n_tokens=self.n_tokens, layers=1).cuda()

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

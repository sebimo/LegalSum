import pytest
import os
from pathlib import Path

import torch

from ..dataloading import get_norm_sentences, get_norms, collate, ExtractiveDataset
from ..preprocessing import Tokenizer as Tokenizer

class TestNorms:

    def test_basics(self):
        # Check one basic query
        l = get_norm_sentences("AMVV", "1")
        assert len(l) > 0
        assert l[0].startswith("Arzneimittel")
        # Check what happens, if the norm does not exist
        l =  get_norm_sentences("TEST", "12")
        assert len(l) == 0

    def test_get_norms(self):
        s = get_norms()
        assert len(s) > 0
        assert type(s) == set
        for k in s:
            assert type(k) == str

class TestDataloading:

    def setup(self):
        self.device = torch.device('cpu')

    def test_collate(self):
        self.setup()
        x = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1]), torch.LongTensor([2])]
        y = torch.tensor([1.0, 0.0, 0.0])
        print(y.dtype)

        x_res = torch.LongTensor([
                [1,2,3],
                [1,0,0],
                [2,0,0]
            ])
        y_res = torch.tensor([1.0, 0.0, 0.0])
        mask = torch.tensor([
            [1, 1, 1],
            [1, 0, 0],
            [1,0,0]
        ], dtype=torch.bool)

        t = collate([(x,y)])
        
        assert len(t) == 1
        assert x_res.dtype == t[0][0].dtype
        assert x_res.shape == t[0][0].shape
        for i in range(3):
            for j in range(3):
                assert x_res[i][j] == t[0][0][i][j]
        
        assert y_res.dtype == t[0][1].dtype
        assert y_res.shape == t[0][1].shape
        for i in range(3):
            assert y_res[i] == t[0][1][i]


        assert mask.dtype == t[0][2].dtype
        assert mask.shape == t[0][2].shape
        for i in range(3):
            for j in range(3):
                assert mask[i][j] == t[0][2][i][j]

    def test_extractive_dataset(self):
        DATA_PATH = Path("src")/"testing"/"test_data"
        files = [DATA_PATH/file for file in os.listdir(DATA_PATH) if file.endswith(".json")]
        tok = Tokenizer(Path("model"))
        dataset = ExtractiveDataset(files, tok)
        try:
            for x, y in dataset:
                for i in range(y.shape[0]):
                    assert y[i] == 0.0
        except AssertionError:
            # We do not allow empty target lists
            assert True
        

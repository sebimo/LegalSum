import pytest

from ..training import Trainer

class TestTraining():

    def test_creation(self):
        t = Trainer(None, None, None, None, None)
        assert True
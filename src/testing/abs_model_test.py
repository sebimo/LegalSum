import pytest
import torchtest
import torch
from torch.nn import Embedding

from ..abs_model import reload_abs_model, parse_abs_run_parameters, AbstractiveModel, Decoder, RNNPrevEncoder, CrossSentenceRNN, CrossSentenceCNN, CNNCrossEncoder, HierarchicalCrossEncoder
from ..embedding import GloVe

class TestModelLoading:

    def test_fileparsing(self):
        filename = "28_03_2021__194202_model_CNN_CNN_PRE_RNN_DEC_LIN_lr_5.004752426924066e-06_abstractive_1_embedding_glove_attention_NONE_loss_type_ABS_target_NONE"
        parameters = parse_abs_run_parameters(filename)
        res = {
            "modelfile": "model/28_03_2021__194202.model",
            "model": "CNN_CNN_PRE_RNN_DEC_LIN",
            "lr": 5.004752426924066e-06,
            "embedding": "glove",
            "attention": "NONE"
        }
        assert parameters == res

    def test_model_loading(self):
        filename = "28_03_2021__194202_model_CNN_CNN_PRE_RNN_DEC_LIN_lr_5.004752426924066e-06_abstractive_1_embedding_glove_attention_NONE_loss_type_ABS_target_NONE"
        parameters = parse_abs_run_parameters(filename)
        model, embedding = reload_abs_model(parameters)
        assert isinstance(model, AbstractiveModel)
        assert isinstance(model.body_encoder, CNNCrossEncoder)
        assert isinstance(model.body_encoder.cross_sentence_layer, CrossSentenceCNN)
        assert isinstance(model.prev_encoder, RNNPrevEncoder)
        assert isinstance(model.decoder, Decoder)
        assert isinstance(embedding, GloVe)
        word_mapping = embedding.get_word_mapping()
        assert word_mapping["id2tok"][len(word_mapping["id2tok"])] == "<end>"
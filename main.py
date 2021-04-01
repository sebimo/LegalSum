import os
# Some package is overwriting the ctrl-c event, thus we need to stop this
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from pathlib import Path
from typing import Tuple

import numpy as np

from src.training import Trainer
from src.model import RNNEncoder, HierarchicalEncoder, CNNEncoder, CNNCrossEncoder, HierarchicalCrossEncoder, CrossSentenceCNN, CrossSentenceRNN, RNNCrossEncoder
from src.abs_model import AbstractiveModel, Decoder, RNNPrevEncoder, CrossSentenceRNN as AbsCrossSentenceRNN, CNNCrossEncoder as AbsCNNCrossEncoder, CrossSentenceCNN as AbsCrossSentenceCNN, HierarchicalCrossEncoder as AbsHierarchicalCrossEncoder
from src.abs_model import GuidedAbstractiveModel, GuidedHierarchicalCrossEncoder, LDecoder, AttentionDecoder, LRNNPrevEncoder
from src.embedding import Word2Vec, GloVe
from src.dataloading import ExtractiveDataset, get_train_files, get_val_files, LossType, transform_cutoff, get_greedy_train_files, get_greedy_val_files
from src.dataloading import AbstractiveDataset
from src.preprocessing import Tokenizer
from src.model_logging.logger import Logger as MyLogger

ABSTRACTIVE = True
TOKENIZER_PATH = Path("model")
LOGGER_ON = True
EMBEDDINGS = ["training", "word2vec", "glove"]

def start_extractive():
    logger = MyLogger(database=Path("logging")/"training.db")
    logger.setup_extractive()

    logger.set_status(LOGGER_ON)

    embedding_size = 100
    cross_sentence_size = [100, 100]
    attention = "NONE"

    num_epochs = 20

    # ATTENTION: We can include here a different mapping between tokens and ids, if we for example use word2vec
    tok = Tokenizer(TOKENIZER_PATH, normalize=True, mapping=embedding.get_word_mapping())
    for loss_type in [LossType.BCE]:
        loss_mapping = {
            LossType.BCE: "BCE",
            LossType.HAMM_HINGE: "HammHinge",
            LossType.HAMM_LOGI: "HammLogi",
            LossType.SUBSET_HINGE: "SubHinge",
            LossType.SUBSET_LOGI: "SubLogi"
        }

        database = "extractive.db"
        trainset = ExtractiveDataset(get_train_files(), tok, loss_type, database=database)
        valset = ExtractiveDataset(get_val_files(), tok, loss_type, database=database)

        for _ in range(2):
            embedding = GloVe(embedding_size=embedding_size)
            model = RNNEncoder(
                        embedding_layer=embedding,
                        embedding_size=embedding_size
                    )
            lr = random_log_lr()

            logger_params = {
                "model": model.get_name(),
                "lr": lr,
                "abstractive": 0,
                "embedding": embedding.get_name(), 
                "attention": attention,
                "loss_type": loss_mapping[loss_type],
                "target": "ONE"
            }
            logger.start_experiment(logger_params)

            trainer = Trainer(model, trainset, valset, logger, False)
            trainer.train(loss_type, epochs=num_epochs, lr=lr)
        
        for _ in range(1):
            embedding = GloVe(embedding_size=embedding_size)
            cross_sentence = CrossSentenceRNN(cross_sentence_size)
            model = RNNCrossEncoder(
                        embedding_layer=embedding,
                        cross_sentence_layer=cross_sentence,
                        cross_sentence_size=cross_sentence_size
                    )
            lr = random_log_lr()

            logger_params = {
                "model": model.get_name(),
                "lr": lr,
                "abstractive": 0,
                "embedding": embedding.get_name(), 
                "attention": attention,
                "loss_type": loss_mapping[loss_type],
                "target": "ONE"
            }
            logger.start_experiment(logger_params)

            trainer = Trainer(model, trainset, valset, logger, False)
            trainer.train(loss_type, epochs=num_epochs, lr=lr)

        for _ in range(2):
            embedding = GloVe(embedding_size=embedding_size)
            cross_sentence = CrossSentenceCNN(cross_sentence_size)
            model = RNNCrossEncoder(
                        embedding_layer=embedding,
                        cross_sentence_layer=cross_sentence,
                        cross_sentence_size=cross_sentence_size
                    )
            lr = random_log_lr()

            logger_params = {
                "model": model.get_name(),
                "lr": lr,
                "abstractive": 0,
                "embedding": embedding.get_name(), 
                "attention": attention,
                "loss_type": loss_mapping[loss_type],
                "target": "ONE"
            }
            logger.start_experiment(logger_params)

            trainer = Trainer(model, trainset, valset, logger, False)
            trainer.train(loss_type, epochs=num_epochs, lr=lr)

        database = "extractive_greedy.db"
        trainset = ExtractiveDataset(get_greedy_train_files(), tok, loss_type, database=database)
        valset = ExtractiveDataset(get_greedy_val_files(), tok, loss_type, database=database)

        for _ in range(2):
            embedding = GloVe(embedding_size=embedding_size)
            model = RNNEncoder(
                        embedding_layer=embedding,
                        embedding_size=embedding_size
                    )
            lr = random_log_lr()

            logger_params = {
                "model": model.get_name(),
                "lr": lr,
                "abstractive": 0,
                "embedding": embedding.get_name(), 
                "attention": attention,
                "loss_type": loss_mapping[loss_type],
                "target": "GREEDY"
            }
            logger.start_experiment(logger_params)

            trainer = Trainer(model, trainset, valset, logger, False)
            trainer.train(loss_type, epochs=num_epochs, lr=lr)

def start_abstractive():
    logger = MyLogger(database=Path("logging")/"training.db")
    logger.setup_abstractive()

    logger.set_status(LOGGER_ON)

    embedding_size = 100
    embedding = GloVe(embedding_size=embedding_size, abstractive=True)
    cross_sentence_size = [100, 100]
    attention = "DOT"

    num_epochs = 300

    # ATTENTION: We can include here a different mapping between tokens and ids, if we for example use word2vec
    tok = Tokenizer(TOKENIZER_PATH, normalize=True, mapping=embedding.get_word_mapping())
    loss_type = LossType.ABS
    loss_mapping = {
        LossType.ABS: "ABS"
    }

    trainset = AbstractiveDataset(get_train_files(), tok)
    valset = AbstractiveDataset(get_val_files(), tok)

    for _ in range(1):
        decoder = AttentionDecoder(input_sizes=[embedding_size]+([cross_sentence_size[1]]*2),
                                   output_size=tok.get_num_tokens())
        prev_encoder = RNNPrevEncoder(embedding, embedding_size=embedding_size)
        cross_sentence = AbsCrossSentenceRNN(cross_sentence_size=cross_sentence_size)
        body_encoder = AbsHierarchicalCrossEncoder(embedding, 
                                                      cross_sentence, 
                                                      embedding_size=embedding_size,
                                                      cross_sentence_size=cross_sentence_size,
                                                      attention=attention)
        
        model = AbstractiveModel(
            body_encoder,
            prev_encoder,
            decoder,
            prev_size=[embedding_size]
        )
        # See et al lr
        lr = 0.005

        logger_params = {
            "model": model.get_name(),
            "lr": lr,
            "abstractive": 1,
            "embedding": embedding.get_name(), 
            "attention": attention,
            "loss_type": loss_mapping[loss_type],
            "target": "NONE"
        }
        logger.start_experiment(logger_params)

        trainer = Trainer(model, trainset, valset, logger, True)
        trainer.train_abs(epochs=num_epochs, lr=lr, train_step_size=100, val_step_size=10, patience=30)

def random_log_lr(lr_range: Tuple[float, float]=(1e-3, 1e-6)) -> float:
    """ Will draw a random learning rate on a logarithmic scale, i.e. drawing the  lr from (1e-5, 1e-4) is as likely as from (1e-2,1e-1) """
    # convert bounds to logarithmic scale
    log_lower = np.log(lr_range[0])
    log_upper = np.log(lr_range[1])
    log_lr = np.random.uniform(low=log_lower, high=log_upper)
    lr = np.exp(log_lr)
    return lr

if __name__ == "__main__":
    if ABSTRACTIVE:
        start_abstractive()
    else:
        start_extractive()

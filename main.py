from pathlib import Path
from typing import Tuple

import numpy as np

from src.training import Trainer
from src.model import RNNEncoder, HierarchicalEncoder, CNNEncoder, CNNCrossEncoder, HierarchicalCrossEncoder, CrossSentenceCNN, CrossSentenceRNN
from src.embedding import Word2Vec, GloVe
from src.dataloading import ExtractiveDataset, get_train_files, get_val_files, LossType, transform_cutoff, get_greedy_train_files, get_greedy_val_files
from src.preprocessing import Tokenizer
from src.model_logging.logger import Logger as MyLogger

ABSTRACTIVE = False
TOKENIZER_PATH = Path("model")
LOGGER_ON = True
EMBEDDINGS = ["training", "word2vec", "glove"]

def start_extractive():
    logger = MyLogger(database=Path("logging")/"training.db")
    if ABSTRACTIVE:
        logger.setup_abstractive()
    else:
        logger.setup_extractive()

    logger.set_status(LOGGER_ON)

    embedding_size = 100
    embedding = GloVe(embedding_size=embedding_size)
    cross_sentence_size = [100, 100]
    attention = "DOT"
    database = "extractive_greedy.db"

    # ATTENTION: We can include here a different mapping between tokens and ids, if we for example use word2vec
    tok = Tokenizer(TOKENIZER_PATH, normalize=True, mapping=embedding.get_word_mapping())
    for loss_type in [LossType.HAMM_HINGE, LossType.HAMM_LOGI, LossType.SUBSET_HINGE, LossType.SUBSET_LOGI]:
        loss_mapping = {
            LossType.BCE: "BCE",
            LossType.HAMM_HINGE: "HammHinge",
            LossType.HAMM_LOGI: "HammLogi",
            LossType.SUBSET_HINGE: "SubHinge",
            LossType.SUBSET_LOGI: "SubLogi"
        }
        # Depending on the loss, we need different targets from the dataset
        trainset = ExtractiveDataset(get_greedy_train_files(), tok, loss_type, database=database)
        valset = ExtractiveDataset(get_greedy_val_files(), tok, loss_type, database=database)

        for _ in range(1):
            cross_sentence = CrossSentenceRNN(cross_sentence_size)
            model = (
                CNNCrossEncoder(
                    embedding_layer=embedding,
                    cross_sentence_layer=cross_sentence,
                    cross_sentence_size=cross_sentence_size
                ),
                "CNN_RNN"
            )
            lr = random_log_lr()

            logger_params = {
                "model": model[1],
                "lr": lr,
                "abstractive": 1 if ABSTRACTIVE else 0,
                "embedding": embedding.get_name(), 
                "attention": attention,
                "loss_type": loss_mapping[loss_type],
                "target": "GREEDY"
            }
            logger.start_experiment(logger_params)

            trainer = Trainer(model[0], trainset, valset, logger, ABSTRACTIVE)
            trainer.train(loss_type, epochs=50, lr=lr)

def random_log_lr(lr_range: Tuple[float, float]=(1e-3, 1e-5)) -> float:
    """ Will draw a random learning rate on a logarithmic scale, i.e. drawing the  lr from (1e-5, 1e-4) is as likely as from (1e-2,1e-1) """
    # convert bounds to logarithmic scale
    log_lower = np.log(lr_range[0])
    log_upper = np.log(lr_range[1])
    log_lr = np.random.uniform(low=log_lower, high=log_upper)
    lr = np.exp(log_lr)
    return lr

if __name__ == "__main__":
    start_extractive()

from pathlib import Path
from typing import Tuple

import numpy as np

from src.training import Trainer, LossType
from src.model import RNNEncoder, HierarchicalEncoder
from src.embedding import Word2Vec
from src.dataloading import ExtractiveDataset, get_train_files, get_val_files
from src.preprocessing import Tokenizer
from src.model_logging.logger import Logger as MyLogger

ABSTRACTIVE = False
TOKENIZER_PATH = Path("model")
LOGGER_ON = True

def start():
    logger = MyLogger(database=Path("logging")/"training.db")
    if ABSTRACTIVE:
        logger.setup_abstractive()
    else:
        logger.setup_extractive()

    logger.set_status(LOGGER_ON)

    # ATTENTION: We can include here a different mapping between tokens and ids, if we for example use word2vec
    tok = Tokenizer(TOKENIZER_PATH, normalize=True, mapping=None)
    # TODO reduced dataset
    # TODO REMOVE ALL FILES WITH EMPTY REASONING/FACTS OR NO TARGET
    trainset = ExtractiveDataset(get_train_files()[:10], tok)
    valset = ExtractiveDataset(get_val_files()[:10], tok)
    for _ in range(10):
        model = (
            HierarchicalEncoder(embedding_size=200, 
                                n_tokens=tok.get_num_tokens()),
            "HIER"
        )
        lr = random_log_lr()

        logger_params = {
            "model": model[1],
            "lr": lr,
            "abstractive": 1 if ABSTRACTIVE else 0
        }
        logger.start_experiment(logger_params)

        trainer = Trainer(model[0], trainset, valset, logger, ABSTRACTIVE)
        trainer.train(LossType.BCE, epochs=200, lr=lr)

def random_log_lr(lr_range: Tuple[float, float]=(1e-2, 1e-6)) -> float:
    """ Will draw a random learning rate on a logarithmic scale, i.e. drawing the  lr from (1e-5, 1e-4) is as likely as from (1e-2,1e-1) """
    # convert bounds to logarithmic scale
    log_lower = np.log(lr_range[0])
    log_upper = np.log(lr_range[1])
    log_lr = np.random.uniform(low=log_lower, high=log_upper)
    lr = np.exp(log_lr)
    return lr

if __name__ == "__main__":
    start()

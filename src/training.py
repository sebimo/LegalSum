# Main file used for all the machine-learning training
# Combines functionality of dataloading.py and model.py to train the models
from typing import Callable, Dict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SparseAdam
from torch.optim.lr_scheduler import MultiplicativeLR

from .dataloading import ExtractiveDataset, collate
from .evaluation import merge, finalize_statistic, calculate_confusion_matrix
from .model_logging.logger import Logger

class Trainer:

    def __init__(self, 
                 model: nn.Module, 
                 trainset: Dataset,
                 valset: Dataset,
                 logger: Logger,
                 abstractive: bool):
        
        self.logger = logger
        
        self.model = model
        self.trainset = trainset
        self.valset = valset

        # To keep track, if we are currently running over the training or evaluation set
        self.train_mode = True
        # We need this flag to differentiate between the different evaluation functions (for extractive we can look at F1 score, etc. instead of rouge)
        self.abstractive = abstractive
        # If we are using nn.Embeddings, we need to use a sparse optimizer i.e. optim.SparseAdam or optim.SGD (both work on CPU + CUDA)
        self.optim = None
        self.lr_scheduler = None
        # TODO where do we fit possible word models in here???

    def train(self,
              loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              epochs: int= 10,
              lr: float= 5e-4,
              patience: int=5,
              cuda: bool= True,
              workers: int=4):
        """ Starts the training iteration for the given model and dataset
            Params:
                epochs = number of iterations through the whole dataset
                lr = learning rate used for optimization of the model
                lr_scheduler = (if not None) used to update the current learning rate
                patience = after patience epochs with no improvement in validation performance stop the training
                cuda = if we want to use the GPU during training
                workers = number of processes used for data loading
        """
        self.dataloader = DataLoader(self.trainset, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=collate, prefetch_factor=10)
        # We have to see, if all those parameters are necessary for the valloader as well (based on resource consumption)
        self.valloader = DataLoader(self.valset, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate, prefetch_factor=10)
        print("Created dataloaders")

        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()
        print(self.model)

        self.loss = loss
        self.optim = SparseAdam(self.model.parameters(),lr=lr)
        self.lr_scheduler = MultiplicativeLR(self.optim, lambda e: 0.95)

        self.patience = patience
        best_loss = np.inf
        for it in tqdm(range(epochs), desc="Training:"):
            try:
                self.model.train()
                self.train_mode = True
                # Iterate over the trainset
                train_stats = self.__process_epoch__(self.dataloader)
                # TODO logging

                self.model.eval()
                self.train_mode = False
                # Iterate over the valset
                val_stats = self.__process_epoch__(self.valloader)
                # TODO logging
                if val_stats["loss"] < best_loss:
                    self.patience = patience
                    best_loss = val_stats["loss"]
                else:
                    self.patience -= 1
                    if self.patience <= 0:
                        break
            except KeyboardInterrupt:
                decision = input("Save training state? (y/n)")
                if decision.lower() == "y":
                    self.__save_training__()

        self.__finalize_training__() 


    def __process_epoch__(self, dataloader: DataLoader) -> Dict[str, float]:
        epoch_stats = {}
        for data in dataloader:
            stats = self.__process_batch__(data)
            # The loss or other important metrics are saved to the 
            epoch_stats = merge(epoch_stats, stats)
        return finalize_statistic(epoch_stats)

    def __process_batch__(self, data):   
        # reset gradients
        self.optim.zero_grad()

        # run model
        # TODO based on the dataloader this needs to be changed
        x, y = data
        if self.cuda():
            x = x.cuda()
            y = y.cuda()

        # TODO if we are in the abstractive case, this needs to change a bit
        pred = self.model.classify(self.model(x))
        
        # calculate loss
        loss: torch.Tensor = self.loss(pred, y)
        if self.train_mode:
            # backpropagation
            loss.backward()
            # We might want to do some gradient clipping
            # nn.utils.clip_grad_norm_(self.model.parameters(), 1e-2)
            self.optim.step()
            self.lr_scheduler.step()
        
        # Identify which statistics are pushed to the epoch method
        # evaluate batch
        np_loss = loss.cpu().detach().numpy()
        np_pred = pred.cpu().detach().numpy()
        np_true = y.cpu().detach().numpy()

        result = {
            "loss": np_loss[0]
        }
        if self.abstractive:
            # TODO calculate the rouge score form the 
            rouge = {
                "rouge-1": 0.0
            }
            result.update(rouge)
        else:
            np_pred = np.where(np_pred < 0.5, 0., 1.)
            result.update(calculate_confusion_matrix(np_true, np_pred))

        return result

    def __finalize_training__(self):
        raise NotImplementedError

    def __save_training__(self):
        raise NotImplementedError

    def resume_training(self):
        raise NotImplementedError
        

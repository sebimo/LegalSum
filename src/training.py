# Main file used for all the machine-learning training
# Combines functionality of dataloading.py and model.py to train the models
import os
from pathlib import Path
from typing import Callable, Dict, Tuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SparseAdam, Adam
from torch.optim.lr_scheduler import MultiplicativeLR

from .dataloading import ExtractiveDataset, collate, LossType
from .evaluation import merge, merge_epoch, finalize_statistic, calculate_confusion_matrix
from .model import save_model
from .model_logging.logger import Logger

# This threshold will be used to cut up the verdicts for extractive summarization, as it is not feasible to use all the sentences per verdict per train batch at the same time (at least for more complicated models)
SENT_NUM_THRESHOLD = 64

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

    def train(self,
              loss: LossType,
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
        if self.abstractive:
            raise NotImplementedError
        else:
            col = collate

        self.dataloader = DataLoader(self.trainset, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=col, prefetch_factor=10)
        # We have to see, if all those parameters are necessary for the valloader as well (based on resource consumption)
        self.valloader = DataLoader(self.valset, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=col, prefetch_factor=10)
        print("Created dataloaders")

        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()
        print(self.model)

        if loss == LossType.BCE:
            pos_weight = torch.tensor([10.0])
            if self.cuda:
                pos_weight = pos_weight.cuda()
            self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.optim = Adam(list(self.model.parameters()),lr=lr)
        self.lr_scheduler = MultiplicativeLR(self.optim, lambda e: 0.95)
        self.start_epoch = 0

        # Will update the variables model, optim, lr_scheduler; if there is a checkpoint in the model folder (model/checkpoint.pth)
        self.__load_checkpoint__()

        self.patience = patience
        best_loss = np.inf
        for it in tqdm(range(self.start_epoch, epochs), desc="Training"):
            try:
                self.model.train()
                self.train_mode = True

                log_dict = {}
                # Iterate over the trainset
                train_stats = self.__process_epoch__(self.dataloader)
                for k in train_stats:
                    log_dict["train_"+k] = train_stats[k]

                self.model.eval()
                self.train_mode = False
                # Iterate over the valset
                val_stats = self.__process_epoch__(self.valloader)
                for k in val_stats:
                    log_dict["val_"+k] = val_stats[k]
                
                for k in ["train_TP", "train_FP", "train_TN", "train_FN", "val_TP", "val_FP", "val_TN", "val_FN"]:
                #for k in ["val_TP", "val_FP", "val_TN", "val_FN"]:
                    del log_dict[k]
                
                # logging
                self.logger.log_epoch(log_dict)

                if val_stats["loss"] < best_loss:
                    self.patience = patience
                    best_loss = val_stats["loss"]
                    self.__save_model__()
                else:
                    self.patience -= 1
                    if self.patience <= 0:
                        break

            except KeyboardInterrupt:
                # We will use the KeyboardInterrupt, if we want to end/stop a training in between
                decision = input("Save training state? (y/n)")
                if decision.lower() == "y":
                    self.__save_training__(it)
                else:
                    # We have to remove any checkpoint files
                    checkpoint_path = Path("model")/"checkpoint.pth"
                    if checkpoint_path.is_file():
                        os.remove(checkpoint_path)

    def __process_epoch__(self, dataloader: DataLoader) -> Dict[str, float]:
        epoch_stats = {}
        for data in tqdm(dataloader, "Batch"):
            if self.abstractive:
                stats = self.__process_abstr_batch__(data)
            else:
                stats = self.__process_extr_batch__(data)
            # The loss or other important metrics are saved to the 
            epoch_stats = merge_epoch(epoch_stats, stats)
        return finalize_statistic(epoch_stats)

    def __process_abstr_batch__(self, data):
        # reset gradients
        self.optim.zero_grad()

        # run model
        # TODO based on the dataloader this needs to be changed
        raise NotImplementedError
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
        
        # TODO calculate the rouge score form the 
        rouge = {
            "rouge-1": 0.0
        }
        result.update(rouge)

        return result
    
    def __process_extr_batch__(self, data):   
        batch_stats = {}
        # Each entry in data is one document containing an abitrary number of sentences
        for x_b, y_b, mask_b in data:
            # reset gradients
            self.optim.zero_grad()

            # cut up inputs, which are to big for processing, i.e. the num of sentences not above threshold
            # Alternatives to this would be to change the dataloading process and only serve chunks from verdicts with size SENT_NUM_THRESHOLD,
            # but this would complicate the dataloading a lot more than simple splitting of tensors here
            for x, y, mask in self.__minibatch__(x_b, y_b, mask_b):
            
                # run model
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    mask = mask.cuda()

                pred = self.model(x, mask)
                
                # calculate loss
                loss: torch.Tensor = self.loss(pred, y[:,None])
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
                result = {
                    "loss": np_loss
                }

                pred = self.model.classify(pred)

                np_pred = pred.cpu().detach().numpy().squeeze(axis=1)
                np_true = y.cpu().detach().numpy()
                np_pred = np.where(np_pred < 0.5, 0., 1.)
                assert np_pred.shape[0] > 0, "Predictions empty"+str(np_pred.shape)
                assert np_true.shape[0] > 0, "Targets empty"+str(np_true.shape)
                assert np_true.shape == np_pred.shape, "Shape mismatch"+str(np_true.shape) +" != "+str(np_pred.shape)
                result.update(calculate_confusion_matrix(np_true, np_pred))

                batch_stats = merge(batch_stats, result)

        return batch_stats

    def __minibatch__(self, x_b: torch.Tensor, y_b: torch.Tensor, mask_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ We need to cut up the verdicts in some cases, as the documents have to many sentences for feasible GPU processing, i.e. this function yields all the minibatches formed from a batch which have a capped number of sentences """
        x_splits = torch.split(x_b, SENT_NUM_THRESHOLD)
        y_splits = torch.split(y_b, SENT_NUM_THRESHOLD)
        mask_splits = torch.split(mask_b, SENT_NUM_THRESHOLD)
        assert len(x_splits) == len(y_splits) == len(mask_splits)
        for x, y, mask in zip(x_splits, y_splits, mask_splits):
            yield x, y, mask

    def __save_model__(self):
        model_file = Path("model")/(self.logger.experiment + ".model")
        save_model(self.model, model_file)

    def __save_training__(self, it: int):
        checkpoint = {
            'epoch': it,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'lr_scheduler': self.lr_scheduler
        }
        torch.save(checkpoint, Path("model")/"checkpoint.pth")

    def __load_checkpoint__(self):
        checkpoint_path = Path("model")/"checkpoint.pth"
        if checkpoint_path.is_file():
            print("Resume training")
            checkpoint = torch.load()
            self.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model"])
            self.optim.load_state_dict(checkpoint["optim"])
            self.lr_scheduler = checkpoint["lr_scheduler"]

        

# Main file used for all the machine-learning training
# Combines functionality of dataloading.py and model.py to train the models

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:

    def __init__(self, 
                 model: nn.Module=None, 
                 dataloader: DataLoader=None):
        self.model = model
        self.dataloader = dataloader
        # To keep track, if we are currently running over the training or evaluation set
        self.train_mode = True
        #self.evaluation -> to get get costum evaluation based on the evaluation.py script
        #self.optim -> which optimizer to use during training
        # TODO where do we fit possible word models in here???

    def train(self):
        pass

    def process_epoch(self):
        for data in self.dataloader:
            self.process_batch(data)

    def process_batch(self, data):
        # reset gradients
        # TODO run model
        # calculate loss
        if self.train_mode:
            # backpropagation
            pass
        # evaluate batch
        

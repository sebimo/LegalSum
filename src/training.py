# Main file used for all the machine-learning training
# Combines functionality of dataloading.py and model.py to train the models
import os
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Set
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SparseAdam, Adam, SGD, Adagrad
from torch.optim.lr_scheduler import MultiplicativeLR

from .dataloading import ExtractiveDataset, collate, collate_abs, collate_abs_long, LossType
from .preprocessing import Tokenizer
from .evaluation import merge, merge_epoch, finalize_statistic, calculate_confusion_matrix, evaluate
from .model import save_model
from .loss import HammingLossHinge, HammingLossLogistic, SubsetLossHinge, SubsetLossLogistic, CombinedLoss
from .model_logging.logger import Logger

# This threshold will be used to cut up the verdicts for extractive summarization, as it is not feasible to use all the sentences per verdict per train batch at the same time (at least for more complicated models)
SENT_NUM_THRESHOLD = 300

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
        assert not self.abstractive
        col = collate

        self.dataloader = DataLoader(self.trainset, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=col, prefetch_factor=10)
        # We have to see, if all those parameters are necessary for the valloader as well (based on resource consumption)
        self.valloader = DataLoader(self.valset, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=col, prefetch_factor=10)
        print("Created dataloaders")

        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()
        print(self.model)

        self.loss_type = loss

        if self.loss_type == LossType.BCE:
            pos_weight = torch.tensor([40.0]).cuda()
            self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.loss_type == LossType.HAMM_HINGE:
            self.loss = HammingLossHinge
        elif self.loss_type == LossType.HAMM_LOGI:
            self.loss = HammingLossLogistic
        elif self.loss_type == LossType.SUBSET_HINGE:
            self.loss = SubsetLossHinge
        elif self.loss_type == LossType.SUBSET_LOGI:
            self.loss = SubsetLossLogistic

        if self.loss_type != LossType.BCE:
            self.loss = CombinedLoss([self.loss, nn.BCELoss()], [0.55, 0.45])

        self.model_func = self.model.classify if self.loss_type != LossType.BCE else lambda x: x


        self.optim = Adam(self.model.parameters(),lr=lr)
        self.lr_scheduler = MultiplicativeLR(self.optim, _mult_lr_factor_)
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
                train_stats = self.__process_ext_epoch__(self.dataloader)
                for k in train_stats:
                    log_dict["train_"+k] = train_stats[k]

                self.model.eval()
                self.train_mode = False
                # Iterate over the valset
                val_stats = self.__process_ext_epoch__(self.valloader)
                for k in val_stats:
                    log_dict["val_"+k] = val_stats[k]
                
                for k in ["train_TP", "train_FP", "train_TN", "train_FN", "val_TP", "val_FP", "val_TN", "val_FN"]:
                    if k in log_dict:
                        del log_dict[k]
                
                # logging
                self.logger.log_epoch(log_dict)

                if val_stats["loss"] < best_loss:
                    self.patience = patience
                    best_loss = val_stats["loss"]
                    if self.logger.on:
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

    def train_abs(self,
              epochs: int= 30,
              lr: float= 5e-4,
              patience: int=7,
              cuda: bool= True,
              workers: int=1,
              train_step_size: int=1000,
              val_step_size: int=100):
        """ Starts the training iteration for the given model and dataset
            Params:
                epochs = number of iterations through the whole dataset
                lr = learning rate used for optimization of the model
                lr_scheduler = (if not None) used to update the current learning rate
                patience = after patience epochs with no improvement in validation performance stop the training
                cuda = if we want to use the GPU during training
                workers = number of processes used for data loading
                train_step_size = how many batches to process before evaluation
                eval_step_size = how many verdicts to evaluate
        """
        assert self.abstractive
        col = collate_abs_long

        #self.dataloader = DataLoader(self.trainset, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=col, prefetch_factor=4)
        self.dataloader = DataLoader(self.trainset, shuffle=True, pin_memory=True, collate_fn=col, batch_size=10)
        # We have to see, if all those parameters are necessary for the valloader as well (based on resource consumption)
        #self.valloader = DataLoader(self.valset, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=col, prefetch_factor=4)
        self.valloader = DataLoader(self.valset, shuffle=False, pin_memory=True, collate_fn=col, batch_size=10)
        print("Created dataloaders")

        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()
        print(self.model)

        self.optim = Adam(self.model.parameters(),lr=lr)
        self.lr_scheduler = MultiplicativeLR(self.optim, _mult_lr_factor_)
        self.start_epoch = 0

        # Will update the variables model, optim, lr_scheduler; if there is a checkpoint in the model folder (model/checkpoint.pth)
        self.__load_checkpoint__()

        self.patience = patience
        best_loss = np.inf
        try:
            for it in tqdm(range(self.start_epoch, epochs), desc="Training"):
                self.model.train()
                self.train_mode = True

                log_dict = {}
                # Iterate over the trainset
                epoch_stats = {}
                for i, data in tqdm(enumerate(self.__train_iter__()), desc="TrainSteps", total=train_step_size):
                    stats = self.__process_abstr_batch__(data)
                    # The loss or other important metrics are saved to the 
                    epoch_stats = merge_epoch(epoch_stats, stats)
                    if i >= train_step_size:
                        break
                self.lr_scheduler.step()
            
                train_stats =  finalize_statistic(epoch_stats)
                for k in train_stats:
                    log_dict["train_"+k] = train_stats[k]

                self.model.eval()
                self.train_mode = False
                # Iterate over the valset
                val_epoch_stats = {}
                for i, data in tqdm(enumerate(self.__val_iter__()), desc="ValSteps", total=val_step_size):
                    stats = self.__process_abstr_batch__(data)
                    # The loss or other important metrics are saved to the 
                    val_epoch_stats = merge_epoch(val_epoch_stats, stats)
                    if i >= val_step_size:
                        break
                
                val_stats = finalize_statistic(val_epoch_stats)
                for k in val_stats:
                    log_dict["val_"+k] = val_stats[k]
                
                # logging
                self.logger.log_epoch(log_dict)

                if val_stats["loss"] < best_loss:
                    self.patience = patience
                    best_loss = val_stats["loss"]
                    if self.logger.on:
                        self.__save_model__()
                else:
                    self.patience -= 1
                    if self.patience <= 0:
                        break
                
                # We have some GPU memory leak, but up to now it was not possible to find is. Clearing the GPU cache mitigates this problem though.
                torch.cuda.empty_cache()
        except KeyboardInterrupt:
            # We will use the KeyboardInterrupt, if we want to end/stop a training in between
            print("INTERUPTING IS NOT DECENT!11!1!")
            decision = input("Save training state? (y/n)")
            if decision.lower() == "y":
                self.__save_training__(it)
            else:
                # We have to remove any checkpoint files
                checkpoint_path = Path("model")/"checkpoint.pth"
                if checkpoint_path.is_file():
                    os.remove(checkpoint_path)

    def __val_iter__(self):
        # For the abstractive methods, we do not want to evaluate after all the verdicts. In order to do so, we need to seperate the data loading with the training
        # We want to be able to continously load from the valloader
        while True:
            for data in self.valloader:
                yield data

    def __train_iter__(self):
        # For the abstractive methods, we do not want to evaluate after all the verdicts. In order to do so, we need to seperate the data loading with the training
        # We want to be able to continously load from the trainloader
        while True:
            for data in self.dataloader:
                yield data

    def __process_ext_epoch__(self, dataloader: DataLoader) -> Dict[str, float]:
        epoch_stats = {}
        for data in tqdm(dataloader, "Batch"):
            stats = self.__process_extr_batch__(data)
            # The loss or other important metrics are saved to the 
            epoch_stats = merge_epoch(epoch_stats, stats)
        self.lr_scheduler.step()
        return finalize_statistic(epoch_stats)

    def __process_abs_epoch__(self, dataloader: DataLoader) -> Dict[str, float]:
        epoch_stats = {}
        for data in tqdm(dataloader, "Batch"):
            stats = self.__process_abstr_batch__(data)
            # The loss or other important metrics are saved to the 
            epoch_stats = merge_epoch(epoch_stats, stats)
        self.lr_scheduler.step()
        return finalize_statistic(epoch_stats)

    def __process_abstr_batch__(self, data):
        batch_stats = {}
        # Each entry in data is one document containing an abitrary number of sentences
        self.optim.zero_grad()
        for batch in data:
            tar, leng, facts, facts_mask, reason, reason_mask = batch
            # target, lengths, facts, fact mask, reasoning, reasoning_mask, norms
            
            # run model
            if self.cuda:
                tar = tar.cuda()
                facts = facts.cuda()
                facts_mask = facts_mask.cuda()
                reason = reason.cuda()
                reason_mask = reason_mask.cuda()

            # Sentence for sentence generation
            gpu_acc_loss = torch.zeros([1]).cuda()
            gpu_acc_prob = torch.zeros([1]).cuda()
            num_words = 0
            for t, l in self.__abs_minibatch__(tar, leng):
                num_words += l[0].item()-1
                # In this training case, we will produce the output word for word, i.e. we need to mask all words up to the current one
                # We start at index one as the first word is an <unk> token
                for i in range(1, l[0].item()):
                    pred = self.model(t[:,:i], facts, facts_mask, reason, reason_mask)
                    tar_ind = t[:,i]
                    #max_ind = torch.argmax(pred)
                    # Accumulate loss over multiple batches/documents
                    loss = (-torch.log(pred[tar_ind]+1e-12))/l[0]
                    #print(tar_ind, pred[tar_ind], "MAX:", max_ind, pred[max_ind], t[:, :i])
                    if self.train_mode:
                        # backpropagation; split up backprop and optim step, as we otherwise need to keep a gradient model for each word until step
                        loss.backward()
                    gpu_acc_loss += loss.item()
                    gpu_acc_prob += pred[tar_ind].item()

            
            # Identify which statistics are pushed to the epoch method
            # evaluate batch
            result = {
                "loss": gpu_acc_loss.cpu().item(),
                "probability": gpu_acc_prob.cpu().item()/num_words * 100
            }

            batch_stats = merge(batch_stats, result)

        if self.train_mode:
            # We might want to do some gradient clipping
            #nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optim.step()

        return batch_stats

    def __process_abstr_sentence_batch__(self, data):
        # Same abstractive training as above, but the training is done on a sentence level, i.e. the model needs to do the heavy lifting for the predictions
        batch_stats = {}
        # Each entry in data is one document containing an abitrary number of sentences
        for batch in data:
            for tar, leng, facts, facts_mask, reason, reason_mask in batch:
                # target, lengths, facts, fact mask, reasoning, reasoning_mask, norms
                self.optim.zero_grad()
                
                if self.cuda:
                    tar = tar.cuda()
                    leng = leng.cuda()
                    facts = facts.cuda()
                    facts_mask = facts_mask.cuda()
                    reason = reason.cuda()
                    reason_mask = reason_mask.cuda()

                # TODO Change this as above!!!
                loss = torch.zeros(1)
                # Sentence for sentence generation
                for t, l in self.__abs_minibatch__(tar, leng):
                    # In this training case, we will produce the output word for word, i.e. we need to mask all words up to the current one
                    pred = self.model.forward_sentence(t, l, facts, facts_mask, reason, reason_mask)      
                    # Accumulate loss over multiple batches/documents
                    loss -= torch.sum(torch.log(pred+1e-8))
                if self.train_mode:
                    # backpropagation
                    loss.backward()
                    # We might want to do some gradient clipping
                    #nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    self.optim.step()
                    
                
                # Identify which statistics are pushed to the epoch method
                # evaluate batch
                np_loss = loss.cpu().detach().numpy()
                result = {
                    "loss": np_loss
                }

                batch_stats = merge(batch_stats, result)

        return batch_stats
    
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
                pred = self.model_func(pred)
                
                # calculate loss
                # TODO accumulate loss over multiple batches/documents
                loss: torch.Tensor = self.loss(pred, y[:,None])
                if self.train_mode:
                    # backpropagation
                    loss.backward()
                    # We might want to do some gradient clipping
                    #nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    self.optim.step()
                
                # Identify which statistics are pushed to the epoch method
                # evaluate batch
                np_loss = loss.cpu().detach().numpy()
                result = {
                    "loss": np_loss
                }

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

    def __abs_minibatch__(self, target: torch.Tensor, length: torch.Tensor):
        for t, l in zip(torch.split(target, 1), torch.split(length, 1)):
            yield t, l

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
        print("Checking previous checkpoint...")
        checkpoint_path = Path("model")/"checkpoint.pth"
        if checkpoint_path.is_file():
            print("Resume training!")
            checkpoint = torch.load(checkpoint_path)
            self.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model"])
            self.optim.load_state_dict(checkpoint["optim"])
            self.lr_scheduler = checkpoint["lr_scheduler"]

# Putting the evaluation code here is obviously debatable, but its the easiest way without changing the import for any other script; every import we need is given here      
def evaluate_ext_model(model: nn.Module, embedding: nn.Module, verdicts: List[str], max_sents: int=3, equal_length: bool=True) -> List[Dict[str, float]]:
    """ Will evaluate a model on all the verdicts given. Some additional parameters are possible to reduce the length 
        Parameters:
            model -- the given NN used for the predictions
            embedding -- the embeddings used for the token <-> id mapping
            verdicts -- the paths to the verdicts which shall be evaluated
            max_sents -- maximum number of sentences per created summarization
            equal_length -- if the number of sentences from the created summary need to match the number of sentences from the Ground Truth; overrides max_sents
    """
    # Create tokenizer
    tok = Tokenizer(Path("model"), normalize=True, mapping=embedding.get_word_mapping())
    model = model.cuda()
    THRESHOLD = 0.5
    # We will take the 
    MAX_NUM_SENTS = max_sents
    scores = []
    for verdict in tqdm(verdicts):
        gp_sents, body_sents, x, mask = load_verdict(verdict, tok) 
        x = x.cuda()
        mask = mask.cuda()

        pred = model(x, mask)
        pred = model.classify(pred)
        pred = pred.cpu().detach().numpy().squeeze(axis=1)
        if equal_length:
            sent_indices = selection(pred, THRESHOLD, len(gp_sents))
        else:
            sent_indices = selection(pred, THRESHOLD, MAX_NUM_SENTS)
        
        selected_sentences = []
        for i, sent in enumerate(body_sents): 
            if i in sent_indices:
                selected_sentences += sent

        labels = []
        for sent in gp_sents:
            labels += sent

        if len(selected_sentences) == 0:
            selected_sentences = ["<unk>"]
        score = evaluate([labels], [selected_sentences])[0]
        scores.append(score)
    
    return scores

def selection(pred: np.array, threshold: float, max_sents: int) -> Set[int]:
    num_possible_selections = np.sum(np.where(pred >= threshold, 1, 0))
    selections = min(num_possible_selections, max_sents)
    if selection == 0:
        raise ValueError
    pred = np.where(pred >= threshold, pred, 0.0)
    indices = set(pred.argsort()[-selections:].tolist())
    return indices

def evaluate_lead(verdicts: List[str], n: Tuple=(3,0)) -> Dict:
    assert (n[0] > 0 or n[1] > 0) and len(n) > 0
    tok = Tokenizer(Path("model"), normalize=True)
    scores = []
    for verdict in tqdm(verdicts):
        verdict = tok.tokenize_verdict_without_id(verdict) 

        gp = [token for sentence in verdict["guiding_principle"] for token in sentence]
        selected_sentences = []
        if n[0] > 0:
            selected_sentences += verdict["facts"][:n[0]]
        if n[1] > 0:
            selected_sentences += verdict["reasoning"][:n[1]]

        selected_sentences = [token for sentence in selected_sentences for token in sentence]

        if len(selected_sentences) == 0:
            selected_sentences = ["<unk>"]        
        score = evaluate([gp], [selected_sentences])[0]
        scores.append(score)
    
    return scores

def evaluate_random(verdicts: List[str], equal_length: bool=True) -> Dict:
    random.seed(2021)
    tok = Tokenizer(Path("model"), normalize=True)
    scores = []
    for verdict in tqdm(verdicts):
        verdict = tok.tokenize_verdict_without_id(verdict) 

        gp = [token for sentence in verdict["guiding_principle"] for token in sentence]
        text = verdict["facts"]+verdict["reasoning"]
        if equal_length:
            selected_sentences = random.choices(text, k=len(verdict["guiding_principle"]))
        else:
            selected_sentences = random.choices(text, k=3)

        selected_sentences = [token for sentence in selected_sentences for token in sentence]

        if len(selected_sentences) == 0:
            selected_sentences = ["<unk>"]        
        score = evaluate([gp], [selected_sentences])[0]
        scores.append(score)
    
    return scores

def evaluate_abs_model(model: nn.Module, embedding: nn.Module, verdicts: List[str], max_sents: int=3, max_toks: int=150) -> List[Dict[str, float]]:
    """ Will evaluate a model on all the verdicts given. Some additional parameters are possible to reduce the length 
        Parameters:
            model -- the given NN used for the predictions
            embedding -- the embeddings used for the token <-> id mapping
            verdicts -- the paths to the verdicts which shall be evaluated
            max_sents -- maximum number of sentences per created summarization
            max_toks -- maximum number of tokens to generate; stop if either max_sents or max_toks is met
    """
    # Create tokenizer
    tok = Tokenizer(Path("model"), normalize=True, mapping=embedding.get_word_mapping())
    model = model.cuda()
    # We will take the 
    MAX_NUM_SENTS = max_sents
    MAX_NUM_TOKENS = max_toks
    MAX_INDEX = tok.get_num_tokens()-1
    scores = []
    for verdict in tqdm(verdicts):
        gp_sents, facts, facts_mask, reason, reason_mask = load_seperated_verdict(verdict, tok) 
        facts = facts.cuda()
        reason = reason.cuda()
        facts_mask = facts_mask.cuda()
        reason_mask = reason_mask.cuda()

        sent_count = 0
        tok_count = 0
        words = [0]
        # We will also store the probabilities, as it is way easier to expand this to beam search that way
        probs = [1.0]
        while True:
            # Init start vector
            prev_tensor = torch.tensor(words, dtype=torch.long).cuda()[None,:]
            
            pred = model(prev_tensor, facts, facts_mask, reason, reason_mask)
            
            # Get max index; if max_index == max possible index -> end of sentence -> increase sent count
            # We also do not want to produce an unknown token (i.e. exclude 0 from the max)
            index = torch.argmax(pred[1:])+1
            prob = pred[index]
            index = index.cpu().item()
            prob = prob.cpu().item()
            if index == MAX_INDEX:
                sent_count += 1
                words.append(MAX_INDEX)
                probs.append(prob)
            else:
                tok_count += 1
                words.append(index)
                probs.append(prob)

            if sent_count >= MAX_NUM_SENTS:
                break
            elif tok_count >= MAX_NUM_TOKENS:
                break
        
        # Convert the words back to text  
        selected_sentences = list(map(lambda token: tok.id2tok[token], filter(lambda token: token not in [0,MAX_INDEX], words)))

        labels = []
        for sent in gp_sents:
            labels += sent

        if len(selected_sentences) == 0:
            selected_sentences = ["<unk>"]
        score = evaluate([labels], [selected_sentences])[0]
        scores.append(score)
    
    return scores

def evaluate_abs_model_beam(model: nn.Module, embedding: nn.Module, verdicts: List[str], max_sents: int=3, max_toks: int=150, beam_size: int=5) -> List[Dict[str, float]]:
    """ Will evaluate a model on all the verdicts given. Some additional parameters are possible to reduce the length 
        Parameters:
            model -- the given NN used for the predictions
            embedding -- the embeddings used for the token <-> id mapping
            verdicts -- the paths to the verdicts which shall be evaluated
            max_sents -- maximum number of sentences per created summarization
            max_toks -- maximum number of tokens to generate; stop if either max_sents or max_toks is met
    """
    # Create tokenizer
    tok = Tokenizer(Path("model"), normalize=True, mapping=embedding.get_word_mapping())
    model = model.cuda()
    # We will take the 
    MAX_NUM_SENTS = max_sents
    MAX_NUM_TOKENS = max_toks
    MAX_INDEX = tok.get_num_tokens()-1
    scores = []
    for verdict in tqdm(verdicts):
        gp_sents, facts, facts_mask, reason, reason_mask = load_seperated_verdict(verdict, tok) 
        facts = facts.cuda()
        reason = reason.cuda()
        facts_mask = facts_mask.cuda()
        reason_mask = reason_mask.cuda()

        # Each tuple in beam corresponds to one search history with (probability, sentence_count, words) as content
        beams = [(1.0, 0, [0])]
        words = []
        while True:
            new_directions = []
            for prob, sent_count, words in beams:
                # Init start vector
                prev_tensor = torch.tensor(words, dtype=torch.long).cuda()[None,:]
                
                pred = model(prev_tensor, facts, facts_mask, reason, reason_mask)
                
                probs, indices = torch.topk(pred, beam_size)
                indices = indices.cpu()
                probs = probs.cpu()
                for p, i in zip(probs, indices):
                    if i == MAX_INDEX:
                        new_directions.append((prob*p, sent_count+1, words+[i]))
                    else:
                        new_directions.append((prob*p, sent_count, words+[i]))
            
            # We now want the top-beam_size summaries as new beams
            new_directions = sorted(new_directions, key=lambda x: x[0], reverse=True)
            beams = new_directions[:beam_size]

            # Check if we have any finished summary, then take this summary:
            for beam in beams:
                if beam[1] >= MAX_NUM_SENTS:
                    words = beam[2]
                    break
                elif len(beam[2]) >= MAX_NUM_TOKENS:
                    words = beam[2]
                    break

            if len(words) > 0:
                break
        
        assert len(words) > 0
        
        # Convert the words back to text  
        selected_sentences = list(map(lambda token: tok.id2tok[token], filter(lambda token: token not in [0,MAX_INDEX], words)))

        labels = []
        for sent in gp_sents:
            labels += sent

        if len(selected_sentences) == 0:
            selected_sentences = ["<unk>"]
        score = evaluate([labels], [selected_sentences])[0]
        scores.append(score)
    
    return scores

def load_verdict(path: Path, tok: Tokenizer):
    verdict = tok.tokenize_verdict_without_id(path)
    x_1 = list(map(lambda sentence: list(map(lambda token: tok.tok2id[token], sentence)), verdict["facts"]))
    x_2 = list(map(lambda sentence: list(map(lambda token: tok.tok2id[token], sentence)), verdict["reasoning"]))
    
    x = []
    for ind in x_1:
        x.append(torch.LongTensor(ind))
    for ind in x_2:
        x.append(torch.LongTensor(ind))

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    mask = (x!=0)
    return verdict["guiding_principle"], verdict["facts"] + verdict["reasoning"], x, mask

def load_seperated_verdict(path: Path, tok: Tokenizer):
    verdict = tok.tokenize_verdict_without_id(path)
    x_1 = list(map(lambda sentence: list(map(lambda token: tok.tok2id[token], sentence)), verdict["facts"]))
    x_2 = list(map(lambda sentence: list(map(lambda token: tok.tok2id[token], sentence)), verdict["reasoning"]))
    
    f = []
    for ind in x_1:
        f.append(torch.LongTensor(ind))
    while len(f) < 1:
        f.append(torch.LongTensor([0]))
    r = []
    for ind in x_2:
        r.append(torch.LongTensor(ind))
    while len(r) < 1:
        r.append(torch.LongTensor([0]))

    f = torch.nn.utils.rnn.pad_sequence(f, batch_first=True)
    r = torch.nn.utils.rnn.pad_sequence(r, batch_first=True)
    f_mask = (f!=0)
    r_mask = (r!=0)
    return verdict["guiding_principle"], f, f_mask, r, r_mask

def _mult_lr_factor_(it: int) -> float:
    return 0.98
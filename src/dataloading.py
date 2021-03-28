# Contains all the dataloading tasks for the training procedures
# First loading & processing of verdicts will be replaced by a version in a more bare-metal language,
# as those tasks can be easily parallelized.
# Preprocessing tasks will be loaded from preprocessing.py, as we are here concerned about data plumbing!
# Functionality:
#   - load json verdict, extract relevant text portions, tokenize & preprocess text
#      * space-based tokenization
#      * byte-pair tokenization
#   - fetch relevant norm texts for a verdict, tokenize & preprocess text
#   - dataloader that can be used in the model training
import os
import io
from collections import Counter
import pickle
from enum import Enum
from pathlib import Path
from itertools import chain
from typing import List, Tuple, Set, Callable, Dict
import sqlite3

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from rouge import Rouge # https://github.com/pltrdy/rouge

from .preprocessing import load_verdict, TokenizationType
# We use import as to be able to switch out the Tokenizer later on
from .preprocessing import Tokenizer as Tokenizer
from .coverage import find_optimal_coverage, find_greedy_coverage

DATA_PATH = Path("data")/"dataset"
MODEL_PATH = Path("model")
NORM_DB_PATH = Path("data")/"databases"/"norms.db"

# Maximum length of a sentence
CUTOFF = 100

class LossType(Enum):
    BCE = 0
    HAMM_HINGE = 1
    HAMM_LOGI = 2
    SUBSET_HINGE = 3
    SUBSET_LOGI = 4
    ABS = 5

class ExtractiveDataset(Dataset):
    """ Dataset used for the extractive summarization training """

    def __init__(self, verdict_paths: List[Path], tokenizer: Tokenizer, loss_type: LossType, transform: Callable[[List[int]],List[int]]=None, database: str="extractive.db"):
        self.verdicts = verdict_paths
        self.tokenizer = tokenizer
        self.db_path = Path("data")/"databases"/database

        # Each loss needs different type of tensor as target...
        if loss_type == LossType.BCE:
            self.value, self.default, self.dtype = 1.0, 0.0, torch.float32
        else:
            self.value, self.default, self.dtype = 1.0, 0.0, torch.float32
        # It is possible to use the transform function to cap the number of indices per sentence etc.
        self.transform = transform

    def __getitem__(self, index: int):
        verdict_path = self.verdicts[index]
        verdict = self.tokenizer.tokenize_verdict(verdict_path)
        fact_ind, reas_ind = get_ext_target_indices(verdict_path, self.db_path, self.tokenizer)
        
        # Define collate function that can deal with variable list lengths
        x = []
        for ind in chain(verdict["facts"], verdict["reasoning"]):
            x.append(self.__lam__(ind))
        
        assert len(x) > 0, verdict_path
        assert len(fact_ind) > 0 or len(reas_ind) > 0, verdict_path

        # We have to create our targets for the extractive summarization over the facts and reasoning, i.e. we have to combine both their targets
        # into one long tensor
        reasoning_offset = len(verdict["facts"])
        y = torch.tensor([self.default]*(len(x)), dtype=self.dtype)
        for i in fact_ind:
            y[i] = self.value
        for i in reas_ind:
            y[i+reasoning_offset] = self.value

        return x, y

    def __len__(self):
        return len(self.verdicts)

    def __lam__(self, ind: List[int]) -> torch.LongTensor:
        # Define this lambda function as we otherwise cannot use multiple workers for dataloading
        if self.transform is not None:
            return torch.LongTensor(self.transform(ind))
        else:
            return torch.LongTensor(ind)

class AbstractiveDataset(Dataset):
    """ Dataset used for the abstractive summarization training """

    def __init__(self, verdict_paths: List[Path], tokenizer: Tokenizer, transform: Callable[[List[int]],List[int]]=None, load_norms: bool=False):
        self.verdicts = verdict_paths
        self.tokenizer = tokenizer
        self.load_norms = load_norms
        # We need to know when to stop generating -> this is always dependend on the tokenizer, i.e. the lowest number not taken is our ending token
        self.ENDING_TOKEN = self.tokenizer.get_num_tokens()-1

        # It is possible to use the transform function to cap the number of indices per sentence etc.
        self.transform = transform

    def __getitem__(self, index: int):
        verdict_path = self.verdicts[index]
        verdict = self.tokenizer.tokenize_verdict(verdict_path)
        
        # Define collate function that can deal with variable list lengths
        # Only very few sentences are above 40 tokens, we want to cut them here to decrease the variable memory needed in the GPU
        f = []
        for ind in verdict["facts"]:
            f.append(self.__lam__(ind[:40]))
        if len(f) == 0:
            f.append(self.__lam__([0]))

        r = []
        for ind in verdict["reasoning"]:
            r.append(self.__lam__(ind[:40]))
        if len(r) == 0:
            r.append(self.__lam__([0]))
        
        y = []
        for ind in verdict["guiding_principle"]:
            # Startoff sentence with <unk> (we need to create zero vector any way for model as giving the prev encoder no vector would increase the complexity more)
            # Append ending token (token with highest id) to stop generating
            y.append(torch.LongTensor([0]+ind[:50]+[self.ENDING_TOKEN]))

        # We will also reduce the number of sentence verdict to the average + 5
        return f[:40], r[:80], y

    def __len__(self):
        return len(self.verdicts)

    def __lam__(self, ind: List[int]) -> torch.LongTensor:
        # Define this lambda function as we otherwise cannot use multiple workers for dataloading
        if self.transform is not None:
            return torch.LongTensor(self.transform(ind))
        else:
            return torch.LongTensor(ind)

def collate(batch: List[Tuple[List[torch.Tensor], torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """ Will transform the given sentence list (each entry in the first list represents a document; this document contains various number of sentences), to 
        one sentence tensor with all sentences being padded to the same length.
        Returns:
            List[Tuple[
                sentence index tensor = padded tensor version of the initial sentence list,
                y = original target for each sentence,
                lenghts = original lenghts of all sentences before padding,
                mask for the padded version of the sentence indices
            ]]
    """
    # We will introduce this collate function, if we want to deal with batch samples of varying sizes
    # Taken and adapted from @pinocchio https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/13
    
    targets = map(lambda doc: doc[1], batch)

    # We can have multiple documents per batch, each having multiple sentences
    # ATTENTION: As we currently do not need the lengths, their calculation is excluded
    # lengths = map(lambda doc: torch.tensor([sent.shape[0] for sent in doc[0]]), batch)

    batch = map(lambda doc: torch.nn.utils.rnn.pad_sequence(doc[0], batch_first=True), batch)
    
    res = []
    for x, t in zip(batch, targets):
        mask = (x!=0)
        res.append((x,t,mask))
    return res

def collate_abs(batch: List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """ Will transform the given sentence list (each entry in the first list represents a document; sentences from: facts, reasoning, guiding_principle, norms), each to 
        one sentence tensor with all sentences being padded to the same length.
        Returns:
            List[Tuple[
                targets: indices for target words, each row is one sentence,
                length of target sentence,
                facts: indices of facts,
                facts_mask: mask introduced by padding,
                reasoning: indices of reasoning,
                reasoning_mask: mask introduced by padding,
                norms: indices for norm
            ]]
    """
    res = []
    for doc in batch:
        t = torch.nn.utils.rnn.pad_sequence(doc[2], batch_first=True)
        l = torch.tensor([sent.shape[0] for sent in doc[2]])

        f = torch.nn.utils.rnn.pad_sequence(doc[0], batch_first=True)
        f_m = (f!=0)
        r = torch.nn.utils.rnn.pad_sequence(doc[1], batch_first=True)
        r_m = (r!=0)
        res.append([t, l, f, f_m, r, r_m])
    
    return res

def collate_abs_long(batch: List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """ Will transform the given sentence list (each entry in the first list represents a document; sentences from: facts, reasoning, guiding_principle, norms), each to 
        one sentence tensor with all sentences being padded to the same length.
        In this collate function each sentence will be appended to the last, with one one end token in between. I.e. this allows us to train the sentence creation continously.
        Returns:
            List[Tuple[
                targets: indices for target words, each row is one sentence,
                length of target sentence,
                facts: indices of facts,
                facts_mask: mask introduced by padding,
                reasoning: indices of reasoning,
                reasoning_mask: mask introduced by padding,
                norms: indices for norm
            ]]
    """
    res = []
    for doc in batch:
        t = torch.hstack(doc[2]).unsqueeze(0)
        l = torch.tensor([sum(sent.shape[0] for sent in doc[2])])

        f = torch.nn.utils.rnn.pad_sequence(doc[0], batch_first=True)
        f_m = (f!=0)
        r = torch.nn.utils.rnn.pad_sequence(doc[1], batch_first=True)
        r_m = (r!=0)
        res.append([t, l, f, f_m, r, r_m])
    
    return res

# Fix train, val, test split
def fix_data_split(percentage: List[float]=[0.8,0.1,0.1]):
    """ Creates the datasplit via assigning files to the train, val or test set 
        Params:
            - percentage -- List containing the percentages for [train, validation, test] datasets
                            Must add up to one and have a length of 3!
    """
    assert sum(percentage) == 1.0
    assert len(percentage) == 3

    # MAX_SIZE used to filter really long files
    MAX_SIZE = 200
    
    files = [DATA_PATH/file for file in os.listdir(DATA_PATH) if os.path.getsize(DATA_PATH/file)/1024 < MAX_SIZE]
    train_files, temp_files = train_test_split(files, test_size=percentage[1]+percentage[2], shuffle=True, random_state=42)
    test_portion = percentage[2]/(percentage[1] + percentage[2])
    val_files, test_files = train_test_split(temp_files, test_size=test_portion, shuffle=True, random_state=43)
    print("Train:", len(train_files), "; Validation:", len(val_files), "; Test:", len(test_files))
    with io.open(MODEL_PATH/"train_files.pkl", "wb") as f:
        pickle.dump(train_files, f)
    with io.open(MODEL_PATH/"val_files.pkl", "wb") as f:
        pickle.dump(val_files, f)
    with io.open(MODEL_PATH/"test_files.pkl", "wb") as f:
        pickle.dump(test_files, f)

def get_greedy_files():
    """ Will create new train_files and val_files; only those with a valid label in extractive_greedy.db """
    tok =  Tokenizer(Path("model"), normalize=True)
    with io.open(MODEL_PATH/"train_files.pkl", "rb") as f:
        train_files =  pickle.load(f)
    dataset = ExtractiveDataset(train_files, tok, loss_type=LossType.BCE, database="extractive_greedy.db")
    res_train_files = []
    for i, file in tqdm(enumerate(train_files)):
        try:
            _ = dataset[i]
            res_train_files.append(file)
        except:
            pass

    print("New train:", len(res_train_files), "; Old train:", len(train_files))
    final_dataset = ExtractiveDataset(res_train_files, tok, loss_type=LossType.BCE, database="extractive_greedy.db")
    for _ in tqdm(final_dataset):
        pass

    with io.open(MODEL_PATH/"val_files.pkl", "rb") as f:
        val_files = pickle.load(f)
    dataset = ExtractiveDataset(val_files, tok, loss_type=LossType.BCE, database="extractive_greedy.db")
    res_val_files = []
    for i, file in tqdm(enumerate(val_files)):
        try:
            _ = dataset[i]
            res_val_files.append(file)
        except:
            pass

    print("New train:", len(res_val_files), "; Old train:", len(val_files))
    final_dataset = ExtractiveDataset(res_val_files, tok, loss_type=LossType.BCE, database="extractive_greedy.db")
    for _ in tqdm(final_dataset):
        pass

    with io.open(MODEL_PATH/"train_greedy_files.pkl", "wb") as f:
        pickle.dump(res_train_files, f)

    with io.open(MODEL_PATH/"val_greedy_files.pkl", "wb") as f:
        pickle.dump(res_val_files, f)

def get_train_files() -> List[str]:
    """ Returns all the files previously selected to be used for training. """
    with io.open(MODEL_PATH/"train_files.pkl", "rb") as f:
        return pickle.load(f)

def get_greedy_train_files() -> List[str]:
    """ Returns all the files previously selected to be used for greedy training (via get_greedy_files, i.e. subset of the total train_files) """
    with io.open(MODEL_PATH/"train_greedy_files.pkl", "rb") as f:
        return pickle.load(f)

def get_val_files() -> List[str]:
    """ Returns all the files previously selected to be used for validation. """
    with io.open(MODEL_PATH/"val_files.pkl", "rb") as f:
        return pickle.load(f)

def get_greedy_val_files() -> List[str]:
    """ Returns all the files previously selected to be used for greedy validation (via get_greedy_files, i.e. subset of the total val_files) """
    with io.open(MODEL_PATH/"val_greedy_files.pkl", "rb") as f:
        return pickle.load(f)

def get_test_files() -> List[str]:
    """ Returns all the files previously selected to be used for testing. """
    with io.open(MODEL_PATH/"test_files.pkl", "rb") as f:
        return pickle.load(f)

def get_ext_target_indices(verdict: Path, db_path: Path, tok: Tokenizer, create_missing_db: bool=False) -> Tuple[List[int]]:
    """ Returns the indices for sentences in reasoning and facts, which have the highest overlap with a guiding principle sentence.
        -> We do not want to dynamically compute this, but only once and write it to a database with:
            1. verdict name (only the file name)
            2. matched guiding principle (int from 0 to num guiding principles - 1) in the verdict
            3. section (0 = facts, 1 = reasoning)
            4. index (inside of the section)
        Returns:
            Tuple[List[index]] -- for all guiding principles in verdict one sentence; for each segement one list: Tuple.0 == facts, Tuple.1 == reasoning
    """
    file_name = verdict.name
    # We will use sqlite, as it is pretty easy to integrate + port to other machines
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tok_type = 1 if tok.get_type() == TokenizationType.SPACE else 2
    
    cursor.execute("select section, ind from labels where name=:file and tokenizer=:tok ;", {"file": file_name, "tok": tok_type})
    res = cursor.fetchall()
    facts = []
    reasoning = []
    for row in res:
        if row[0] == 0:
            facts.append(row[1])
        else:
            reasoning.append(row[1])

    conn.close()
    return (facts, reasoning)

def create_ext_target_db(db_path: Path, tok: Tokenizer, data_folder: Path=DATA_PATH):
    """ Creates the database used for querying the gold labels for the extractive summarization task. Based on a one-to-one mapping for the guiding principles sentences to verdict sentence. """
    print("Creating gold labels for extractive summarization:")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("create table labels (name text not null, tokenizer integer not null, gid integer not null, section integer not null, ind integer not null, primary key (name,tokenizer,gid));")
        conn.commit()
    except sqlite3.OperationalError:
        print("Resuming from previously build table")
    # We will use a higher rouge score for getting the gold results; if there is no result, we have to come back to the lower rouge scores
    rouge = Rouge(metrics=["rouge-1", "rouge-2"])

    for file in tqdm(os.listdir(data_folder), desc="Generating gold labels"):
        verdict = tok.tokenize_verdict_without_id(data_folder/file)
        tok_type = 1 if tok.get_type() == TokenizationType.SPACE else 2

        # Only execute based on num of gps already in db

        # It is necessary to rejoin the tokens into one string
        for segment in verdict:
            verdict[segment] = list(map(lambda sentence: " ".join(sentence), verdict[segment]))

        indices = ([], [])
        # Calculate indices for each guiding principle
        for num_gp, gp in enumerate(verdict["guiding_principle"]):
            #cursor.execute("select * from labels ")
            max_rouge = {
                "rouge-1": {
                    "segment": -1,
                    "index": -1,
                    "score": 0.0
                },
                "rouge-2": {
                    "segment": -1,
                    "index": -1,
                    "score": 0.0
                }
            }
            # Run over all sentences in facts
            for i, s in enumerate(verdict["facts"]):
                score = rouge.get_scores(gp, s)
                if score[0]["rouge-2"]["f"] > max_rouge["rouge-2"]["score"]:
                    max_rouge["rouge-2"]["segment"] = 0
                    max_rouge["rouge-2"]["index"] = i
                    max_rouge["rouge-2"]["score"] = score[0]["rouge-2"]["f"]
                if score[0]["rouge-1"]["f"] > max_rouge["rouge-1"]["score"]:
                    max_rouge["rouge-1"]["segment"] = 0
                    max_rouge["rouge-1"]["index"] = i
                    max_rouge["rouge-1"]["score"] = score[0]["rouge-2"]["f"]
            # Run over all sentences in reasoning
            for i, s in enumerate(verdict["reasoning"]):
                score = rouge.get_scores(gp, s)
                if score[0]["rouge-2"]["f"] > max_rouge["rouge-2"]["score"]:
                    max_rouge["rouge-2"]["segment"] = 1
                    max_rouge["rouge-2"]["index"] = i
                    max_rouge["rouge-2"]["score"] = score[0]["rouge-2"]["f"]
                if score[0]["rouge-1"]["f"] > max_rouge["rouge-1"]["score"]:
                    max_rouge["rouge-1"]["segment"] = 1
                    max_rouge["rouge-1"]["index"] = i
                    max_rouge["rouge-1"]["score"] = score[0]["rouge-2"]["f"]
            
            # If we have a maximum rouge-2 score, we will take that indice; otherwise take the rouge-1 sentence
            if max_rouge["rouge-2"]["segment"] != -1:
                indices[max_rouge["rouge-2"]["segment"]].append((num_gp, max_rouge["rouge-2"]["index"]))
            elif max_rouge["rouge-1"]["segment"] != -1:
                indices[max_rouge["rouge-1"]["segment"]].append((num_gp, max_rouge["rouge-1"]["index"]))
            else:
                with io.open("logging/missing_ext_goldlabel.txt", "a+") as f:
                    f.write(file + " : " + str(num_gp) + "\n")

        assert all(i > -1 for _, i in indices[0]) and all(i > -1 for _, i in indices[1])

        for gid, i in indices[0]:
            cursor.execute("insert into labels values (?, ?, ?, ?, ?);", (file, tok_type, gid, 0, i))
        for gid, i in indices[1]:
            cursor.execute("insert into labels values (?, ?, ?, ?, ?);", (file, tok_type, gid, 1, i))
        conn.commit()
    
    conn.close()

def create_ext_greedy_db(db_path: Path, tok: Tokenizer, data_folder: Path=DATA_PATH):
    """ Creates the database used for querying the gold labels for the extractive summarization task. Greedily takes sentences until the we cannot improve further.
        Acutally we try to find the exact set of sentences, which in the bigram setting boils down to the NP-complete problem of finding a minimal spanning set in a hypergraph.
        But in our case we should be fine, as we only need to look at sentences which have a total overlap of 2-grams.
    """
    print("Creating gold labels for extractive summarization:")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("create table labels (name text not null, tokenizer integer not null, section integer not null, ind integer not null, primary key (name,tokenizer,section,ind));")
        conn.commit()
    except sqlite3.OperationalError:
        print("Resuming from previously build table")
    
    counts = []
    for file in tqdm(os.listdir(data_folder), desc="Generating gold labels"):
        verdict = tok.tokenize_verdict(data_folder/file)
        tok_type = 1 if tok.get_type() == TokenizationType.SPACE else 2

        # We cannot find the optimal coverage as this is an NP-Hard problem and even the greedy selection might be unfeasible
        #indices = find_optimal_coverage(verdict["guiding_principle"], verdict["facts"], verdict["reasoning"])
        indices = find_greedy_coverage(verdict["guiding_principle"], verdict["facts"], verdict["reasoning"])

        if len(indices[0]) == 0 and len(indices[1]) == 0:        
            with io.open("logging/missing_ext_greedy_goldlabel.txt", "a+") as f:
                f.write(file + "\n")

        assert all(i > -1 for i in indices[0]) and all(i > -1 for i in indices[1])

        for i in indices[0]:
            cursor.execute("insert into labels values (?, ?, ?, ?);", (file, tok_type, 0, i))
        for i in indices[1]:
            cursor.execute("insert into labels values (?, ?, ?, ?);", (file, tok_type,  1, i))
        conn.commit()
    
    conn.close()
    with open(Path("src")/"analysis"/"rouge_counts.pkl", "wb") as f:
        pickle.dump(counts, f)

def get_norm_sentences(norm: str, paragraph: str, db_path: Path=NORM_DB_PATH) -> List[str]:
    """ Query the norms.db to get the paragraph and sentence of a referenced norm 
        ATTENTION: It is still necessary to use the Tokenizer on the resulting sentences
        Returns:
            - List[str] = List of sentences making up the norm and paragraph
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("select sentence, content from norms where name=:n and paragraph=:p", {"n": norm, "p": paragraph})
    sentences = list(map(lambda t: t[1], cursor.fetchall()))
    return sentences

def get_norms(db_path: Path=NORM_DB_PATH) -> Set[str]:
    """ Creates a set containing all norms from the database. 
        This can be used to faster check, whether a norm is contained in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("select distinct name from norms") 
    norms = set(map(lambda t: t[0], cursor.fetchall()))
    return norms

def transform_cutoff(sentence: List[int]) -> List[int]:
    """ Will restrict the length of a sentence by CUTOFF """
    return sentence[:CUTOFF]

if __name__ == "__main__":
    tok = Tokenizer(MODEL_PATH)
    #tok.create_token_id_mapping()
    #fix_data_split()
    #create_ext_target_db(Path("data")/"databases"/"extractive.db", tok)
    create_ext_greedy_db(Path("data")/"databases"/"extractive_greedy.db", tok)
    
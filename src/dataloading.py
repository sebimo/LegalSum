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
from pathlib import Path
from itertools import chain
from typing import List, Tuple, Set, Callable
import sqlite3

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from rouge import Rouge # https://github.com/pltrdy/rouge

from .preprocessing import load_verdict, TokenizationType
# We use import as to be able to switch out the Tokenizer later on
from .preprocessing import Tokenizer as Tokenizer

DATA_PATH = Path("data")/"dataset"
MODEL_PATH = Path("model")
NORM_DB_PATH = Path("data")/"databases"/"norms.db"

class ExtractiveDataset(Dataset):
    """ Dataset used for the extractive summarization training """

    def __init__(self, verdict_paths: List[Path], tokenizer: Tokenizer, transform: Callable[[List[int]],List[int]]=lambda x: x):
        self.verdicts = verdict_paths
        self.tokenizer = tokenizer
        self.db_path = Path("data")/"databases"/"extractive.db"
        # It is possible to use the transform function to cap the number of indices per sentence etc.
        self.transform = transform

    def __getitem__(self, index: int):
        verdict_path = self.verdicts[index]
        print(verdict_path)
        verdict = self.tokenizer.tokenize_verdict(verdict_path)
        fact_ind, reas_ind = get_ext_target_indices(verdict_path, self.db_path, self.tokenizer)
        
        # Define collate function that can deal with variable list lengths
        x = list(map(lambda ind: torch.LongTensor(self.transform(ind)), chain(verdict["facts"], verdict["reasoning"])))

        # We have to create our targets for the extractive summarization over the facts and reasoning, i.e. we have to combine both their targets
        # into one long tensor
        reasoning_offset = len(verdict["facts"])
        y = torch.Tensor([0.0]*(len(x)))
        for i in fact_ind:
            y[i] = 1.0
        for i in reas_ind:
            y[i+reasoning_offset] = 1.0

        return x, y

    def __len__(self):
        return len(self.verdicts)

def collate(batch: List[Tuple[List[torch.Tensor], torch.Tensor]], device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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
    
    targets = map(lambda doc: doc[1].to(device), batch)

    # We can have multiple documents per batch, each having multiple sentences
    lengths = map(lambda doc: torch.tensor([sent.shape[0] for sent in doc[0]]).to(device), batch)

    batch = map(lambda doc: torch.nn.utils.rnn.pad_sequence(doc[0], batch_first=True).to(device), batch)
    
    res = []
    for x, t, l in zip(batch, targets, lengths):
        mask = (x!=0).to(device)
        res.append((x,t,l,mask))
    return res


# TODO fix train, val, test split
def fix_data_split(percentage: List[float]=[0.8,0.1,0.1]):
    """ Creates the datasplit via assigning files to the train, val or test set 
        Params:
            - percentage -- List containing the percentages for [train, validation, test] datasets
                            Must add up to one and have a length of 3!
    """
    assert sum(percentage) == 1.0
    assert len(percentage) == 3
    
    files = os.listdir(DATA_PATH)
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

def get_train_files():
    """ Returns all the files previously selected to be used for training. """
    with io.open(MODEL_PATH/"train_files.pkl", "rb") as f:
        return pickle.load(f)

def get_val_files():
    """ Returns all the files previously selected to be used for validation. """
    with io.open(MODEL_PATH/"val_files.pkl", "rb") as f:
        return pickle.load(f)

def get_test_files():
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
    # Query db for info
    try:
        cursor.execute("select section, ind from labels where name=:file and tokenizer=:tok ;", {"file": file_name, "tok": tok_type})
        res = cursor.fetchall()
    except sqlite3.OperationalError:
        # If this does not work, the table needs to be created + populated
        if create_missing_db:
            create_ext_target_db(db_path, tok)
    
    # Re-run the query
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
    """ Creates the database used for querying the gold labels for the extractive summarization task """
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

if __name__ == "__main__":
    tok = Tokenizer(MODEL_PATH)
    #tok.create_token_id_mapping()
    #fix_data_split()
    create_ext_target_db(Path("data")/"databases"/"extractive.db", tok)
    
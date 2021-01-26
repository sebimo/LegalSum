# Contains all the dataloading tasks for the training procedures
# First loading & processing of verdicts will be replaced by a version in a more bare-metal language,
# as those tasks can be easily parallelized.
# Preprocessing tasks will be loaded from preprocessing.py, as we are here concerned about data plumbing!
# Functionality:
#   - load json verdict, extract relevant text portions, tokenize & preprocess text
#      * space-based tokenization
#      * byte-pair tokenization
#  (- fetch relevant norm texts for a verdict, tokenize & preprocess text)
#   - dataloader that can be used in the model training
import os
import io
from collections import Counter
import pickle
from pathlib import Path
from itertools import chain
from typing import List, Tuple
import sqlite3

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from preprocessing import load_verdict
# We use import as to be able to switch out the Tokenizer later on
from preprocessing import Tokenizer as Tokenizer

DATA_PATH = Path("data")/"dataset"
MODEL_PATH = Path("model")

class ExtractiveDataset(Dataset):
    """ Dataset used for the extractive summarization training """

    def __init__(self, verdict_paths: List[Path], tokenizer: Tokenizer):
        self.verdicts = verdict_paths
        self.tokenizer = tokenizer
        self.db_path = Path("..")/"data"/"databases"/"extractive.db"

    def __get_item__(self, index: int):
        verdict_path = self.verdicts[index]
        verdict = self.tokenizer.tokenize_verdict(verdict_path)
        fact_ind, reas_ind = get_ext_target_indices(verdict_path, self.db_path, self.tokenizer)
        # TODO define collate function that can deal with variable list lengths
        x = list(map(lambda ind: torch.LongTensor(ind), chain(verdict["facts"], verdict["reasoning"])))
        reasoning_offset = len(verdict["facts"])
        y = [0]*(len(x))
        for i in fact_ind:
            y[i] = 1
        for i in reas_ind:
            y[i+reasoning_offset] = 1
        return x, y

    def __len__(self):
        return len(self.verdicts)

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

def get_ext_target_indices(verdict: Path, db_path: Path, tok: Tokenizer) -> Tuple[List[int]]:
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
    # Query db for info
    try:
        cursor.execute("select section, index from labels where name=:file ;", {"file": file_name})
        res = cursor.fetchall()
    except sqlite3.OperationalError:
        # If this does not work, the table needs to be created + populated
        create_ext_target_db(db_path, tok)
    
    # Re-run the query
    cursor.execute("select section, index from labels where name=:file ;", {"file": file_name})
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

def create_ext_target_db(db_path: Path, tok: Tokenizer):
    """ Creates the database used for querying the gold labels for the extractive summarization task """
    # TODO get library for rouge scores!
    print("Creating gold labels for extractive summarization:")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("create table labels (name text, gid int, section int, index int)")
    conn.commit()
    for file in tqdm(os.listdir(DATA_PATH)):
        verdict = tok.tokenize_verdict(DATA_PATH/file)
        # TODO Calculate indices
        indices = ([], [])
        for gid, i in enumerate(indices[0]):
            cursor.execute("insert into labels values (?, ?, ?, ?)", (file, gid, 0, i))
        offset = len(indices[0])
        for gid, i in enumerate(indices[1]):
            cursor.execute("insert into labels values (?, ?, ?, ?)", (file, gid+offset, 1, i))
        conn.commit()
    
    conn.close()

if __name__ == "__main__":
    tok = Tokenizer(MODEL_PATH)
    #tok.create_token_id_mapping()
    #fix_data_split()
    get_ext_target_indices(Path("src")/"testing"/"test_data"/"short.json", Path("data")/"database"/"test.db")
    
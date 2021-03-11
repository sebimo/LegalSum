# Help scripts for the Rouge.ipynb as some functions would simply take to long to execute in the notebook
# Those will be executed here and their result written to disk

import sys
sys.path.append("src")
from pathlib import Path
import pickle
import os, io
from collections import defaultdict
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import seaborn as sns

from preprocessing import Tokenizer
from evaluation import evaluate

DATA_PATH = Path("data")/"dataset"
tok = Tokenizer(Path("model"))

def verdict_score(file):
        verdict = tok.tokenize_verdict_without_id(DATA_PATH/file)
        # Used for the final score calculation
        gp_ref = [j for i in verdict["guiding_principle"] for j in i]
        sentences = []
        # We will simplify the strategy here a bit and only look for 2-gram overlaps and ignore any gps with no 2-gram overlap
        for gp in verdict["guiding_principle"]:
            best_score = 0.0
            sentence = None
            for sent in verdict["facts"]:
                score = evaluate([gp], [sent])
                if best_score < score[0]["rouge-2"]["f"]:
                    sentence = sent
                    best_score = score[0]["rouge-2"]["f"]
                    
            for sent in verdict["reasoning"]:
                score = evaluate([gp], [sent])
                if best_score < score[0]["rouge-2"]["f"]:
                    sentence = sent
                    best_score = score[0]["rouge-2"]["f"]
                    
            if sentence is not None:
                sentences.append(sentence)
        
        ref_sentences = [j for i in sentences for j in i]
        if len(ref_sentences) == 0:
            ref_sentences = ["<unk>"]
        score = evaluate([gp_ref], [ref_sentences])
        return score[0]

def scores_same():
    """ Will calculate all the possible scores achievable, if we pick the same number of sentences as found in the guiding principle """
    with Pool(16) as p:
        verdict_score
        scores = p.map(verdict_score, os.listdir(DATA_PATH)[50000:])

    with open(Path("src")/"analysis"/"misc_data"/"same_scores2.pkl", "wb") as f:
        pickle.dump(scores, f)

def scores_greedy():
    """ Will calculate all the possible scores achievable, if we greedily pick sentences """
    scores = []
    lengths = []
    for file in tqdm(os.listdir(DATA_PATH)):
        verdict = tok.tokenize_verdict_without_id(DATA_PATH/file)
        # Used for the final score calculation
        gp_ref = [j for i in verdict["guiding_principle"] for j in i]
        sent_tokens = []
        poss_sentences = verdict["facts"] + verdict["reasoning"]
        best_score = 0.0
        for sent in poss_sentences:
            test = sent_tokens + sent
            score = evaluate([gp_ref], [test])
            if score[0]["rouge-2"]["f"] > best_score:
                best_score = score[0]["rouge-2"]["f"]
                sent_tokens = test
        
        if len(sent_tokens) == 0:
            sent_tokens = ["<unk>"]
        score = evaluate([gp_ref], [sent_tokens])
        lengths.append((len(gp_ref), len(sent_tokens)))
        scores.append(score[0])

    with open(Path("src")/"analysis"/"misc_data"/"greedy_scores_lengths.pkl", "wb") as f:
        pickle.dump((scores, lengths), f)

if __name__ == "__main__":
    scores_same()

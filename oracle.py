from tqdm import tqdm
from pathlib import Path
import os

from src.dataloading import get_ext_target_indices, get_test_files
from src.preprocessing import Tokenizer
from src.evaluation import evaluate

tok = Tokenizer(Path("model"))
DATA_PATH = Path("data/dataset")
DB_PATH = Path("data/databases/extractive.db")

scores = []
for file in tqdm(get_test_files()):
    verdict = tok.tokenize_verdict_without_id(file)
    f_inds, r_inds = get_ext_target_indices(file, DB_PATH, tok)

    sentences = []
    for ind in f_inds:
        sentences += verdict["facts"][ind]
    for ind in r_inds:
        sentences += verdict["reasoning"][ind]
    
    gp = []
    for sent in verdict["guiding_principle"]:
        gp += sent
    
    try:
        score = evaluate([gp], [sentences])[0]
        scores.append((score["rouge-1"]["f"], score["rouge-2"]["f"], score["rouge-l"]["f"]))
    except RecursionError:
        continue

r1 = sum(map(lambda x: x[0], scores))/len(scores)
r2 = sum(map(lambda x: x[1], scores))/len(scores)
rl = sum(map(lambda x: x[2], scores))/len(scores)
print("One-to-One R1:", r1, "; R2:", r2, "; RL:", rl)

DB_PATH = Path("data/databases/extractive_greedy.db")
scores = []
for file in tqdm(get_test_files()):
    verdict = tok.tokenize_verdict_without_id(file)
    f_inds, r_inds = get_ext_target_indices(file, DB_PATH, tok)

    sentences = []
    for ind in f_inds:
        sentences += verdict["facts"][ind]
    for ind in r_inds:
        sentences += verdict["reasoning"][ind]
    
    gp = []
    for sent in verdict["guiding_principle"]:
        gp += sent

    if len(sentences) == 0:
        sentences = ["<unk>"]

    try:
        score = evaluate([gp], [sentences])[0]
        scores.append((score["rouge-1"]["f"], score["rouge-2"]["f"], score["rouge-l"]["f"]))
    except RecursionError:
        continue

r1 = sum(map(lambda x: x[0], scores))/len(scores)
r2 = sum(map(lambda x: x[1], scores))/len(scores)
rl = sum(map(lambda x: x[2], scores))/len(scores)
print("Greedy R1:", r1, "; R2:", r2, "; RL:", rl)
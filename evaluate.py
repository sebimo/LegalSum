# Script used to evaluate the different models on the validation and test set
import os
import io
import json
from typing import List, Dict
from collections import defaultdict
from pathlib import Path

from src.dataloading import get_val_files, get_test_files
from src.training import evaluate_ext_model
from src.model import parse_run_parameters, reload_model

def average(scores: List[Dict]) -> Dict:
    num = len(scores)
    acc_scores = defaultdict(lambda: defaultdict(float))
    for score in scores:
        for rouge in score:
            for measure in score[rouge]:
                acc_scores[rouge][measure] += score[rouge][measure]
    
    d = dict()
    for rouge in acc_scores:
        for measure in acc_scores[rouge]:
            if rouge not in d:
                d[rouge] = {}
            d[rouge][measure] = acc_scores[rouge][measure]/num
    
    return d


if __name__ == "__main__":
    try:
        with io.open("logging/ext_perf.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except: 
        data = []
    for folder in os.listdir(Path("logging/runs/extractive"))[len(data):]:
        print(folder)
        params = parse_run_parameters(folder)
        model, embedding = reload_model(params)
        scores = evaluate_ext_model(model, embedding, get_val_files(), equal_length=True)
        avg_scores = average(scores)
        print(avg_scores)
        data.append({folder: avg_scores})
        with io.open("logging/ext_perf.json", "w+", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=False, indent=4, ensure_ascii=False)

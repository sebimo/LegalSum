# Script used to evaluate the different models on the validation and test set
import os
import io
import json
from typing import List, Dict
from collections import defaultdict
from pathlib import Path

from src.dataloading import get_val_files, get_test_files
from src.training import evaluate_ext_model, evaluate_lead, evaluate_random, evaluate_abs_model
from src.model import parse_run_parameters, reload_model
from src.abs_model import parse_abs_run_parameters, reload_abs_model

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

def evaluate_leads(verdicts: List[str]):
    # (f,r): f sentences from the beginning of facts and r sentences from the beginning of reasoning 
    acc_score = []
    for l in [(3,0), (0,3), (3,3)]:
        print("Lead:", l)
        scores = evaluate_lead(verdicts, n=l)
        avg_scores = average(scores)
        print(avg_scores)
        acc_score.append({str(l[0])+"_"+str(l[1]): avg_scores})
    with io.open("logging/lead3_test_perf.json", "w+", encoding="utf-8") as f:
        json.dump(acc_score, f, sort_keys=False, indent=4, ensure_ascii=False)

def evaluate_randoms(verdicts: List[str]):
    print("Random")
    scores = evaluate_random(verdicts, False)
    avg_scores = average(scores)
    print(avg_scores)
    acc_score = [avg_scores]
    with io.open("logging/random_test_perf.json", "w+", encoding="utf-8") as f:
        json.dump(acc_score, f, sort_keys=False, indent=4, ensure_ascii=False)

def evaluate_ext_models():
    file = "logging/ext_test_perf.json"
    try:
        with io.open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except: 
        data = []
    for folder in os.listdir(Path("logging/runs/extractive"))[len(data):]:
        print(folder)
        params = parse_run_parameters(folder)
        model, embedding = reload_model(params)
        scores = evaluate_ext_model(model, embedding, get_test_files(), equal_length=False)
        avg_scores = average(scores)
        print(avg_scores)
        data.append({folder: avg_scores})
        with io.open(file, "w+", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=False, indent=4, ensure_ascii=False)

def evaluate_abs_models(test: bool=False):
    if test:
        file = "logging/abs_test_perf.json"
        file_function = get_test_files
    else:
        file = "logging/abs_perf.json"
        file_function = get_val_files

    try:
        with io.open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except: 
        data = []
    for folder in os.listdir(Path("logging/runs/abstractive"))[len(data):]:
        print(folder)
        params = parse_abs_run_parameters(folder)
        model, embedding = reload_abs_model(params)
        scores = evaluate_abs_model(model, embedding, file_function())
        avg_scores = average(scores)
        print(avg_scores)
        data.append({folder: avg_scores})
        with io.open(file, "w+", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=False, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    #evaluate_ext_models()
    #evaluate_leads(get_test_files())
    #evaluate_randoms(get_test_files())
    evaluate_abs_models()

# Script used to evaluate the different models on the validation and test set
import os
import io
import json
from typing import List, Dict
from collections import defaultdict
from pathlib import Path

from src.dataloading import get_val_files, get_test_files
from src.training import evaluate_ext_model, evaluate_lead, evaluate_random, evaluate_abs_model, evaluate_abs_model_beam, evaluate_template_model_beam
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
    file = "logging/ext_test_final_perf.json"
    try:
        with io.open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except: 
        data = []
    file_set = set()
    for x in data:
        for k in x:
            file_set.add(k)
    #for folder in os.listdir(Path("logging/runs/extractive")):
    for folder in ["21_03_2021__052013_model_HIER_lr_0.004780360096094034_abstractive_0_embedding_glove_attention_DOT_loss_type_BCE_target_ONE", "21_03_2021__071758_model_CNN_lr_0.0006914126259474318_abstractive_0_embedding_glove_attention_DOT_loss_type_BCE_target_ONE", "01_04_2021__161403_model_RNN_lr_0.00012153982361919725_abstractive_0_embedding_glove_attention_NONE_loss_type_BCE_target_ONE", "23_03_2021__170034_model_HIER_CNN_lr_0.005877329339332577_abstractive_0_embedding_glove_attention_DOT_loss_type_BCE_target_ONE", "23_03_2021__130758_model_HIER_RNN_lr_0.00022496321243825225_abstractive_0_embedding_glove_attention_DOT_loss_type_BCE_target_ONE", "23_03_2021__112800_model_CNN_CNN_lr_8.982024088259283e-05_abstractive_0_embedding_glove_attention_DOT_loss_type_BCE_target_ONE", "23_03_2021__043421_model_CNN_RNN_lr_1.7051404676904372e-05_abstractive_0_embedding_glove_attention_DOT_loss_type_BCE_target_ONE", "02_04_2021__142825_model_RNN_CNN_lr_0.0007555632457530207_abstractive_0_embedding_glove_attention_NONE_loss_type_BCE_target_ONE", "26_03_2021__161617_model_RNN_RNN_lr_1.5744493612097835e-05_abstractive_0_embedding_glove_attention_DOT_loss_type_BCE_target_ONE"]:
        if folder in file_set:
            continue
        print(folder)
        params = parse_run_parameters(folder)
        model, embedding = reload_model(params)
        scores = evaluate_ext_model(model, embedding, get_test_files(), equal_length=False)
        avg_scores = average(scores)
        print(avg_scores)
        data.append({folder: avg_scores})
        with io.open(file, "w+", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=False, indent=4, ensure_ascii=False)

def evaluate_abs_models(test: bool=True):
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
    for folder in ["14_04_2021__085315_model_GUIDED_ABS_GHIER_RNN_PRE_RNN_DEC_LIN_lr_0.0005_abstractive_1_embedding_glove_attention_DOT_loss_type_ABS_target_NONE"]:
        print(folder)
        params = parse_abs_run_parameters(folder)
        model, embedding = reload_abs_model(params)
        scores = evaluate_abs_model_beam(model, embedding, file_function())
        avg_scores = average(scores)
        print(avg_scores)
        data.append({folder: avg_scores})
        with io.open(file, "w+", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=False, indent=4, ensure_ascii=False)

def evaluate_template_models(test: bool=True):
    if test:
        file = "logging/abs_test_template_perf.json"
        file_function = get_test_files
    else:
        file = "logging/abs_template_perf.json"
        file_function = get_val_files

    try:
        with io.open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        data = []

    for folder in ["14_04_2021__084919_model_TEMPLATE_ABS_GHIER_RNN_PRE_RNN_DEC_TLIN_lr_0.0005_abstractive_1_embedding_glove_attention_DOT_loss_type_ABS_target_NONE"]:
        print(folder)
        params = parse_abs_run_parameters(folder)
        model, embedding = reload_abs_model(params)
        scores = evaluate_template_model_beam(model, embedding, file_function())
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

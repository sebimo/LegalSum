# Script used to evaluate the different models on the validation and test set
import os
import io
import json
from typing import List, Dict
from collections import defaultdict
from pathlib import Path
import re

import spacy
from spacy.lang.de import German
from tqdm import tqdm
import pytextrank
from lexrank import STOPWORDS, LexRank

from src.dataloading import get_val_files, get_test_files
from src.training import evaluate_ext_model, evaluate_lead, evaluate_random, evaluate_abs_model, evaluate_abs_model_beam, evaluate_template_model_beam
from src.evaluation import evaluate_prebuild, evaluate
from src.preprocessing import Tokenizer
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
    for folder in ["23_03_2021__130758_model_HIER_RNN_lr_0.00022496321243825225_abstractive_0_embedding_glove_attention_DOT_loss_type_BCE_target_ONE"]:
        #if folder in file_set:
        #    continue
        params = parse_run_parameters(folder)
        model, embedding = reload_model(params)
        scores = evaluate_ext_model(model, embedding, get_test_files(), equal_length=False)
        avg_scores = average(scores)
        print(avg_scores)
        data.append({folder: avg_scores})
        """with io.open(file, "w+", encoding="utf-8") as f:
            json.dump(data, f, sort_keys=False, indent=4, ensure_ascii=False)"""

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
    for folder in os.listdir(Path("logging/runs/abstractive")):
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


def evaluate_pointer():
    generated_folder = Path("../Summarization/logs/legal/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-206339/decoded")
    reference_folder = Path("../Summarization/logs/legal/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-206339/reference")
    generated_files = os.listdir(generated_folder)
    reference_files = os.listdir(reference_folder)
    scores = []
    for gen, ref in tqdm(zip(generated_files, reference_files)):
        assert gen[:-len("_decoded.txt")] == ref[:-len("_reference.txt")], gen + " " + ref
        with io.open(generated_folder/gen, "r", encoding="utf-8") as f:
            gen_text = f.readlines()
        gen_text = " ".join(gen_text)
        with io.open(reference_folder/ref, "r", encoding="utf-8") as f:
            ref_text = f.readlines()
        ref_text = " ".join(ref_text)
        scores.append(evaluate_prebuild([ref_text], [gen_text])[0])
    
    avg_scores = average(scores)
    print(avg_scores)

def evaluate_textrank():
    nlp = spacy.load("de_core_news_sm")
    
    # char count offsets to transfer the presegmented sentences to the spacy document
    SENTENCE_OFFSETS = []
    @German.component("Segmentation")
    def improve_segmentation(doc):
        cum_offsets = []
        sum = 0
        for offset in SENTENCE_OFFSETS:
            sum += offset
            cum_offsets.append(sum)
        
        sum = 0
        continue_token = False
        for i, token in enumerate(doc):
            tok_length = len(token.text_with_ws)
            # If we set the sentence start in the last iteration
            if continue_token:
                sum += tok_length
                continue_token = False
                continue

            # space is not included in the offset calculation, i.e. we can get the sentence by checking <=
            if cum_offsets[0] > sum and cum_offsets[0] <= sum+tok_length:
                if i+1 < len(doc):
                    doc[i+1].is_sent_start = True
                doc[i].is_sent_start = False
                continue_token = True

                cum_offsets.pop(0)
                if len(cum_offsets) <= 1:
                    break
            else:
                doc[i].is_sent_start = False
            sum += tok_length  

        return doc

    nlp.add_pipe("Segmentation", before="parser")
    nlp.add_pipe("textrank")    

    tok = Tokenizer(Path("model"), normalize=True)
    pattern = re.compile(r"<((num)|(norm)|(anon))>")

    MAX_NUM_SENTS = 3
    counter = 0
    scores = []
    for verdict in tqdm(get_test_files()):
        verdict = tok.tokenize_verdict_without_id(verdict)
        sentences = verdict["facts"] + verdict["reasoning"]
        sentences = list(map(lambda sentence: pattern.subn("", " ".join(sentence))[0], sentences))
        text = " ".join(sentences)
        SENTENCE_OFFSETS = list(map(lambda sentence: len(sentence)+1, sentences))
        doc = nlp(text)

        selected_sentences = []
        for sent in doc._.textrank.summary(limit_sentences=MAX_NUM_SENTS):
            selected_sentences += sent.text.split()

        labels = []
        for sent in verdict["guiding_principle"]:
            labels += sent

        if len(selected_sentences) == 0:
            selected_sentences = ["<unk>"]

        try:
            score = evaluate([labels], [selected_sentences])[0]
            scores.append(score)
        except RecursionError:
            counter += 1
            continue

    print("Excluded", counter)
    avg_scores = average(scores)
    print(avg_scores)

def evaluate_lexrank():
    tok = Tokenizer(Path("model"), normalize=True)

    MAX_NUM_SENTS = 3
    scores = []
    docs = []
    for verdict in tqdm(get_test_files()):
        verdict = tok.tokenize_verdict_without_id(verdict)
        sentences = list(map(lambda sentence: " ".join(sentence), verdict["facts"] + verdict["reasoning"]))
        docs.append(sentences)

    lxr = LexRank(docs, stopwords=STOPWORDS["de"])

    for verdict in tqdm(get_test_files()):
        verdict = tok.tokenize_verdict_without_id(verdict)
        sentences = list(map(lambda sentence: " ".join(sentence), verdict["facts"] + verdict["reasoning"]))

        selected_sentences = lxr.get_summary(sentences, summary_size=MAX_NUM_SENTS, threshold=.1)
        sentences = list(map(lambda sentence: sentence.split(), selected_sentences))
        selected_sentences = []
        for sentence in sentences:
            selected_sentences += sentence

        labels = []
        for sent in verdict["guiding_principle"]:
            labels += sent

        if len(selected_sentences) == 0:
            selected_sentences = ["<unk>"]

        try:
            score = evaluate([labels], [selected_sentences])[0]
            scores.append(score)
        except RecursionError:
            continue

    avg_scores = average(scores)
    print(avg_scores)

if __name__ == "__main__":
    evaluate_lexrank()
    evaluate_textrank()
    evaluate_ext_models()
    evaluate_leads(get_test_files())
    evaluate_randoms(get_test_files())
    evaluate_abs_models()
    evaluate_pointer()

from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from rouge import Rouge # https://github.com/pltrdy/rouge

rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])

def evaluate(labels: List[List[str]], predictions: List[List[str]]) -> List[Dict[str, Dict[str, float]]]:
    """ Wrapper function around the rouge implementation, as it expects a continous text, we only get tokens 
        Compares each label sentence with its corresponding prediction 
        Returns:
            - List for each pair of sentences with dict containing all metrics; each metric mapping to f, r, p
              for f1, recall, precision of sentence pair and metric
    """
    labels = list(map(lambda sentence: " ".join(sentence), labels))
    predictions = list(map(lambda sentence: " ".join(sentence), predictions))
    assert len(labels) == len(predictions)
    return rouge.get_scores(labels, predictions)

def evaluate_indices(labels: List[List[int]], predictions: List[List[int]]) -> List[Dict[str, Dict[str, float]]]:
    """ Wrapper function around rouge implementation, as it expects a continuous text.
        If we only have indices and do not want to retranslate them to tokens, we could just compare the resulting number sequence.
        ATTENTION: Do not use as final evaluation, as the scores are highly dependent on the current system setup (Tokenizer,...)!
        Returns:
            - List for each pair of sentences with dict containing all metrics; each metric mapping to f, r, p
              for f1, recall, precision of sentence pair and metric
    """
    labels = list(map(lambda sentence: " ".join(map(lambda tok: str(tok), sentence)), labels))
    predictions = list(map(lambda sentence: " ".join(map(lambda tok: str(tok), sentence)), predictions))
    assert len(labels) == len(predictions)
    return rouge.get_scores(labels, predictions)

def set_metrics(metrics: List[str]=["rouge-1", "rouge-2", "rouge-l"]):
    """ Resets the metrics returned by evaluate """
    global rouge
    old_metrics = rouge.metrics
    try:
        rouge = Rouge(metrics=metrics)
    except ValueError:
        print("Used unknown metrics; reapplying old ones")
        rouge = Rouge(metrics=old_metrics)

def calculate_confusion_matrix(y_true: np.array, y_pred: np.array) -> Dict[str, int]:
    """ This is just a wrapper around the sklearn confusion matrix, transformed to a dict to have a unified statistics format """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        # Error: confusion_matrix does not return tuple of size 4
        # We could also just set the res values here to some arbitrary error value?
        # Because it seems that the confusion_matrix will only return a smaller tuple, if there are no predictions?
        print(y_true)
        print(y_pred)
        raise ValueError
    res = {}
    res["TN"] = tn
    res["FP"] = fp
    res["FN"] = fn
    res["TP"] = tp
    return res

def merge(batch_stats: Dict[str, Tuple[float, int]], stats: Dict[str, float]) -> Dict[str, Tuple[float, int]]:
    """ Combines the local batch stats with the epoch stats """
    result = {}
    if len(batch_stats) == 0:
        for k in stats:
            result[k] = (stats[k], 1)
    else:
        for k in stats:
            # For direct counter variables, we do not want to increase the count
            if k in ["TP", "TN", "FP", "FN"]:
                inc = 0
            else:
                inc = 1

            result[k] = (stats[k]+batch_stats[k][0], batch_stats[k][1]+inc)
    return result

def merge_epoch(epoch_stats: Dict[str, Tuple[float, int]], stats: Dict[str, Tuple[float, int]]) -> Dict[str, Tuple[float, int]]:
    """ Combines the local batch stats with the epoch stats """
    result = {}
    if len(epoch_stats) == 0:
        for k in stats:
            result[k] = stats[k]
    else:
        for k in stats:
            # For direct counter variables, we do not want to increase the count
            if k not in ["TP", "TN", "FP", "FN"]:
                result[k] = (stats[k][0] + epoch_stats[k][0], stats[k][1] + epoch_stats[k][1])
            else:
                result[k] = (stats[k][0] + epoch_stats[k][0], epoch_stats[k][1])
    
    return result

def finalize_statistic(epoch_stats: Dict[str, Tuple[float, int]]) -> Dict[str, float]:
    result = {}
    for k in epoch_stats:
        value, count = epoch_stats[k]
        # Generally, we are just going for the average value
        result[k] = value/count
    if "TP" in result and "FP" in result and "TN" in result and "FN" in result and "Recall" not in result and "Precision" not in result and "F1" not in result:
        result["Recall"] = result["TP"]/(result["TP"]+result["FN"]) if result["TP"]+result["FN"] > 0 else 0
        result["Precision"] = result["TP"]/(result["TP"]+result["FP"]) if result["TP"]+result["FP"] > 0 else 0
        result["F1"] = 2 * (result["Recall"]*result["Precision"])/(result["Recall"] + result["Precision"]) if result["Recall"] + result["Precision"] > 0 else 0
    return result


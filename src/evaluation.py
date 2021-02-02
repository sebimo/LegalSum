from typing import List, Dict

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

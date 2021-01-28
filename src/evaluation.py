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

def set_metrics(metrics: List[str]=["rouge-1", "rouge-2", "rouge-l"]):
    """ Resets the metrics returned by evaluate """
    global rouge
    old_metrics = rouge.metrics
    try:
        rouge = Rouge(metrics=metrics)
    except ValueError:
        print("Used unknown metrics; reapplying old ones")
        rouge = Rouge(metrics=old_metrics)

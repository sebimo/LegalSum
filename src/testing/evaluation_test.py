import pytest

import numpy as np
import sklearn.metrics as metrics

from ..evaluation import evaluate, evaluate_indices, set_metrics, merge, finalize_statistic, calculate_confusion_matrix

class TestRouge:

    def test_basics(self):
        h = [["Hello", "World"]]
        t = [["Die", "Würde", "des", "Menschen"]]
        res = evaluate(h, t)[0]
        for k in res:
            assert res[k]["f"] == 0.0

        h = [["Die", "World"]]
        t = [["Die", "Würde", "des", "Menschen"]]
        res = evaluate(h, t)[0]
        assert res["rouge-1"]["f"] > 0.0
        assert res["rouge-2"]["f"] == 0.0
        assert res["rouge-l"]["f"] > 0.0

    def test_metric_change(self):
        set_metrics(["rouge-1"])
        h = [["Hello", "World"]]
        t = [["Die", "Würde", "des", "Menschen"]]
        res = evaluate(h, t)[0]
        assert len(res) == 1
        assert "rouge-1" in res
        set_metrics(["error"])
        h = [["Hello", "World"]]
        t = [["Die", "Würde", "des", "Menschen"]]
        res = evaluate(h, t)[0]
        assert len(res) == 1
        assert "rouge-1" in res
        assert "error" not in res

    def test_evaluate_indices(self):
        set_metrics(["rouge-1", "rouge-2", "rouge-l"])
        h = [[1, 2], [1, 3, 4, 5]]
        t = [[1, 3], [1, 3, 4, 5]]
        res = evaluate_indices(h, t)
        assert res[0]["rouge-1"]["f"] > 0.0
        assert res[0]["rouge-2"]["f"] == 0.0
        assert res[0]["rouge-l"]["f"] > 0.0

        assert res[1]["rouge-1"]["f"] > 0.0
        assert res[1]["rouge-2"]["f"] > 0.0
        assert res[1]["rouge-l"]["f"] > 0.0

    def test_confusion_matrix(self):
        t = np.array([1, 0, 0, 1, 1, 0, 0])
        p = np.array([1, 1, 0, 0, 1, 0, 0])
        res = {
            "TP": 2,
            "FP": 1,
            "FN": 1,
            "TN": 3
        }
        y = calculate_confusion_matrix(t, p)
        assert len(res) == len(y)
        for k in res:
            assert res[k] == y[k]

    def test_merge(self):
        e = {}
        b = {
            "TP": 10,
            "FP": 20,
            "TN": 30,
            "FN": 2,
            "Test": 0.5
        }
        res = {
            "TP": (10, 1),
            "FP": (20, 1),
            "TN": (30, 1),
            "FN": (2, 1),
            "Test": (0.5, 1)
        }
        y = merge(e, b)
        for k in res:
            assert res[k] == y[k]
        b = {
            "TP": 10,
            "FP": 20,
            "TN": 30,
            "FN": 2,
            "Test": 0.6
        }
        y = merge(res, b)
        res = {
            "TP": (20, 1),
            "FP": (40, 1),
            "TN": (60, 1),
            "FN": (4, 1),
            "Test": (1.1, 2)
        }
        for k in res:
            assert res[k] == y[k]
        
    def test_finalize_statistic(self):
        e = {
            "TP": (20, 1),
            "FP": (40, 1),
            "TN": (60, 1),
            "FN": (4, 1),
            "Test": (1.1, 2)
        }
        res = {
            "TP": 20,
            "FP": 40,
            "TN": 60,
            "FN": 4,
            "Test": 0.55,
            "Recall": 5/6,
            "Precision": 1/3,
            "F1": 10/21
        }
        y = finalize_statistic(e)
        for k in res:
            assert res[k] == y[k]

        e = {
            "TP": (20, 1),
            "FP": (0, 1),
            "TN": (0, 1),
            "FN": (0, 1)
        }
        res = finalize_statistic(e)
        assert res["F1"] == 1.0
        assert res["Recall"] == 1.0
        assert res["Precision"] == 1.0

        e = {
            "TP": (0, 1),
            "FP": (0, 1),
            "TN": (20, 1),
            "FN": (0, 1)
        } 
        # This case is really senseless from a prediction standpoint, as we do not have any targets
        # But we still need to check that our implementation does the same as the sklearn implementation
        res = finalize_statistic(e)
        assert res["F1"] == 0.0
        assert res["Recall"] == 0.0
        assert res["Precision"] == 0.0

        for _ in range(10):
            # The upperbound (2 in this case) is exclusive
            t = np.random.randint(0,2,100)
            p = np.random.randint(0,2,100)
            d = calculate_confusion_matrix(t, p)
            stats = merge({}, d)
            stats = finalize_statistic(stats)
            assert stats["F1"] == metrics.f1_score(t, p)
            assert stats["Precision"] == metrics.precision_score(t, p)
            assert stats["Recall"] == metrics.recall_score(t, p)


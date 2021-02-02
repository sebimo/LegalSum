import pytest

from ..evaluation import evaluate, evaluate_indices, set_metrics

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

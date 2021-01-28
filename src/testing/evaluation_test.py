import pytest

from ..evaluation import evaluate, set_metrics

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

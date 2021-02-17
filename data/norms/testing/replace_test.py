import pytest

from ..replace import split, process_normchain
from ..normdb import NormDBStub

class Test:

    def test_split(self):
        t = ["Hallo Welt, die"]
        res = [["Hallo", "Welt,", "die"]]
        assert list(split(t)) == res
        t = ["Hallo Welt, die"]
        res = [["Hallo", "Welt,", "die"]]
        assert list(split(t)) == res
        t = ["Hallo Welt,\n\t \vdie"]
        res = [["Hallo", "Welt,", "die"]]
        assert list(split(t)) == res
        # Multiple sentences
        t = ["Hallo Welt,\n\t \vdie", "Das ist Satz 2."]
        res = [["Hallo", "Welt,", "die"], ["Das", "ist", "Satz", "2."]]
        assert list(split(t)) == res
        t = ["I.Die Revision hat Erfolg."]
        res = [["I.Die", "Revision", "hat", "Erfolg."]]
        assert list(split(t)) == res
        t = ["Herr ... hat nichts getan."]
        res = [["Herr", "...", "hat", "nichts", "getan."]]
        assert list(split(t)) == res

    def test_normchain(self):
        db = NormDBStub()
        t = ["RL 96/34/EG des Rates; BBesG §13"]
        r, n = process_normchain(t, db)
        assert r == ["RL 96/34/EG des Rates", "BBesG §13"] 
        assert n == {"__norm1__": "RL 96/34/EG des Rates", "__norm2__": "BBesG §13"}
        t2 = ["SUrlV § 13 Abs. 1, SUrlV § 15"]
        r, n = process_normchain(t2, db)
        assert r == ["SUrlV § 13 Abs. 1", "SUrlV § 15"] 
        assert n == {"__norm3__": "SUrlV § 13 Abs. 1", "__norm4__": "SUrlV § 15"}
        t3 = ["§§ 46, 46a, 47 BRAO"]
        r, n = process_normchain(t3, db)
        assert r == ["§ 46 BRAO", "§ 46a BRAO", "§ 47 BRAO"]
        assert n == {"__norm5__": "§ 46 BRAO", "__norm6__": "§ 46a BRAO", "__norm7__": "§ 47 BRAO"}
        t3 = ["§§ 46, 46a BRAO"]
        r, n = process_normchain(t3, db)
        assert r == ["§ 46 BRAO", "§ 46a BRAO"]
        assert n == {"__norm5__": "§ 46 BRAO", "__norm6__": "§ 46a BRAO"}
        t4 = ["__norm12969__"]
        r, n = process_normchain(t4, db)
        assert r == t4
        assert n == {}
        t5 = [
            "__norm1672__",
            "__norm257023__",
            "__norm156677__",
            "__norm162047__",
            "__norm257024__"
        ]
        r, n = process_normchain(t5, db)
        assert r == t5
        assert n == {}

import pytest

from ..replace import split, process_normchain, process_sentence
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
        assert r == ["BRAO § 46", "BRAO § 46a", "BRAO § 47"]
        assert n == {"__norm5__": "BRAO § 46", "__norm6__": "BRAO § 46a", "__norm7__": "BRAO § 47"}
        t3 = ["§§ 46, 46a BRAO"]
        r, n = process_normchain(t3, db)
        assert r == ["BRAO § 46", "BRAO § 46a"]
        assert n == {"__norm5__": "BRAO § 46", "__norm6__": "BRAO § 46a"}
        # Mixture:
        t = ["§§ 46, 46a BRAO; SUrlV § 13 Abs. 1, SUrlV § 15"]
        r, _ = process_normchain(t, db)
        assert r == ["BRAO § 46", "BRAO § 46a", "SUrlV § 13 Abs. 1", "SUrlV § 15"]
        t = ["SUrlV § 13 Abs. 1, SUrlV § 15;§§ 46, 46a BRAO" ]
        r, _ = process_normchain(t, db)
        assert r == ["SUrlV § 13 Abs. 1", "SUrlV § 15", "BRAO § 46", "BRAO § 46a"]
        t = ["SUrlV § 13 Abs. 1, SUrlV § 15, §§ 46, 46a BRAO" ]
        r, _ = process_normchain(t, db)
        assert r == ["SUrlV § 13 Abs. 1", "SUrlV § 15", "BRAO § 46", "BRAO § 46a"]
        t = ["§§ 46, 46a BRAO, SUrlV § 13 Abs. 1, SUrlV § 15"]
        r, _ = process_normchain(t, db)
        assert r == ["BRAO § 46", "BRAO § 46a", "SUrlV § 13 Abs. 1", "SUrlV § 15"]
        # Known norms
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

    def test_sentence(self):
        db = NormDBStub()
        sentence = "Zulassungsgrund nach §124 Abs.2 Nr.1 VwGO ist nicht hinreichend dargelegt"
        res, norms = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach __norm1__ ist nicht hinreichend dargelegt"

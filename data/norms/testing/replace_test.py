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
        t = ["BGB §§ 434, 437, 440, 323, 278, 823"]
        r, _ = process_normchain(t, db)
        assert r == ["BGB § 434", "BGB § 437", "BGB § 440", "BGB § 323", "BGB § 278", "BGB § 823"]
        
        # Edge cases:
        t = ["Normen: BGB §§ 434, 437, 440, 323, 278, 823"]
        r, _ = process_normchain(t, db)
        assert r == ["BGB § 434", "BGB § 437", "BGB § 440", "BGB § 323", "BGB § 278", "BGB § 823"]
        t = ["Gesetz: BGB §§ 434, 437, 440, 323, 278, 823"]
        r, _ = process_normchain(t, db)
        assert r == ["BGB § 434", "BGB § 437", "BGB § 440", "BGB § 323", "BGB § 278", "BGB § 823"]
        t = ["VwGO § 167 Abs. S. 1, ZPO § 766 Abs. 1, ZPO § 795, AGB § 362 Abs. 1"]
        r, _ = process_normchain(t, db)
        assert r == ["VwGO § 167 Abs. S. 1", "ZPO § 766 Abs. 1", "ZPO § 795", "AGB § 362 Abs. 1"]
        t = ["BGB §§ 133, 157;BGB § 611;BAT SR 2 l II; Änderungs-Tarifvertrag zum BAT § 2"]
        r, _ = process_normchain(t, db)
        assert r == ["BGB § 133", "BGB § 157", "BGB § 611", "BAT SR 2 l II", "Änderungs-Tarifvertrag zum BAT § 2"]
        t = ["§§ 242, 355, 495 BGB, Art. 247 § 6 Abs. 2, 9 Abs. 1 EGBGB"]
        r, _ = process_normchain(t, db)
        # One of the parsing edge cases which would be to difficult to handle all the cases due to time constraints
        assert r == ["BGB § 242", "BGB § 355", "BGB § 495", "Art. 247 § 6 Abs. 2, 9 Abs. 1 EGBGB"]
        t = ["StVO §§ 37 Abs. 2, 49"]
        r, _ = process_normchain(t, db)
        assert r == ["StVO § 37 Abs. 2", "StVO § 49"]
        t = ["§§ 242, 355, 495 BGB, Art. 247 § 6 Abs. 2, 9 Abs. 1 EGBGB, GG § 20"]
        r, _ = process_normchain(t, db)
        assert r == ["BGB § 242", "BGB § 355", "BGB § 495", "Art. 247 § 6 Abs. 2, 9 Abs. 1 EGBGB", "GG § 20"]
        t = ["GKG §§ 50, 54, § 58 Abs. 2 InsO; § 34 Abs. 1, § 99 ZPO"]
        r, _ = process_normchain(t, db)
        assert r == ["GKG § 50", "GKG § 54", "InsO § 58 Abs. 2", "ZPO § 34 Abs. 1", "ZPO § 99"]
        t = ["§ 266 StGB, § 7 d SGB"]
        r, _ = process_normchain(t, db)
        assert r == ["StGB § 266", "SGB § 7 d"]
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
        assert len(norms) == 1
        assert norms["__norm1__"] == "§124 Abs.2 Nr.1 VwGO"
        sentence = "Zulassungsgrund nach §124 Abs.2 Nr.1 ist nicht hinreichend dargelegt"
        res, norms = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach __norm2__ ist nicht hinreichend dargelegt"
        # Multi-Norms -> those will be replaced with one unique identifier, as we only want to filer them from the text
        sentence = "Zulassungsgrund nach § 124 BGB, § 125 ZPO"
        res, _ = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach __norm3__"
        # Check for presegmented cases
        sentence = "Zulassungsgrund nach § __norm123__"
        res, _ = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach § __norm123__"
        # Check for single characters and numbers
        sentence = "Zulassungsgrund nach § 1 Abs. A b 23 GG"
        res, _ = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach __norm4__"
        sentence = "Zulassungsgrund nach § 1 Abs. 1a 23 GG"
        res, _ = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach __norm5__"


import pytest

from ..replace import split, process_normchain, process_sentence, process_segment, get_special_after, get_special_before
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
        t = ["WaffG § 45 Abs. 2, 5 Abs. 1 Nr. 2; StAG § 1, § 30"]
        r, _ = process_normchain(t, db)
        assert r == ["WaffG § 45 Abs. 2", "WaffG § 5 Abs. 1 Nr. 2", "StAG § 1", "StAG § 30"]
        
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
        # Empty norms
        t = ["GKG § 50;;GKG § 54"]
        r, n = process_normchain(t, db)
        assert r == ["GKG § 50", "GKG § 54"]

    def test_sentence(self):
        db = NormDBStub()
        sentence = "Zulassungsgrund nach §124 Abs.2 Nr.1 VwGO ist nicht hinreichend dargelegt"
        res, norms = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach __norm1__ ist nicht hinreichend dargelegt"
        assert len(norms) == 1
        assert norms["__norm1__"] == "§124 Abs.2 Nr.1 VwGO"
        sentence = "Zulassungsgrund nach §124 Abs.2 Nr.1 GG ist nicht hinreichend dargelegt"
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
        # Check for special characters at beginning or end or norm
        sentence = "Zulassungsgrund nach (§ 1 Abs. 1a 23 GG)"
        res, _ = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach (__norm5__)"
        sentence = "Zulassungsgrund nach § 1 Abs. 1a 23 GG."
        res, _ = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach __norm5__."
        # Check with special characters
        sentence = "Zulassungsgrund nach § 1 Abs. 1a 23 ist nicht gültig."
        res, _ = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach __norm6__ ist nicht gültig."
        sentence = "Zulassungsgrund nach (§ 1 Abs. 1a 23) ist nicht gültig."
        res, _ = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach (__norm6__) ist nicht gültig."
        sentence = "Zulassungsgrund nach BGB § 124 Abs. 2 ist nicht hinreichend dargelegt"
        res, _ = process_sentence(sentence, db)
        assert res == "Zulassungsgrund nach __norm7__ ist nicht hinreichend dargelegt"
        sentence = "15.3. zu § 15 ZVG hingewiesen."
        res, _ = process_sentence(sentence, db)
        assert res == "15.3. zu __norm8__ hingewiesen."
        sentence = "Überdies ist auch strittig, ob und ggf. in welchen Fällen ein Nachweisverzicht im Sinne von § 726 Abs. 1 ZPO statthaft und wirksam ist (siehe insoweit auch die Ausführungen in der von der Gläubigerseite benannten Entscheidung des Oberlandesgerichts München vom 23.6.2016 – 34 Wx 189/16)."
        res, _ = process_sentence(sentence, db)
        assert res == "Überdies ist auch strittig, ob und ggf. in welchen Fällen ein Nachweisverzicht im Sinne von __norm9__ statthaft und wirksam ist (siehe insoweit auch die Ausführungen in der von der Gläubigerseite benannten Entscheidung des Oberlandesgerichts München vom 23.6.2016 – 34 Wx 189/16)."
        sentence = "Ablauf der 6-monatigen Kündigungsfrist gem. § 1193 Abs. 1 BGB stellen dabei eine aufschiebende Bedingung im Sinne von __norm22__ dar"
        res, _ = process_sentence(sentence, db)
        assert res == "Ablauf der 6-monatigen Kündigungsfrist gem. __norm10__ stellen dabei eine aufschiebende Bedingung im Sinne von __norm22__ dar"
        sentence = "Nachweises der Fälligkeit deren Nichtvorliegen für den Notar eindeutig erkennbar auf der Hand lag."
        res, _ = process_sentence(sentence, db)
        assert res == "Nachweises der Fälligkeit deren Nichtvorliegen für den Notar eindeutig erkennbar auf der Hand lag."
        sentence = "Nacherfüllung gänzlich unmöglich wäre (§ 275 Abs. 1 BGB) – bedarf deshalb keiner Entscheidung."
        res, _ = process_sentence(sentence, db)
        assert res == "Nacherfüllung gänzlich unmöglich wäre (__norm11__) – bedarf deshalb keiner Entscheidung."
        sentence = "Fristsetzung unter den in § 323 Abs. 2 BGB und § 440 BGB abschließend geregelten Voraussetzungen"
        res, _ = process_sentence(sentence, db)
        assert res == "Fristsetzung unter den in __norm12__ abschließend geregelten Voraussetzungen"
        sentence = "deutschen Rechtsordnung nicht fremd (z.B. § 38 BZRG) und vom Verurteilten"
        res, _ = process_sentence(sentence, db)
        assert res == "deutschen Rechtsordnung nicht fremd (z.B. __norm13__) und vom Verurteilten"

    def test_special_separation(self):
        tok = "(BGB"
        res1 = get_special_before(tok)
        res2 = get_special_after(tok)
        assert res1 == ("(", "BGB")
        assert res2 == ("", "(BGB")
        tok = "((BGB"
        res1 = get_special_before(tok)
        res2 = get_special_after(tok)
        assert res1 == ("((", "BGB")
        assert res2 == ("", "((BGB")
        tok = "(BGB)"
        res1 = get_special_before(tok)
        res2 = get_special_after(tok)
        assert res1 == ("(", "BGB)")
        assert res2 == (")", "(BGB")
        tok = "(BGB))"
        res1 = get_special_before(tok)
        res2 = get_special_after(tok)
        assert res1 == ("(", "BGB))")
        assert res2 == ("))", "(BGB")
        tok = "BGB."
        res1 = get_special_before(tok)
        res2 = get_special_after(tok)
        assert res1 == ("", "BGB.")
        assert res2 == (".", "BGB")
        tok = "2BGB2"
        res1 = get_special_before(tok)
        res2 = get_special_after(tok)
        assert res1 == ("", "2BGB2")
        assert res2 == ("", "2BGB2")
        tok = "§"
        res1 = get_special_before(tok)
        res2 = get_special_after(tok)
        assert res1 == ("", "§")
        assert res2 == ("", "§")


    def test_segment(self):
        db = NormDBStub()
        segment = [
            "Der zulässige Antrag, über den im Einverständnis der Beteiligten der Berichterstatter anstelle des Senats entscheidet (§§ 87a Abs. 2 und 3, 125 Abs. 1 VwGO), ist unbegründet.",
            "Das Antragsvorbringen führt nicht auf einen der in Anspruch genommenen Zulassungsgründe der ernstlichen Zweifel an der Richtigkeit der erstinstanzlichen Entscheidung (§ 124 Abs. 2 Nr. 1 VwGO, nachfolgend unter 1.), des Vorliegens besonderer tatsächlicher oder rechtlicher Schwierigkeiten der Rechtssache (§ 124 Abs.2 Nr. 2 VwGO, nachfolgend unter 2.) oder ihrer grundsätzlichen Bedeutung (§124 Abs. 2 Nr. 3 VwGO, nachfolgend unter 3.).",
            "1. Es bestehen zunächst keine ernstlichen Zweifel an der Richtigkeit der erstinstanzlichen Entscheidung im Sinne des § 124 Abs. 2 Nr. 1 VwGO."
        ]
        res, _ = process_segment(segment, db)
        y = [
            "Der zulässige Antrag, über den im Einverständnis der Beteiligten der Berichterstatter anstelle des Senats entscheidet (__norm1__), ist unbegründet.",
            "Das Antragsvorbringen führt nicht auf einen der in Anspruch genommenen Zulassungsgründe der ernstlichen Zweifel an der Richtigkeit der erstinstanzlichen Entscheidung (__norm2__, nachfolgend unter 1.), des Vorliegens besonderer tatsächlicher oder rechtlicher Schwierigkeiten der Rechtssache (__norm3__, nachfolgend unter 2.) oder ihrer grundsätzlichen Bedeutung (__norm4__, nachfolgend unter 3.).",
            "1. Es bestehen zunächst keine ernstlichen Zweifel an der Richtigkeit der erstinstanzlichen Entscheidung im Sinne des __norm2__."
        ]
        assert res == y

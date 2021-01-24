import pytest
from pathlib import Path

from ..preprocessing import is_roman_enum, is_alpha_enum, remove_special_characters, finalize, replace_tokens, split, load_verdict, process_segment

class TestTokenization:

    def setup(self):
        pass

    def test_roman_enum(self):
        # Uppercase
        assert is_roman_enum("I")
        assert is_roman_enum("II")
        assert is_roman_enum("III")
        assert is_roman_enum("IV")
        assert is_roman_enum("V")
        # Lowercase
        assert is_roman_enum("ii")
        assert is_roman_enum("iv")
        assert is_roman_enum("i")
        # Counterexample
        assert not is_roman_enum("1")
        assert not is_roman_enum("In")
        # We exclude this case as we have already remove special characters
        assert not is_roman_enum("I.")

    def test_alpha_enum(self):
        # Uppercase
        assert is_alpha_enum("A")
        assert is_alpha_enum("B")
        assert is_alpha_enum("C")
        assert is_alpha_enum("D")
        # Lowercase
        assert is_alpha_enum("a")
        assert is_alpha_enum("b")
        assert is_alpha_enum("c")
        # Double
        assert is_alpha_enum("aa")
        assert is_alpha_enum("bb")
        # Counterexamples
        assert not is_alpha_enum("ab")

    def test_remove_special(self):
        t = [[]]
        res = [[]]
        assert list(remove_special_characters(t)) == res
        t = [["<num>", "<anon>", "<norm>"]]
        res = [["<num>", "<anon>", "<norm>"]]
        assert list(remove_special_characters(t)) == res
        t = [["Die"]]
        res = [["Die"]]
        assert list(remove_special_characters(t)) == res
        # Testing special characters
        t = [["Dieser", "Text", "ist", "sichtbar"]]
        res = [["Dieser", "Text", "ist", "sichtbar"]]
        assert list(remove_special_characters(t)) == res 
        t = [["Hallo.", "Dieser", "Text", "ist", "sichtbar"]]
        res = [["Hallo", "Dieser", "Text", "ist", "sichtbar"]]
        assert list(remove_special_characters(t)) == res
        t = [["Some", "special", "characters:", "-", ";", ")", "Afterwards"]]
        res = [["Some", "special", "characters", "Afterwards"]]
        # Testing parentheses
        t = [["Die", "Würde", "(auch", "Respekt;", "Leben", "oder", "Test)", "des", "Menschen"]]
        res = [["Die", "Würde", "des", "Menschen"]]
        assert list(remove_special_characters(t)) == res
        t = [["Die", "Würde", "(auch)", "-Respekt", "Leben;", "(oder", "Test)", "des", "Menschen"]]
        res = [["Die", "Würde", "Respekt", "Leben", "des", "Menschen"]]
        assert list(remove_special_characters(t)), res
        t = [["Was", "passiert", "wenn", "(die", "Klammer", "nur", "aufgeht"]]
        res = [["Was", "passiert", "wenn"]]
        assert list(remove_special_characters(t)) == res
        # Mutli-Sentences
        t = [["Hallo.", "Dieser", "Text", "ist", "sichtbar"],
            ["Was", "passiert", "wenn", "(die", "Klammer", "nur", "aufgeht"],
            ["Die", "Würde", "(auch)", "-Respekt", "Leben;", "(oder", "Test)", "des", "Menschen"]]
        res = [["Hallo", "Dieser", "Text", "ist", "sichtbar"],
              ["Was", "passiert", "wenn"],
              ["Die", "Würde", "Respekt", "Leben", "des", "Menschen"]]
        assert list(remove_special_characters(t)) == res

    def test_finalize(self):
        t = [[""]]
        res = []
        assert list(finalize(t)) == res
        t = [["<num>", "<anon>", "<norm>"]]
        res = [["<anon>", "<norm>"]]
        assert list(finalize(t)) == res
        t = [["Die", ""]]
        res = [["Die"]]
        assert list(finalize(t)) == res
        t = [["<num>" , "Wird", "eine", "Grunddienstbarkeit", "nach"]]
        res = [["Wird", "eine", "Grunddienstbarkeit", "nach"]]
        assert list(finalize(t)) == res
        t = [["I" , "Wird", "eine", "Grunddienstbarkeit", "nach"], [], [""]]
        res = [["Wird", "eine", "Grunddienstbarkeit", "nach"]]
        assert list(finalize(t)) == res

    def test_replace_tokens(self):
        t = [["Wird", "2", "zu", "3?"]]
        res = [["Wird", "<num>", "zu", "<num>"]]
        assert list(replace_tokens(t)) == res
        t = [["Frau", "...", "wird", "zu", "zwei", "Jahren", "Haft", "verurteilt."],
            ["Dies", "beruht", "auf", "__norm1__"]]
        res = [["Frau", "<anon>", "wird", "zu", "zwei", "Jahren", "Haft", "verurteilt."],
            ["Dies", "beruht", "auf", "<norm>"]]
        assert list(replace_tokens(t)) == res
        t = [["Lass", "uns", "ein", "paar", "Daten:", "12.3.2020"]]
        res = [["Lass", "uns", "ein", "paar", "Daten:", "<num>"]]
        assert list(replace_tokens(t)) == res
        t = [["Lass", "uns", "..", "ein", "paar", "Daten:", "12.3.2020"]]
        res = [["Lass", "uns", "..", "ein", "paar", "Daten:", "<num>"]]
        assert list(replace_tokens(t)) == res

    def test_split(self):
        t = ["Hallo Welt, die"]
        res = [["hallo", "welt,", "die"]]
        assert list(split(t)) == res
        t = ["Hallo Welt, die"]
        res = [["Hallo", "Welt,", "die"]]
        assert list(split(t, normalize=False)) == res
        t = ["Hallo Welt,\n\t \vdie"]
        res = [["hallo", "welt,", "die"]]
        assert list(split(t)) == res
        # Multiple sentences
        t = ["Hallo Welt,\n\t \vdie", "Das ist Satz 2."]
        res = [["hallo", "welt,", "die"], ["das", "ist", "satz", "2."]]
        assert list(split(t)) == res
        t = ["Hallo Welt,\n\t \vdie", "Das ist Satz 2."]
        res = [["Hallo", "Welt,", "die"], ["Das", "ist", "Satz", "2."]]
        assert list(split(t, normalize=False)) == res

    def test_segment(self):
        t = [
            "1. Wird eine Grunddienstbarkeit nach Teilung des dienenden Grundstücks an einem Teil gelöscht.",
            "2. Gegen die Eintragung des Widerspruchs kann mit dem Ziel, diesen zu löschen, unbeschränkte Beschwerde erhoben werden."
        ]
        res = [
            ["wird", "eine", "grunddienstbarkeit", "nach", "teilung", "des", "dienenden", "grundstücks", "an", "einem", "teil", "gelöscht"],
            ["gegen", "die", "eintragung", "des", "widerspruchs", "kann", "mit", "dem", "ziel", "diesen", "zu", "löschen", "unbeschränkte", "beschwerde", "erhoben", "werden"]
        ]
        assert list(process_segment(t)) == res
        t = [
            "I.",
            "Die Beteiligte ist als Eigentümerin vom Grundbesitz im Grundbuch eingetragen.",
            "Ende 2007 verkaufte sie hiervon Teilflächen, welche nach Messungsanerkennung und Auflassung vom 28.2.2008 nun das Grundstück FlSt. 12/2 bilden.",
            "Für die jeweiligen Eigentümer des neu geschaffenen Grundstücks wurde eine Grunddienstbarkeit (Geh- und Fahrtrecht) an dem Grundstück FlSt. 12 der Beteiligten mit folgendem Inhalt (Abschnitt X.2. der Urkunde vom 10.12.2007) bestellt:",
            "Der jeweilige Eigentümer des herrschenden Grundstücks ist berechtigt neben dem Eigentümer des dienenden Grundstücks, auf dem dienenden Grundstück zu gehen und mit Fahrzeugen aller Art zu fahren."
        ]
        res = [
            ["Die", "Beteiligte", "ist", "als", "Eigentümerin", "vom", "Grundbesitz", "im", "Grundbuch", "eingetragen"],
            ["Ende", "<num>", "verkaufte", "sie", "hiervon", "Teilflächen", "welche", "nach", "Messungsanerkennung", "und", "Auflassung", "vom", "<num>", "nun", "das", "Grundstück", "FlSt", "<num>", "bilden"],
            ["Für", "die", "jeweiligen", "Eigentümer", "des", "neu", "geschaffenen", "Grundstücks", "wurde", "eine", "Grunddienstbarkeit", "an", "dem", "Grundstück", "FlSt", "<num>", "der", "Beteiligten", "mit", "folgendem", "Inhalt", "bestellt"],
            ["Der", "jeweilige", "Eigentümer", "des", "herrschenden", "Grundstücks", "ist", "berechtigt", "neben", "dem", "Eigentümer", "des", "dienenden", "Grundstücks", "auf", "dem", "dienenden", "Grundstück", "zu", "gehen", "und", "mit", "Fahrzeugen", "aller", "Art", "zu", "fahren"]
        ]
        assert list(process_segment(t, normalize=False)) == res
        t = [
            "II.",
            "Die Beschwerde hat keinen Erfolg."
        ]
        res = [
            ["die", "beschwerde", "hat", "keinen", "erfolg"] 
        ]
        assert list(process_segment(t)) == res

    def test_integration(self):
        t = load_verdict(Path("src")/"testing"/"test_data"/"short.json")
        res = {
            "guiding_principle": [
                    ["wird", "eine", "grunddienstbarkeit", "nach", "teilung", "des", "dienenden", "grundstücks", "an", "einem", "teil", "gelöscht"],
                    ["gegen", "die", "eintragung", "des", "widerspruchs", "kann", "mit", "dem", "ziel", "diesen", "zu", "löschen", "unbeschränkte", "beschwerde", "erhoben", "werden"]
                ],
            "facts": [
                    ["die", "beteiligte", "ist", "als", "eigentümerin", "vom", "grundbesitz", "im", "grundbuch", "eingetragen"],
                    ["ende", "<num>", "verkaufte", "sie", "hiervon", "teilflächen", "welche", "nach", "messungsanerkennung", "und", "auflassung", "vom", "<num>", "nun", "das", "grundstück", "flst", "<num>", "bilden"],
                    ["für", "die", "jeweiligen", "eigentümer", "des", "neu", "geschaffenen", "grundstücks", "wurde", "eine", "grunddienstbarkeit", "an", "dem", "grundstück", "flst", "<num>", "der", "beteiligten", "mit", "folgendem", "inhalt", "bestellt"],
                    ["der", "jeweilige", "eigentümer", "des", "herrschenden", "grundstücks", "ist", "berechtigt", "neben", "dem", "eigentümer", "des", "dienenden", "grundstücks", "auf", "dem", "dienenden", "grundstück", "zu", "gehen", "und", "mit", "fahrzeugen", "aller", "art", "zu", "fahren"]
                ],
            "reasoning": [["die", "beschwerde", "hat", "keinen", "erfolg"]]
        }
        assert t == res


# Functionality to resolve referenced norms to a continuous text
# As the norms are written in various different forms, we will use some invariants to resolve matching text segments for a given norm:
#  - if a norm is referenced, its title is also mentioned (by matching all words in ref against norm directory, we can get the norm)
#  - position in a norm is always referenced from generic to specific, i.e. the first number in a reference will almost certainly be the paragraph number
# -> with those two invariants, we should be able to match any common norm reference to a text

import pickle
import io
import os
import string
import urllib.request as http
import urllib.error as error
import json
from pathlib import Path
from typing import Set, Dict
import sqlite3

from tqdm import tqdm
from bs4 import BeautifulSoup
from spacy.lang.de import German

import sys
sys.path.append("data/sentences")
from sentence_segmenter import SentenceSegmenter

NORM_PATH = Path("..")/"HiWi"/"Normen"

# TODO We need to do this the other way around: aggregate all possible norms, then check if they are referenced in any verdict
def get_possible_norms():
    links = dict()
    path = 'https://www.gesetze-im-internet.de/Teilliste_'
    # There is no page for norms starting with "X", thus it is excluded
    suffixes = list(range(1, 10)) + list(map(lambda i: string.ascii_uppercase[i], range(23))) + list(map(lambda i: string.ascii_uppercase[i], range(24,26)))
    for i in tqdm(suffixes):
        file = path + str(i) + '.html'

        try:
            content = http.urlopen(file).read()
        except error.HTTPError:
            print("Cannot request:",file)
            continue

        parsed_html = BeautifulSoup(content, features="html.parser")
        for tag in parsed_html.find_all('abbr'):
            if tag.text!='PDF':
                abbr = tag.text.lstrip().rstrip()
                absolut_link = "https://www.gesetze-im-internet.de"+tag.parent["href"][1:]
                if abbr not in links:
                    links[abbr] = absolut_link
                else:
                    print("We already have:", abbr)

    return links

def filter_norms(norms: Dict[str, str]):
    used_norms = set()
    for file in tqdm(os.listdir(Path("data")/"dataset")):
        with io.open(Path("data")/"dataset"/file, "r", encoding="utf-8") as f:
            verdict = json.load(f)
            # There was some writing error; empty norms are list, but we expect a dictionary!
            if len(verdict["norms"]) == 0:
                continue
            for _, norm in verdict["norms"].items():
                norm_split = norm.split(" ")[0]
                if norm_split in norms:
                    used_norms.add(norm_split)


    return dict(filter(lambda elem: elem[0] in used_norms, norms.items()))

def aggregate_paragraphs(link: str):
    content = http.urlopen(link).read()
    parsed_html = BeautifulSoup(content, features="html.parser")

    base_url = "/".join(link.split("/")[:-1])

    paragraph_links = {}
    for tag in parsed_html.find_all('a'):
        if  "weggefallen" not in tag.text and "----" not in tag.text and "bis" not in tag.text and "(gegenstandslos)" not in tag.text:
            if tag.text.startswith("§") or tag.text.startswith("Art "):
                absolut_link = base_url + "/" + tag["href"]
                assert not tag.text.startswith("§§"), tag.text + " does not contain single paragraph. " + link
                paragraph_number = tag.text.split(" ")[1]
                assert paragraph_number.isnumeric() or (paragraph_number[:-1].isnumeric() and paragraph_number[-1].isalpha()), paragraph_number + " is no number. " + link
                paragraph_links[paragraph_number] = absolut_link

    # To check whether we have any edge cases
    assert len(paragraph_links) > 0, "Found no paragraphs for " + link

    paragraph_content = {}
    for paragraph, link in paragraph_links.items():
        content = get_paragraph_content(link)
        paragraph_content[paragraph] = content

    return paragraph_content

def get_paragraph_content(link: str):
    content = http.urlopen(link).read()
    parsed_html = BeautifulSoup(content, features="html.parser")

    segments = []
    for tag in parsed_html.find_all('div', class_="jurAbsatz"):
        segments.append(tag.get_text())

    return "\n".join(segments)

def create_norm_db(db_path: Path):
    """
    Create the norm db which contains all the norms and their paragraphs segmented into sentences
    """
    segmenter = SentenceSegmenter(path=["data", "sentences"])
    nlp = German()
    nlp.add_pipe(segmenter)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Be aware that we cannot use integer for paragraph, as it may be a combined emueration symbol e.g. 1a
        cursor.execute("create table norms (name text not null, paragraph text not null, sentence integer not null, content text not null, primary key (name,paragraph,sentence));")
        conn.commit()
    except sqlite3.OperationalError:
        print("Resuming from previously build table")

    for file in tqdm(os.listdir(NORM_PATH)):
        name = file[:-len(".json")]
        cursor.execute("select * from norms where name=:n", {"n": name})
        # Check whether, we have already processed this norm
        if cursor.fetchone() is not None:
            continue
        if name in ["BGBEG", "EGBGB"]:
            continue

        with io.open(NORM_PATH/file, "r", encoding="utf-8") as f:
            norm = json.load(f)
        for paragraph_num, paragraph_text in norm.items():
            doc = nlp(paragraph_text)
            for i, sent in enumerate(doc.sents):
                cursor.execute("insert into norms values (?, ?, ?, ?);", (name, paragraph_num, i, sent.text))
        conn.commit()

if __name__ == "__main__":
    #norms = get_possible_norms()
    #print(len(norms))
    #with io.open(Path("data")/"norms"/"norm2links.pkl", "wb") as f:
    #    pickle.dump(norms, f)
    #with io.open(Path("data")/"norms"/"norm2links.pkl", "rb") as f:
    #    norms = pickle.load(f)
    #norms = filter_norms(norms)
    #with io.open(Path("data")/"norms"/"used_norm2links.pkl", "wb") as f:
    #    pickle.dump(norms, f)
    """with io.open(Path("data")/"norms"/"used_norm2links.pkl", "rb") as f:
        links = pickle.load(f)
    # The following norms are empty:
    del links["BürgPoRPakt"]
    del links["ComKrimÜbkG"]
    del links["DiplBezÜbk"]
    del links["EuPatAuslProt"]
    del links["EuRHiÜbk"]
    del links["EuVtrÜbk"]
    del links["FlüAbk"]
    del links["IntWarenBezÜbk"]
    del links["IntZLuftAbk"]
    del links["MSA"]
    del links["NATOTrStat"]
    del links["NATOTrStatZAbk"]
    del links["SaarVtr"]
    del links["TV-H"]
    del links["TVöD"]
    del links["UhAnsprAuslÜbk"]
    del links["UNCh"]
    del links["UNWaVtrÜbk"]
    del links["VollstrZustÜbk"]
    del links["VtrRKonv"]
    del links["VtrRKonvG"]
    del links["MontrÜbk"]
    # TODO the following norms might need some extra processing
    del links["IntPatÜbkG"]
    del links["MoselSchPV"]
    del links["RV"]

    # TODO BerlinFG, EuAuslfÜbk, KVR, StVÜbk

    norm_path = Path("..")/"HiWi"/"Normen"

    for tag, link in tqdm(links.items()):
        tag = tag.replace("/", "_")
        if not os.path.exists(norm_path/(tag+".json")):
            try:
                content = aggregate_paragraphs(link)
                with io.open(norm_path/(tag+".json"), "w", encoding="utf-8") as f:
                    json.dump(content, f, sort_keys=False, indent=4, ensure_ascii=False)
            except AssertionError as e:
                print(e)
                print(tag)"""

    create_norm_db(Path("data")/"databases"/"norms.db")
    
# Script is used to go through all documents and replace all norms with their identifier __normxyz__
import re
from typing import Iterable, List
from pathlib import Path
from configparser import ConfigParser
import requests

USER_INFO = dict()

def split(sentences: List[str]) -> Iterable[List[str]]:
    return map(lambda sentence: list(filter(lambda tok: tok is not None and len(tok) > 0, re.split(r"[\s]+", sentence))), sentences)

def read_user_info(path: Path=Path("data")/"norms"/"config.ini"):
    global USER_INFO
    config = ConfigParser()
    # Obviously this only works with the correct user information
    config.read(path)
    USER_INFO = config["USER"]

def annotate_publisher(sentences: List[str]):
    if len(USER_INFO) == 0:
        raise ValueError("Did not import the configuartion.")
    
    annotated_sentences = []
    for sentence in sentences:
        r = requests.post(USER_INFO["url"], data={"text": sentence}, auth=(USER_INFO["username"], USER_INFO["password"]))
        annotated_sentences.append(r.content.decode("utf-8"))
    return annotated_sentences

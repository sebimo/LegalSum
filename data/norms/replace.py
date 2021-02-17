# Script is used to go through all documents and replace all norms with their identifier __normxyz__
import re
import io
import json
from typing import Iterable, List, Dict, Tuple
from pathlib import Path
from configparser import ConfigParser
import requests
from functools import reduce
from enum import Enum

from .normdb import NormDatabase

USER_INFO = dict()

# Used for the custom matching method; all tokens in IGNORE_LIST will be jumped over and no norm cut will be made
IGNORE_LIST = ["Abs", "Satz"]

class NormState(Enum):
    DOUBLE = 0
    SINGLE = 1

def split(sentences: List[str]) -> Iterable[List[str]]:
    return map(lambda sentence: list(filter(lambda tok: tok is not None and len(tok) > 0, re.split(r"[\s]+", sentence))), sentences)

def process_verdict(filepath: Path):
    """ Will load the verdict and look through its 
         - normchain (Check that this is created as a dictionary)
         - guiding principles
         - facts
         - reasoning 
        and replace norms found in them. guiding principles, facts, reasoning will be handled by the same code.
        normchain will have its own code
        
        Will rewrite the verdict to the path, if any changes have happened.
    """
    verdict = __load_verdict__(filepath)
    norms = dict()
    guid_princ_0, n = process_segment(verdict["guiding_principle"][0])
    if len(n) > 0:
        verdict["guiding_principle"][0] = guid_princ_0
        norms.update(n)

    guid_princ_1, n = process_segment(verdict["guiding_principle"][1])
    if len(n) > 0:
        verdict["guiding_principle"][1] = guid_princ_1
        norms.update(n)

    facts, n = process_segment(verdict["facts"])
    if len(n) > 0:
        verdict["facts"] = facts
        norms.update(n)

    reasoning, n = process_segment(verdict["reasoning"])
    if len(n) > 0:
        verdict["reasoning"] = reasoning
        norms.update(n)

    chain, n = process_normchain(verdict["normchain"])
    if len(n) > 0:
        verdict["normchain"] = chain
        norms.update(n)

    if len(verdict["norms"]) == 0:
        verdict["norms"] = norms
    else:
        verdict["norms"].update(norms)

    __write_verdict__(verdict, filepath)

    
def process_segment(segment: List[str], normDB: NormDatabase) -> Tuple[List[str], Dict[str, str]]:
    """ Go through all the sentences in the segment and check, if they do contain any norms which were not found so far 
        Replace them + collect into a dictionary
    """
    raise NotImplementedError

def process_sentence(sentence: str, normDB: NormDatabase) -> Tuple[List[str], Dict[str, str]]:
    pass

def process_normchain(chain: List[str], normDB: NormDatabase) -> Tuple[List[str], Dict[str, str]]:
    """ Will go through the normchain and replace all occurences of norms, i.e. everything not __normxyz__
        Returns True, if any new norms are found + the placeholders for the norms + their placeholder <-> norm dict
    """
    # Check, if we even need to process the normchain
    if len(chain) == 0 or chain[0].startswith("__norm"):
        return (chain, {})

    # Split up based on ";" -> those are for certain independent norms
    norms = map(lambda entry: list(map(lambda norm: norm.strip(), entry.split(";"))), chain)
    norms = reduce(lambda x,y: x+y, norms, [])

    processed_norms = []
    for norm in norms:
        # Does the current norm have multiple parts? Each part will be seperated by a ","
        if "," not in norm:
            processed_norms.append(norm)
        else:
            # Split on "," -> a norm might be linked to the previous norm, i.e. we need to keep info from the previous split element
            # Cases:
            #  * "§§ x, y, z NORM" -> "§ x NORM", "§ y NORM", "§ z NORM" 
            #    -> this case ends, if we are at the end of the text or a comma part with §
            #    = NormState.DOUBLE
            #  * "NORM § x, § y, § z" -> "§ x NORM", "§ y NORM", "§ z NORM" 
            #    -> this case ends, if we are at the end of the text or a comma part with §§ or a comma part which is no starting with §
            #    = NormState.SINGLE
            raise NotImplementedError

    norm_dict = {}
    for norm in norms:
        placeholder = normDB.register_norm(norm)
        norm_dict[placeholder] = norm
    return norms, norm_dict


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

def __load_verdict__(path: Path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def __write_verdict__(dic: Dict, save_path: Path):
    with io.open(save_path, "w", encoding="utf-8") as f:
        json.dump(dic, f, sort_keys=False, indent=4, ensure_ascii=False)

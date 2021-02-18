# Script is used to go through all documents and replace all norms with their identifier __normxyz__
import re
import io
import os
import json
from typing import Iterable, List, Dict, Tuple
from pathlib import Path
from configparser import ConfigParser
import requests
from functools import reduce
from enum import Enum

from .normdb import NormDatabase

USER_INFO = dict()

NORM_PATH = Path("..")/"HiWi"/"Normen"

# Used for the custom matching method; all tokens in IGNORE_LIST will be jumped over and no norm cut will be made
IGNORE_LIST = ["Abs", "Satz"]
NORM_SET = set()

# States used for parsing the normchain
class NormState(Enum):
    DOUBLE = 0
    SINGLE = 1
    UNKNOWN = -1

# States used for searching through the individual sentences
class SearchMode(Enum):
    UNKNOWN = -1
    NORM = 1

def split(sentences: List[str]) -> Iterable[List[str]]:
    return map(lambda sentence: list(filter(lambda tok: tok is not None and len(tok) > 0, re.split(r"[\s]+", sentence))), sentences)

def process_verdict(filepath: Path, normDB: NormDatabase):
    """ Will load the verdict and look through its 
         - normchain (Check that this is created as a dictionary)
         - guiding principles
         - facts
         - reasoning 
        and replace norms found in them. guiding principles, facts, reasoning will be handled by the same code.
        normchain will have its own code
        
        Will rewrite the verdict to the path, if any changes have happened.
    """
    # We could move this one layer up, but we do not want to spill the implementation details there
    setup_norm_set()
    verdict = __load_verdict__(filepath)
    norms = dict()
    guid_princ_0, n = process_segment(verdict["guiding_principle"][0], normDB)
    if len(n) > 0:
        verdict["guiding_principle"][0] = guid_princ_0
        norms.update(n)

    guid_princ_1, n = process_segment(verdict["guiding_principle"][1], normDB)
    if len(n) > 0:
        verdict["guiding_principle"][1] = guid_princ_1
        norms.update(n)

    facts, n = process_segment(verdict["facts"], normDB)
    if len(n) > 0:
        verdict["facts"] = facts
        norms.update(n)

    reasoning, n = process_segment(verdict["reasoning"], normDB)
    if len(n) > 0:
        verdict["reasoning"] = reasoning
        norms.update(n)

    chain, n = process_normchain(verdict["normchain"], normDB)
    if len(n) > 0:
        verdict["normchain"] = chain
        norms.update(n)

    if len(verdict["norms"]) == 0:
        verdict["norms"] = norms
    else:
        verdict["norms"].update(norms)

    __write_verdict__(verdict, filepath)


def setup_norm_set():
    """ Gets all the known abbreviation for known norms from the file names of the norm dataset """
    global NORM_SET
    if len(NORM_SET) == 0:
        for file in os.listdir(NORM_PATH):
            NORM_SET.add(file[:len(".json")])
    
def process_segment(segment: List[str], normDB: NormDatabase) -> Tuple[List[str], Dict[str, str]]:
    """ Go through all the sentences in the segment and check, if they do contain any norms which were not found so far 
        Replace them + collect into a dictionary
    """
    processed_segments = []
    norms = {}
    for sentence in segment:
        processed_sentence, extracted_norms = process_sentence(sentence, normDB)
        processed_segments.append(processed_sentence)
        norms.update(extracted_norms)

    return processed_segments, norms   

def process_sentence(sentence: str, normDB: NormDatabase) -> Tuple[List[str], Dict[str, str]]:
    """ This is not a perfect algorithm for finding norms in the text, but it is a non-destructive algorithm.
        That means any norm which is removed from the text, can be later swapped back in and the text stays the same.
        This algorithm will only find a norm, if it is in the norm dataset & its abbreviation is exactly matched (this might not hold for all norms!)
        This limitation is important, as those are the only norms we can match towards a specific text anyways.
    """
    split = sentence.split(" ")
    pieces = []
    norms = {}
    current_norm = None
    state = SearchMode.UNKNOWN
    PATIENCE_START = 2
    patience = PATIENCE_START
    for token in split:
        if state == SearchMode.UNKNOWN:
            if token in NORM_SET or "§" in token:
                current_norm = [token]
                state = SearchMode.NORM
                patience = PATIENCE_START
            else:
                pieces.append(token)
        else:
            # We need to be here a bit more precise, especially with the patience, i.e. when do we stop if we reduced the patience?
            if token.isnumeric() or token in IGNORE_LIST or token in NORM_SET :
                current_norm.append(token)
            elif patience > 0:
                # We might want to introduce some patience value, s.t. we are more robust against wrongly written tokens
                # I.e. we can look two tokens into the future
                patience -= 1
                current_norm.append(token)
            else:
                norm = " ".join(current_norm)
                placeholder = normDB.register_norm(norm)
                norms[placeholder] = norm
                current_norm = None

                pieces.append(placeholder)
                state = SearchMode.UNKNOWN

    finalized_sentence = " ".join(pieces)
    return finalized_sentence, norms

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
            norm_split = norm.split(",")
            current_paragraphs = []
            current_norm = None
            state = NormState.UNKNOWN
            for split in norm_split:
                split = split.strip()
                if state == NormState.UNKNOWN:
                    assert len(current_paragraphs) == 0
                    assert current_norm is None

                    if "§" not in split:
                        processed_norms.append(split)
                    if split.startswith("§§"):
                        current_paragraphs = [split[2:].strip()]
                        state = NormState.DOUBLE
                    elif split.split(" ")[0].isalnum():
                        # Now we have identified that we are in a norm with "§" + it starts with a word 
                        # We now will identify the norm (= everything before the "§") + its paragraphs
                        paragraph_split = split.split("§")
                        assert len(paragraph_split) == 2, "Norm does not follow the standard format: " +str(paragraph_split)
                        current_norm = paragraph_split[0].strip()
                        current_paragraphs = [paragraph_split[1].strip()]
                        state = NormState.SINGLE
                    else:
                        raise ValueError("Starting norm does not follow known format: "+str(split))

                elif state == NormState.DOUBLE:
                    # We will end this state, if we find a split which does contain a "§"
                    assert len(current_paragraphs) > 0
                    assert current_norm is None
                    if "§" in split:
                        # Heuristic: The last word in the last current_paragraph is the norm
                        current_norm = current_paragraphs[-1].split(" ")[-1]
                        current_paragraphs[-1] = " ".join(current_paragraphs[-1].split(" ")[:-1])
                        processed_norms += finalize_norm(current_norm, current_paragraphs)

                        # We now differentiate between the different cases as above -> because we are starting a new norm
                        if split.startswith("§§"):
                            current_paragraphs = [split[2:].strip()]
                            current_norm = None
                            state = NormState.DOUBLE
                        elif split.split(" ")[0].isalnum():
                            paragraph_split = split.split("§")
                            assert len(paragraph_split) == 2, "Norm does not follow the standard format: " +str(paragraph_split)
                            current_norm = paragraph_split[0].strip()
                            current_paragraphs = [paragraph_split[1].strip()]
                            state = NormState.SINGLE
                        else:
                            raise ValueError("Norm does not follow known format: "+str(split))
                    else:
                        current_paragraphs.append(split)
                elif state == NormState.SINGLE:
                    assert len(current_paragraphs) > 0
                    assert current_norm is not None
                    if "§§" in split:
                        processed_norms += finalize_norm(current_norm, current_paragraphs)
                        if split.startswith("§§"):
                            current_paragraphs = [split[2:].strip()]
                            current_norm = None
                            state = NormState.DOUBLE
                        else:
                            raise ValueError("Norm does not follow known format: "+str(split))
                    elif "§" in split:
                        # Check if we are beginning a new norm
                        # ATTENTION: we can only do this differentiation, because we assume that in NormState.SINGLE every individual norm has its own "§"
                        if split.split(" ")[0].isalnum():
                            processed_norms += finalize_norm(current_norm, current_paragraphs)
                            
                            paragraph_split = split.split("§")
                            assert len(paragraph_split) == 2, "Norm does not follow the standard format: " +str(paragraph_split)
                            current_norm = paragraph_split[0].strip()
                            current_paragraphs = [paragraph_split[1].strip()]
                            state = NormState.SINGLE
                        else:
                            # We want to remove everything before the first "§"
                            split = list(map(lambda entry: entry.strip(), split.split("§")[1:]))
                            current_paragraphs += split
                    else:
                        current_paragraphs.append(split)

            # Include all remaining results from current_paragraphs + current_norm
            if len(current_paragraphs) > 0:
                if state == NormState.DOUBLE:
                    current_norm = current_paragraphs[-1].split(" ")[-1]
                    current_paragraphs[-1] = " ".join(current_paragraphs[-1].split(" ")[:-1])
                processed_norms += finalize_norm(current_norm, current_paragraphs)

    norm_dict = {}
    for norm in processed_norms:
        placeholder = normDB.register_norm(norm)
        norm_dict[placeholder] = norm
    return processed_norms, norm_dict

def finalize_norm(norm: str, paragraphs: List[str]) -> List[str]:
    """ Will replicate norm to the paragraphs s.t. each paragraph has the norm information """
    result = []
    for p in paragraphs:
        result.append(norm+" § "+p)
    return result

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

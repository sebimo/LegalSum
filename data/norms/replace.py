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
import string
from collections import Counter

from tqdm import tqdm

import sys
sys.path.append("data/norms")
from normdb import NormDatabase, NormDBStub

USER_INFO = dict()

NORM_PATH = Path("..")/"HiWi"/"Normen"
DATA_PATH = Path("data")/"dataset"

# Used for the custom matching method; all tokens in SEPARATOR_LIST will be jumped over and no norm cut will be made
SEPARATOR_LIST = ()
LETTERS = string.ascii_letters+"äÄüÜöÖß"
LETTERS_PLUS = LETTERS+"-/"
NUMBERS = string.digits
NUMBERS_PLUS = NUMBERS+"."
ABBREV_LETTER = LETTERS+"."
CHARACTERS = LETTERS + NUMBERS
CHARS_PLUS = CHARACTERS + "§"
# Counter used to identify common tokens around §-sign
NEW_IGNORES = Counter()
NORM_SET = set()

# Flag used to enable writing the verdicts back to disk
ENABLE_REWRITING = False

# States used for parsing the normchain
class NormState(Enum):
    DOUBLE = 0
    SINGLE = 1
    ART = 2
    UNKNOWN = -1

# States used for searching through the individual sentences
class SearchMode(Enum):
    UNKNOWN = -1
    NORM = 1

def annotate_verdicts():
    """ Will go through all the verdicts and tries to reannotate them. Based on ENABLE_REWRITING they are commited back to disk. """
    normDB = NormDatabase(Path("data")/"norms"/"norms.db") if ENABLE_REWRITING else NormDBStub()
    for verdict in tqdm(os.listdir(DATA_PATH), desc="Processing"):
        print(verdict)
        process_verdict(DATA_PATH/verdict, normDB)
        break

    # Code was used to identify which tokens are commonly used around §-signs
    """with io.open(Path("data")/"norms"/"norms_ignored_words.txt", "a+", encoding="utf-8") as f:
        for token in NEW_IGNORES.most_common():
            if all(c in ABBREV_LETTER for c in token[0]):
                f.write(str(token)+"\n")"""

def process_verdict(filepath: Path, normDB: NormDatabase):
    """ Will load the verdict and look through its 
         - normchain (Check that this is created as a dictionary)
         - guiding principles
         - facts
         - reasoning 
        and replace norms found in them. guiding principles, facts, reasoning will be handled by the same code.
        normchain will have its own code
        
        Will rewrite the verdict to the path, if any changes have happened (and ENABLE_REWRITING is set)
    """
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

    try:
        chain, n = process_normchain(verdict["normchain"], normDB)
    except AssertionError:
        print(filepath)
        raise AssertionError
    if len(n) > 0:
        verdict["normchain"] = chain
        norms.update(n)

    if len(verdict["norms"]) == 0:
        verdict["norms"] = norms
    else:
        verdict["norms"].update(norms)

    if ENABLE_REWRITING:
        raise ValueError("Should not write right now.")
        __write_verdict__(verdict, filepath)
    
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
    if len(SEPARATOR_LIST) == 0:
        read_separator_list()
    assert len(SEPARATOR_LIST) > 0
    if len(NORM_SET) == 0:
        setup_norm_set()
    assert len(NORM_SET) > 0

    split = sentence.split(" ")
    pieces = []
    norms = {}
    current_norm = []
    # We might need to keep track of the special characters before and after a norm
    special_before = ""
    special_after = ""
    state = SearchMode.UNKNOWN
    for token in split:
        cleaned_token = "".join(filter(str.isalpha, token.lower()))
        if state == SearchMode.UNKNOWN:
            if cleaned_token in NORM_SET or "§" in token:
                # We need to separate any special characters at the beginning from the token
                special_before, res_token = get_special_before(token)
                current_norm = [res_token]
                state = SearchMode.NORM
            else:
                pieces.append(token)
        else:
            assert len(current_norm) >= 1
            # We need to be here a bit more precise, especially with the patience, i.e. when do we stop if we reduced the patience?
            if all(c in NUMBERS_PLUS for c in token) or cleaned_token in NORM_SET or is_paragraph_token(token):
                current_norm.append(token)
            else:
                if len(current_norm) > 1 or "§" not in current_norm[0]: 
                    special_after, current_norm[-1] = get_special_after(current_norm[-1])
                    norm = " ".join(current_norm)
                    placeholder = normDB.register_norm(norm)
                    norms[placeholder] = norm
                    pieces.append(special_before + placeholder + special_after)

                    current_norm = []
                    special_before = ""
                    special_after = ""
                else:
                    # Check for special_before
                    current_norm[0] = special_before + current_norm[0]
                    for p in current_norm:
                        pieces.append(p)
                    current_norm = []

                if cleaned_token in NORM_SET or "§" in token:
                    # Check for special_before
                    special_before, res_token = get_special_before(token)
                    current_norm = [res_token]
                    state = SearchMode.NORM
                else:
                    pieces.append(token)
                    state = SearchMode.UNKNOWN

    if len(current_norm) > 0:
        special_after, current_norm[-1] = get_special_after(current_norm[-1])
        norm = " ".join(current_norm)
        placeholder = normDB.register_norm(norm)
        norms[placeholder] = norm
        current_norm = None
        pieces.append(special_before + placeholder + special_after)

    finalized_sentence = " ".join(pieces)
    return finalized_sentence, norms

def process_normchain(chain: List[str], normDB: NormDatabase) -> Tuple[List[str], Dict[str, str]]:
    """ Will go through the normchain and replace all occurences of norms, i.e. everything not __normxyz__
        Returns True, if any new norms are found + the placeholders for the norms + their placeholder <-> norm dict
    """
    # Check, if we even need to process the normchain
    if len(chain) == 0 or chain[0].startswith("__norm"):
        return (chain, {})
    
    if len(chain) == 1:
        if chain[0].startswith("=="):
            chain[0] = chain[0][len("=="):].strip()
        if chain[0].startswith("Normen:"):
            chain[0] = chain[0][len("Normen:"):].strip()
        elif chain[0].startswith("Gesetz:"):
            chain[0] = chain[0][len("Gesetz:"):].strip()
        elif chain[0].startswith("Gesetze:"):
            chain[0] = chain[0][len("Gesetze:"):].strip()
        elif chain[0].startswith("Vorschriften:"):
            chain[0] = chain[0][len("Vorschriften:"):].strip()

    # Split up based on ";" -> those are for certain independent norms
    norms = map(lambda entry: list(map(lambda norm: norm.strip(), entry.split(";"))), chain)
    norms = reduce(lambda x,y: x+y, norms, [])
    split_norms = []
    for n in norms:
        for s in n.split("i.V.m."):
            split_norms.append(s.replace("§ §", "§§"))
    norms = split_norms

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
                assert state == NormState.UNKNOWN or (current_norm is not None or len(current_paragraphs) > 0)
                split = split.strip()
                # So this is the most problematic case, as it literally can have any writing style + 100% correctly parsing is not really feasible here due to the time constraints
                # We will simplify this case a bit and include all tokens between article and the next norm reference
                if state == NormState.ART or split.startswith("Art."):
                    # In the first runthrough we need to clean the norm buffer
                    if state != NormState.ART and len(current_paragraphs) > 0:
                        current_norm, current_paragraphs = finalize_current_norm(current_norm, current_paragraphs)
                        processed_norms += finalize_norm(current_norm, current_paragraphs)
                        current_norm = None
                        current_paragraphs = []
                    assert current_norm is None
                    current_paragraphs, current_norm, state = process_state_art(split, norm, processed_norms, current_paragraphs, current_norm)
                elif state == NormState.UNKNOWN:
                    assert len(current_paragraphs) == 0
                    assert current_norm is None
                    current_paragraphs, current_norm, state = process_state_unknown(split, norm, processed_norms)
                elif state == NormState.DOUBLE:
                    # We will end this state, if we find a split which does contain a "§"
                    assert len(current_paragraphs) > 0
                    current_paragraphs, current_norm, state = process_state_double(split, norm, processed_norms, current_paragraphs, current_norm)
                elif state == NormState.SINGLE:
                    assert len(current_paragraphs) > 0
                    current_paragraphs, current_norm, state = process_state_single(split, norm, processed_norms, current_paragraphs, current_norm)

            # Include all remaining results from current_paragraphs + current_norm
            if len(current_paragraphs) > 0:
                if state == NormState.ART:
                    processed_norms.append(", ".join(current_paragraphs))
                else:
                    current_norm, current_paragraphs = finalize_current_norm(current_norm, current_paragraphs)
                    processed_norms += finalize_norm(current_norm, current_paragraphs)

    norm_dict = {}
    for norm in processed_norms:
        placeholder = normDB.register_norm(norm)
        norm_dict[placeholder] = norm
    return processed_norms, norm_dict

def process_state_unknown(split: str, norm: str, processed_norms: List[str]) -> Tuple[List[str], str, NormState]:
    """ State transition, if we are beginning with processing the normchain """
    if "§" not in split:
        processed_norms.append(split)
        return [], None, NormState.UNKNOWN
    else:
        current_paragraphs, current_norm, state = new_norm_state_transition(split, norm, processed_norms)
        return current_paragraphs, current_norm, state

def process_state_double(split: str, norm: str, processed_norms: List[str], current_paragraphs: List[str], current_norm: str) -> Tuple[List[str], str, NormState]:
    """ State transition, if we are processing a norm with multiple paragraphs i.e. with §§ """
    if "§" in split:
        # Heuristic: The last word in the last current_paragraph is the norm
        current_norm, current_paragraphs = finalize_current_norm(current_norm, current_paragraphs)
        processed_norms += finalize_norm(current_norm, current_paragraphs)

        # We now differentiate between the different cases as above -> because we are starting a new norm
        current_paragraphs, current_norm, state = new_norm_state_transition(split, norm, processed_norms)
        return current_paragraphs, current_norm, state
    else:
        current_paragraphs.append(split)
        return current_paragraphs, current_norm, NormState.DOUBLE

def process_state_single(split: str, norm: str, processed_norms: List[str], current_paragraphs: List[str], current_norm: str) -> Tuple[List[str], str, NormState]:
    if "§§" in split:
        current_norm, current_paragraphs = finalize_current_norm(current_norm, current_paragraphs)
        processed_norms += finalize_norm(current_norm, current_paragraphs)
        if split.startswith("§§"):
            current_paragraphs = [split[2:].strip()]
            current_norm = None
        else:
            subsplit = split.split("§§")
            assert len(subsplit) == 2
            current_norm = subsplit[0].strip()
            current_paragraphs = [subsplit[1].strip()]
        state = NormState.DOUBLE
    elif "§" in split:
        # Check if we are beginning a new norm
        # ATTENTION: we can only do this differentiation, because we assume that in NormState.SINGLE every individual norm has its own "§"
        if split.split(" ")[0].isalnum():
            current_norm, current_paragraphs = finalize_current_norm(current_norm, current_paragraphs)
            processed_norms += finalize_norm(current_norm, current_paragraphs)
            
            paragraph_split = split.split("§")
            assert len(paragraph_split) == 2, "Norm does not follow the standard format: " +str(paragraph_split)
            current_norm = paragraph_split[0].strip()
            current_paragraphs = [paragraph_split[1].strip()]
            state = NormState.SINGLE
        elif current_norm is None and not split.split(" ")[-1].isnumeric():
            current_paragraphs.append(split.replace("§", "").strip())
            current_norm, current_paragraphs = finalize_current_norm(current_norm, current_paragraphs)
            processed_norms += finalize_norm(current_norm, current_paragraphs)

            current_norm = None
            current_paragraphs = []
            state = NormState.UNKNOWN
        else:
            # We want to remove everything before the first "§"
            split = list(map(lambda entry: entry.strip(), split.split("§")[1:]))
            current_paragraphs += split
            state = NormState.SINGLE
    else:
        current_paragraphs.append(split)
        state = NormState.SINGLE
    
    return current_paragraphs, current_norm, state

def process_state_art(split: str, norm: str, processed_norms: List[str], current_paragraphs: List[str], current_norm: str) -> Tuple[List[str], str, NormState]:
    """ State transition, if we are currently processing an "Art. xxx NORM" -> try to find a norm (any alpha token) """
    subsplit = split.split(" ")
    if subsplit[-1].isnumeric():
        current_paragraphs.append(split)
        state = NormState.ART
    # There might be the more precise elif, but we will just assume that everything that is not a number is a norm
    else:
        # We have found a norm -> this only works as all norms are written as "DESCRIPTOR NUM (SUBDESCRIPTOR NUM)* NORM"
        current_paragraphs.append(split)
        processed_norms.append(", ".join(current_paragraphs))
        current_paragraphs = []
        current_norm = None
        state = NormState.UNKNOWN
    return current_paragraphs, current_norm, state

def new_norm_state_transition(split: str, norm: str, processed_norms: List[str]) -> Tuple[List[str], str, NormState]:
    # We now differentiate between the different cases as above -> because we are starting a new norm
    if split.startswith("§§"):
        current_paragraphs = [split[2:].strip()]
        current_norm = None
        state = NormState.DOUBLE
    elif "§§" in split:
        subsplit = split.split("§§")
        assert len(subsplit) == 2, "Norm does not follow the standard format: " +str(split)+ " from "+norm
        current_paragraphs = [subsplit[1].strip()]
        current_norm = subsplit[0].strip()
        state = NormState.DOUBLE
    elif split.startswith("§"):
        paragraph_split = split.split("§")
        assert len(paragraph_split) == 2, "Norm does not follow the standard format: " +str(split)+ " from "+norm
        # Check if the current norm is finished or not
        if not split.split(" ")[-1].isnumeric():
            split = split.replace("§ ", "")
            subsplit = list(filter(lambda s: len(s) > 0, split.split(" ")))
            processed_norms += finalize_norm(subsplit[-1], [" ".join(subsplit[:-1])])
            
            current_norm = None
            current_paragraphs = []
            state = NormState.UNKNOWN
        else:
            current_norm = None
            current_paragraphs = [paragraph_split[1].strip()]
            state = NormState.SINGLE
    # It depends on the implementation of isalnum(), if we want to keep it or just use the second part 
    # (i.e. it might be faster this way, instead of streaming over all characters); For now it stays here
    elif split.split(" ")[0].isalnum() or all(c in LETTERS_PLUS for c in split.split(" ")[0]):
        # Now we have identified that we are in a norm with "§" + it starts with a word 
        # We now will identify the norm (= everything before the "§") + its paragraphs
        paragraph_split = split.split("§")
        assert len(paragraph_split) == 2, "Norm does not follow the standard format: " +str(split)+ " from "+norm
        current_norm = paragraph_split[0].strip()
        current_paragraphs = [paragraph_split[1].strip()]
        state = NormState.SINGLE
    else:
        raise AssertionError("Norm does not follow known format: "+str(split)+ " from "+norm)

    return current_paragraphs, current_norm, state

def finalize_current_norm(current_norm: str, current_paragraphs: List[str]) -> Tuple[str, List[str]]:
    """ Called before appending norm to check if we have found a current_norm so far """
    if current_norm is None:
        current_norm = current_paragraphs[-1].split(" ")[-1].strip()
        current_paragraphs[-1] = " ".join(current_paragraphs[-1].split(" ")[:-1])

    return current_norm, current_paragraphs

def is_paragraph_token(token: str):
    """ Checks if the token is starting with any of the words defined in SEPARATOR_LIST. 
        As there are many possible writing styles for referencing a specific passage in a norm,
        we need to write a robust function for filtering them.
    """
    paragraph_token = token.lower().startswith(SEPARATOR_LIST) or len(token) == 1 or token[0].isnumeric()
    return paragraph_token
    # Code used to collect all the tokens which are commonly found around §-signs
    """if token.lower().startswith(SEPARATOR_LIST):
        return True
    else:
        NEW_IGNORES.update([token])
        return False"""

def finalize_norm(norm: str, paragraphs: List[str]) -> List[str]:
    """ Will replicate norm to the paragraphs s.t. each paragraph has the norm information """
    result = []
    for p in paragraphs:
        result.append(norm+" § "+p)
    return result

def read_separator_list():
    global SEPARATOR_LIST
    seps = []
    with io.open(Path("data")/"norms"/"norms_separator.txt", "r", encoding="utf-8") as f:
        for l in f.readlines():
            seps.append(l.lower().strip())
    SEPARATOR_LIST = tuple(seps)

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

def split(sentences: List[str]) -> Iterable[List[str]]:
    return map(lambda sentence: list(filter(lambda tok: tok is not None and len(tok) > 0, re.split(r"[\s]+", sentence))), sentences)

def get_special_before(token: str) -> Tuple[str, str]:
    """ Will split the token into special character at the beginning + remaining token, e.g. "(BGB" -> ("(", "BGB") """
    res_before = []
    res_tok = ""
    for i, c in enumerate(token):
        if c not in CHARS_PLUS:
            res_before.append(c)
        else:
            res_tok = token[i:]
            break
    return "".join(res_before), res_tok

def get_special_after(token: str) -> Tuple[str, str]:
    """ Will split the token into special character at the end + remaining token, e.g. "BGB)" -> (")", "BGB") """
    # Obviously it is a bit of a hacky solution to reverse the string and then just use get_special_before
    # But as the only time, when this will be called is at the end of a norm + norm references are really short tokens, this is sufficient
    rev_token = token[::-1]
    rev_before, rev_tok = get_special_before(rev_token)
    return rev_before[::-1], rev_tok[::-1]

def setup_norm_set():
    """ Gets all the known abbreviation for known norms from the file names of the norm dataset """
    global NORM_SET
    if len(NORM_SET) == 0:
        for file in os.listdir(NORM_PATH):
            assert file.endswith(".json")
            NORM_SET.add(file.replace(".json", "").lower())

def __load_verdict__(path: Path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def __write_verdict__(dic: Dict, save_path: Path):
    raise ValueError("Should not write right now.")
    with io.open(save_path, "w", encoding="utf-8") as f:
        json.dump(dic, f, sort_keys=False, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    annotate_verdicts()

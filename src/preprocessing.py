# Contains all the preprocessing code + data loading
# We want to test a more bare-metal version of this as well
import io
import os
import json
from pathlib import Path
from collections import defaultdict, Counter
import pickle
import re
from enum import Enum
from typing import List, Dict, Iterable

from tqdm import tqdm

# -----------------------------------------------
# Some static variables used throughout this file
class TokenizationType(Enum):
    SPACE = 1
    BYTE_PAIR = 2

CHAR_SET = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'Ä', 'Ü', 'Ö', 'ä', 'ü', 'ö', 'ß', '<', '>'}

CHAR_MAP = defaultdict(lambda: "")

for c in CHAR_SET:
    CHAR_MAP[ord(c)] = c

DATA_PATH = Path("data")/"dataset"
# -----------------------------------------------


class Tokenizer:

    def __init__(self, path: Path, normalize: bool=True):
        """ Initializes an empty tokenizer and will load a trained tokenizer, if the path exists """
        self.path = path
        self.normalize = normalize
        self.tok2id = None
        self.id2tok = None
        if (self.path/"tokenizer.pkl").exists():
            with io.open(self.path/"tokenizer.pkl", "rb") as f:
                state = pickle.load(f)
                self.tok2id = state["tok2id"]
                self.id2tok = state["id2tok"]

    def create_token_id_mapping(self, threshold: int=10, max_num_tokens: int=50000):
        """ Creates the mapping between tokens proceduced by preprocessing to token ids used in training 
            Params:
                - threshold -- define how often a token needs to apprear to keep it in the corpus
        """
        frequency = Counter()
        for file in tqdm(os.listdir(DATA_PATH)):
            verdict = load_verdict(DATA_PATH/file, normalize=self.normalize)
            for gp in verdict["guiding_principle"]:
                frequency.update(gp)
            for f in verdict["facts"]:
                frequency.update(f)
            for r in verdict["reasoning"]:
                frequency.update(r)
        
        # We will use a defaultdict, s.t. we can just map any unseen token to id 0, which translates back to unknown
        self.tok2id = defaultdict(int)
        assert self.tok2id["<unk>"] == 0
        self.tok2id["<unk>"] = 0
        self.id2tok = {0: "<unk>"}
        # We can differntiate here between choosing tokens which have a certain number of appearances;
        # More applicable here would be to choose a max_num_tokens
        # TODO do some token analysis, i.e. how often they occur and their count distribution etc.
        if max_num_tokens > 0:
            # -1 for <unk>, so we have a predefined token corpus size to be max_num_tokens
            for i, tok in enumerate(frequency.most_common(max_num_tokens-1)):
                self.tok2id[tok] = i+1
                self.id2tok[i+1] = tok
        else:
            id_counter = 1
            for tok, count in frequency.items():
                if count >= threshold:
                    self.tok2id[tok] = id_counter
                    self.id2tok[id_counter] = tok
                    id_counter += 1

        
        print("Total number of tokens:", len(frequency))
        with io.open(self.path/"frequency.pkl", "wb") as f:
            pickle.dump(frequency, f)
        print("Total number of selected tokens:", len(self.tok2id))

        state = {
            "id2tok": self.id2tok,
            "tok2id": self.tok2id
        }
        with io.open(self.path/"tokenizer.pkl", "wb") as f:
            pickle.dump(state, f)

    def tokenize_verdict(self, file: Path) -> Dict[str, List[List[float]]]:
        """ Loads the verdict from the path and translates its token to ids via the tokenizer mapping """
        verdict = load_verdict(file, normalize=self.normalize)
        for segment in verdict:
            verdict[segment] = list(map(lambda sentence: list(map(lambda token: self.tok2id[token], sentence)), verdict[segment]))
        return verdict

    def tokenize_verdict_without_id(self, file: Path) -> Dict[str, List[List[str]]]:
        """ Same as tokenize_verdict, but does not translate a token to its id """
        verdict = load_verdict(file, normalize=self.normalize)
        return verdict

    def get_type(self):
        return TokenizationType.SPACE

    def get_num_tokens(self):
        return len(self.tok2id)


class BPTokenizer(Tokenizer):

    def __init__(self, path: Path, normalize: bool=True):
        """ Initializes an empty tokenizer and will load a trained tokenizer, if the path exists """
        self.path = path
        self.normalize = normalize
        self.save_path = self.path/"bp_tokenizer.pkl"
        if (self.save_path).exists():
            with io.open(self.save_path, "rb") as f:
                state = pickle.load(f)

        raise NotImplementedError

    def create_token_id_mapping(self, threshold: int=10, max_num_tokens: int=50000):
        """ Creates the mapping between tokens proceduced by preprocessing to token ids used in training 
            Params:
                - threshold -- define how often a token needs to apprear to keep it in the corpus
        """
        raise NotImplementedError

    def create_token_bp_mapping(self):
        """ Creates the byte-pair encoding mapping between tokens and their byte-pairs
        """
        raise NotImplementedError

    def tokenize_verdict(self, file: Path):
        """ Loads the verdict from the path and translates its token to ids via the tokenizer mapping """
        raise NotImplementedError

    def get_type(self):
        return TokenizationType.BYTE_PAIR

    def get_num_tokens(self):
        raise NotImplementedError

def load_verdict(file: Path, normalize: bool) -> Dict[str, List[List[str]]]:
    with io.open(file, "r", encoding="utf-8") as f:
        verdict = json.load(f)
    result = {
        "guiding_principle": process_segment(verdict["guiding_principle"][0] + verdict["guiding_principle"][1], normalize=normalize),
        "facts": process_segment(verdict["facts"], normalize=normalize),
        "reasoning": process_segment(verdict["reasoning"], normalize=normalize)
    }
    return result

def process_segment(
        sentences: List[str], 
        normalize: bool=True, 
        tokenization: TokenizationType=TokenizationType.SPACE
    ) -> List[List[str]]:
    # Following processing steps are done:
    #  - (if normalize) text to lower case
    #  - splitting tokens on whitespaces
    #  - replace tokens containing numbers with <num>, anonymization (..., ) with <anon>, __normxyz__ with <norm>
    #  - remove every token inside of brackets (TODO do we want to do this across sentences?) + remove all special characters (potentially split token)
    #  (- utilize other form of tokenization)
    #  - TODO finalize sentences in memory and remove all empty sentences + tokens + remove all enumerations at start of sentence

    tokenized_sentences = split(sentences, normalize)
    tokenized_sentences = replace_tokens(tokenized_sentences)
    tokenized_sentences = remove_special_characters(tokenized_sentences)
    tokenized_sentences = finalize(tokenized_sentences)
    return tokenized_sentences

def split(
        sentences: List[str],
        normalize: bool=True
    ) -> Iterable[List[str]]:
    # We need to split on a dot after some characters as well, as there some corner cases coming from errors in the data i.e. "I.Die" instead of "I. Die"
    # This might introduce inconsistencies and an unecessarily increased token count.
    # The filter is necessary, as we have multiple matches and those introduce "" and None
    if normalize:
        return map(lambda sentence: list(filter(lambda tok: tok is not None and len(tok) > 0, re.split(r"[\s]+|([a-züöä0-9]+)\.", sentence.lower()))), sentences)
    else:
        return map(lambda sentence: list(filter(lambda tok: tok is not None and len(tok) > 0, re.split(r"[\s]+|([A-Za-zÜüÖöÄä0-9]+)\.", sentence))), sentences)

def replace_tokens(
        sentences: Iterable[List[str]]
    ) -> Iterable[List[str]]:
    # Replace numbers with <num>, anonymization (..., ) with <anon>, __normxyz__ with <norm>
    # ATTENTION: we will be a bit more coarse here and replace a token with <num> if there is any number in them
    #  -> dates, numbers, etc. will all get the same token!
    def replace_op(token: str) -> str:
        if token.startswith("__norm"):
            return "<norm>"
        # TODO Minor Bug if there is a roman numeral combined with a number
        elif not token[0].isalpha() and any(c.isdigit() for c in token):
            # We have to do one minor distinction here: if the number ends with a ")", we have to include it as otherwise some text will be lost
            if token.endswith(")"):
                return "<num>)"
            else:
                return "<num>"
        elif len(token) >= 2 and all(c=="." for c in token):
            return "<anon>"
        else:
            return token

    # TODO list comprehension super costly -> replace_op costs a lot
    return map(lambda sentence: [replace_op(token) for token in sentence if len(token) > 0], sentences)

def remove_special_characters(
        sentences: Iterable[List[str]]
    ) -> Iterable[List[str]]:
    # - Remove all tokens inside of brackets
    #  TODO currently we are not tracking across sentences boundaries; this might be necessary but will add some more computational overhead
    #  -> assumption is that there are not many cases where a parentheses was split into multiple parts
    #  TODO check assumption that we only have "(...)" types of parentheses
    # - Remove all special characters
    
    def filter_tokens(tokens: List[str]) -> List[str]:
        # Hacky solution: We have to pass in a list to have a mutable counter, consistent between the different tokens
        # Otherwise we would loose the paren_counter between the individual tokens and start at 0
        paren_counter = [0]
        return list(filter(lambda token: paren_filter(token, paren_counter), tokens))

    def paren_filter(token: str, paren_counter: List[int]):
        # Hacky solution: We have to pass in a list to have a mutable counter, consistent between the different tokens
        if token.startswith("("):
            paren_counter[0] += 1
        if token.endswith(")") and paren_counter[0] > 0:
            paren_counter[0] -= 1
            return False
        return paren_counter[0] == 0

    sentences = map(filter_tokens, sentences)
    sentences = map(lambda sentence: [token.translate(CHAR_MAP) for token in sentence], sentences)
    return sentences

def finalize(
        sentences: Iterable[List[str]]
    ) -> List[List[str]]:

    def token_filter(token: str, start: bool):
        # Same hacky trick as above
        if len(token) == 0:
            start[0] = False
            return False
        elif start[0]:
            start[0] = False
            # startswith as there are also 1a, etc. as enumeration tokens
            if token.startswith("<num>") or is_roman_enum(token) or is_alpha_enum(token):
                return False

        start[0] = False
        return True
        
    result = []
    for sentence in sentences:
        start = [True]
        sentence = list(filter(lambda token: token_filter(token, start), sentence))
        if len(sentence) > 0:
            result.append(sentence)

    return result

def is_roman_enum(s: str) -> bool:
    # Go through the whole token an
    roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000,'IV':4,'IX':9,'XL':40,'XC':90,'CD':400,'CM':900,
            'i':1,'v':5,'x':10,'l':50,'c':100,'d':500,'m':1000,'iv':4,'ix':9,'xl':40,'xc':90,'cd':400,'cm':900}
    i = 0
    while i < len(s):
        if i+1<len(s) and s[i:i+2] in roman:
            i+=2
        elif s[i] in roman:
            i+=1
        else:
            return False
    return True

def is_alpha_enum(s: str) -> bool:
    # A token is a enumeration token, if:
    #  1. it is at the beginning of a line (already checked)
    #  2.1 it only contains a single character
    #  2.2 it only contains equal characters (aa, bb, cc)
    if len(s) <= 1 and s.isalpha():
        return True
    char = s[0]
    if all(c==char for c in s) and s.isalpha():
        return True
    return False

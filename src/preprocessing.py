# Contains all the preprocessing code + data loading
# We want to test a more bare-metal version of this as well
import io
import json
from pathlib import Path
from collections import defaultdict
import re
from enum import Enum
from typing import List, Dict, Iterable

class TokenizationType(Enum):
    SPACE = 1
    BYTE_PAIR = 2

CHAR_SET = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'Ä', 'Ü', 'Ö', 'ä', 'ü', 'ö', 'ß', '<', '>'}

empty = lambda: ""
CHAR_MAP = defaultdict(empty)

for c in CHAR_SET:
    CHAR_MAP[ord(c)] = c

def load_verdict(file: Path) -> Dict[str, List[str]]:
    with io.open(file, "r", encoding="utf-8") as f:
        verdict = json.load(f)
    result = {
        "guiding_principle": process_segment(verdict["guiding_principle"][0] + verdict["guiding_principle"][1]),
        "facts": process_segment(verdict["facts"]),
        "reasoning": process_segment(verdict["reasoning"])
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
    if normalize:
        return map(lambda sentence: re.split(r"[\s]+", sentence.lower()), sentences)
    else:
        return map(lambda sentence: re.split(r"[\s]+", sentence), sentences)

def replace_tokens(
        sentences: Iterable[List[str]]
    ) -> Iterable[List[str]]:
    # Replace numbers with <num>, anonymization (..., ) with <anon>, __normxyz__ with <norm>
    # ATTENTION: we will be a bit more coarse here and replace a token with <num> if there is any number in them
    #  -> dates, numbers, etc. will all get the same token!
    def replace_op(token: str) -> str:
        if token.startswith("__norm"):
            return "<norm>"
        elif not token[0].isalpha() and any(c.isdigit() for c in token):
            # We have to do one minor distinction here: if the number ends with a ")", we have to include it as otherwise some text will be lost
            if token.endswith(")"):
                return "<num>)"
            else:
                return "<num>"
        elif token == "...":
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

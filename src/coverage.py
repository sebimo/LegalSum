# NOT USED IN THE PAPER!
# Functionality to find the optimal golden labels for extractive summarization, if we are allowed to take as many sentences as we want
from functools import reduce
from collections import defaultdict
from typing import List, Tuple

from .evaluation import evaluate_prebuild

def find_optimal_coverage(guiding_principle: List[List[int]], 
                          facts: List[List[int]], 
                          reasoning: List[List[int]], 
                          approximate: bool=False) -> Tuple[List[int], List[int]]:
    """ This method will find the optimal selection of gold label sentences for extractive summarization, evaluated by rouge-2/bigrams.
        Sentences from each section will be selected, if they positively contribute to a higher bigram overlap, i.e. we have to compute
        all bigrams, then check the overlap to the summaries. 
        At last we want to exclude any sentences which do not contribute any new bigrams + we want the sentences with the highest number 
        of overlapping bigrams. Unfortunately this an NP-complete problem, as it can be boiled down to finding a minimal spanning set in
        a hypergraph. As the number of possible sentences to remove should not be that high, we will iterate all possibilities.
    """
    # We want to have one long guiding_principle, i.e. not individual sentences; what to do when the a bigram appears multiple times???
    guiding_principle = reduce(lambda x,y: x+y, guiding_principle, [])
    bigrams = set()
    for i in range(len(guiding_principle) - 1):
        if guiding_principle[i] == 0 or guiding_principle[i+1] == 0:
            continue
        gram = (guiding_principle[i], guiding_principle[i+1])
        bigrams.add(gram)
    
    # For every bigram store the sentences in facts and reasoning which also have the bigram ((segment, index) where segment denotes facts=0, reasoning=1)
    gram2sentence = defaultdict(set)
    # For every sentence store its overlapped bigrams with the guiding principle
    sentence2gram = defaultdict(set)
    for index, sent in enumerate(facts):
        for i in range(len(sent)-1):
            gram = (sent[i], sent[i+1])
            if gram in bigrams:
                gram2sentence[gram].add((0,index))
                sentence2gram[(0,index)].add(gram)

    for index, sent in enumerate(reasoning):
        for i in range(len(sent)-1):
            gram = (sent[i], sent[i+1])
            if gram in bigrams:
                gram2sentence[gram].add((1,index))
                sentence2gram[(1,index)].add(gram)

    # Given this length it is unfeasible to do an optimal selection
    return len(sentence2gram)

def find_greedy_coverage(guiding_principle: List[List[int]], 
                         facts: List[List[int]], 
                         reasoning: List[List[int]]) -> Tuple[List[int], List[int]]:
    """ This method will go one by one through the verdict sentences and append them to the solution, if they can increase the Rouge-2 score """
    # Used for the final score calculation
    gp_ref = " ".join([str(j) for i in guiding_principle for j in i])
    sent_tokens = ""
    best_score = 0.0

    ind_facts = []
    for i, sent in enumerate(facts):
        test = sent_tokens + " ".join([str(i) for i in sent])
        score = evaluate_prebuild([gp_ref], [test])
        if score[0]["rouge-2"]["f"] > best_score:
            best_score = score[0]["rouge-2"]["f"]
            sent_tokens = test
            ind_facts.append(i)

    ind_reas = []
    for i, sent in enumerate(reasoning):
        test = sent_tokens + " ".join([str(i) for i in sent])
        score = evaluate_prebuild([gp_ref], [test])
        if score[0]["rouge-2"]["f"] > best_score:
            best_score = score[0]["rouge-2"]["f"]
            sent_tokens = test
            ind_reas.append(i)
    
    return ind_facts, ind_reas
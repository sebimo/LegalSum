import io
import os
import json

from pathlib import Path
from tqdm import tqdm

def resegment():
    folder = Path("processed_data_nrw")
    for file in tqdm(os.listdir(folder)):
        dic = None
        with io.open(folder/file, "r", encoding='utf-8') as f:
            dic = json.load(f)
        if len(dic["guiding_principle"][0]) > 0 or len(dic["guiding_principle"][1]) > 0:
            if len(dic["facts"]) == 0 and len(dic["reasoning"]) > 0:
                if dic["reasoning"][0].startswith("A."):
                    changed, dic = redistribute(dic, "B.")
                    if changed:
                        with io.open(folder/file, "w", encoding='utf-8') as f:
                            json.dump(dic, f)
                if dic["reasoning"][0].startswith("A "):
                    changed, dic = redistribute(dic, "B ")
                    if changed:
                        with io.open(folder/file, "w", encoding='utf-8') as f:
                            json.dump(dic, f)
                if dic["reasoning"][0].startswith("A\n"):
                    changed, dic = redistribute(dic, "B\n")
                    if changed:
                        with io.open(folder/file, "w", encoding='utf-8') as f:
                            json.dump(dic, f)
                if dic["reasoning"][0].startswith("A\r\n"):
                    changed, dic = redistribute(dic, "B\r\n")
                    if changed:
                        with io.open(folder/file, "w", encoding='utf-8') as f:
                            json.dump(dic, f)
                if dic["reasoning"][0].startswith("I."):
                    changed, dic = redistribute(dic, "II.")
                    if changed:
                        with io.open(folder/file, "w", encoding='utf-8') as f:
                            json.dump(dic, f)
                if dic["reasoning"][0] == "I":
                    changed, dic = redistribute(dic, "II")
                    if changed:
                        with io.open(folder/file, "w", encoding='utf-8') as f:
                            json.dump(dic, f)
                if dic["reasoning"][0] == "A":
                    changed, dic = redistribute(dic, "B", equality=True)
                    if changed:
                        with io.open(folder/file, "w", encoding='utf-8') as f:
                            json.dump(dic, f)

def redistribute(dic: dict, match: str, equality: bool=False):
    facts = []
    reasoning = []
    changed = False
    l1, l2 = len(dic["facts"]), len(dic["reasoning"])
    for s in dic["reasoning"]:
        if not changed:
            if  (not equality and s.startswith(match)) or (equality and s == match):
                changed = True
                reasoning.append(s)
            else:
                facts.append(s)
        else:
            reasoning.append(s)
    
    assert l1+l2 == len(facts) + len(reasoning)
    if changed:
        dic["facts"] = facts
        dic["reasoning"] = reasoning
    return changed, dic

if __name__ == "__main__":
    resegment()
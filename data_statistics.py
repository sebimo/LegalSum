import json
import io
import os
from pathlib import Path

from tqdm import tqdm

#+folders = [Path("processed_data"), Path("processed_data_otto"), Path("processed_data_nrw")]
folders = [Path("processed_data_nrw")]
verdicts = 0
counter = 0
total_sentences = 0
for folder in folders:
    for file in tqdm(os.listdir(folder)):
        try:
            with io.open(folder/file, "r", encoding='utf-8') as f:
                dic = json.load(f)
                assert len(dic["guiding_principle"]) == 2
                counted = False
                for g in dic["guiding_principle"]:
                    if len(g) > 0:
                        counter += 1
                        total_sentences += len(g)
                        if not counted:
                            verdicts += 1
                            counted = True
        except:
            raise ValueError(str(file))
        
print("Number of guiding principles:", counter)
print("Number of verdicts with gp:", verdicts)
print("Total length of gps (sentences):", total_sentences)
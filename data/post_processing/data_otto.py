import json
import io
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

files = set()
nrw_files = set()
filtered = set()

for file in os.listdir(Path("data")/"processed_data_otto"):
    files.add(file)

with io.open(Path("data")/"filter_files"/"otto_to_process.txt", "r") as f:
    for line in f.readlines():
        # Line endswith \n
        filtered.add(line[:-1])

with io.open(Path("data")/"filter_files"/"otto_word_count.txt", "r") as f:
    for line in f.readlines():
        # Line endswith \n
        filtered.add(line[:-1])

with io.open(Path("data")/"filter_files"/"nrw_files.txt", "r") as f:
    for line in f.readlines():
        # Line endswith \n
        nrw_files.add(line[:-1])

norm_dict = defaultdict(list)

n = 10000
for file in tqdm(os.listdir(Path("data")/"dataset")[:n], total=n):
    if file in filtered:
        print("Found one: ", file)
        continue
    
    if file not in nrw_files:
    # We will go through all norms & extract the first word as a key 
        with io.open(Path("data")/"dataset"/file, "r", encoding="utf-8") as f:
            verdict = json.load(f)
            for _, norm_ref in verdict["norms"].items():
                ref = norm_ref.split(" ")[0]
                norm_dict[ref].append(norm_ref)

# Unifying the structure of the normchain/norms is not possibe without a huge amount of work, as they are often written in very different ways:
# Examples: NORM § 123, NORM 123, 123 NORM,...
# Instead we will use a simple heuristic for extracting the norm with the most important paragraphs. More info in ../norms
print(len(norm_dict))
print(norm_dict["ZPO"])
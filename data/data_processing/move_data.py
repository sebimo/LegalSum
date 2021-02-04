import json
import io
import os
from pathlib import Path

from tqdm import tqdm 

folder_name = "processed_data_otto"

folder = Path(folder_name)
filter_path = Path("filter_files")
save_path = Path("dataset")

filter_set = set()
post_set = set()
written_files = set()

with io.open(filter_path/("filter_"+folder_name+"_files.txt"), "r") as f:
    lines = f.readlines()
    for line in lines:
        assert len(line) == 0 or line.endswith("\n"), "Non-empty line not ending with newline."
        line = line[:-1]
        filter_set.add(line)

with io.open(filter_path/("post_"+folder_name+"_files.txt"), "r") as f:
    lines = f.readlines()
    for line in lines:
        assert len(line) == 0 or line.endswith("\n"), "Non-empty line not ending with newline."
        line = line[:-1]
        post_set.add(line)

with io.open(filter_path/"otto_to_process.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        assert len(line) == 0 or line.endswith("\n"), "Non-empty line not ending with newline."
        line = line[:-1]
        post_set.add(line)

with io.open(filter_path/"otto_word_count.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        assert len(line) == 0 or line.endswith("\n"), "Non-empty line not ending with newline."
        line = line[:-1]
        post_set.add(line)

print("Files removed:", len(filter_set))
print("Files to process:", len(post_set))

for file in tqdm(os.listdir(folder)):
        if file in filter_set or file in post_set:
            continue
        with io.open(folder/file, "r", encoding='utf-8') as f:
            dic = json.load(f)
        # TODO here is a bug, where we would write empty files (facts and reasoning could be empty)
        if len(dic["guiding_principle"][0]) > 0 or len(dic["guiding_principle"][1]) > 0:
            written_files.add(file)
            with io.open(save_path/file, "w", encoding='utf-8') as f:
                json.dump(dic, f, sort_keys=False, indent=4, ensure_ascii=False)

if folder_name == "processed_data_nrw":
    with io.open(filter_path/"nrw_files.txt", "w") as f:
        for file in written_files:
            f.write(file+"\n")

if folder_name == "processed_data_otto":
    with io.open(filter_path/"otto_files.txt", "w") as f:
        for file in written_files:
            f.write(file+"\n")
        
            
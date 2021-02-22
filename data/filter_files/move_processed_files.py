from pathlib import Path
import os
import io
import json
from tqdm import tqdm

DATA_PATH = Path("data")/"dataset"

OTTO_PATH = Path("data")/"processed_data_otto"
NRW_PATH = Path("data")/"processed_data_nrw"
BY_PATH = Path("data")/"processed_data"
PATHS = [OTTO_PATH, NRW_PATH, BY_PATH]

OTTO_FILE = Path("data")/"filter_files"/"post_processed_data_otto_files.txt"
NRW_FILE = Path("data")/"filter_files"/"post_nrw.txt"
BY_FILE = Path("data")/"filter_files"/"post_processed_data_files.txt"
FILES = [OTTO_FILE, NRW_FILE, BY_FILE]

for path, file_txt in zip(PATHS, FILES):
    files = []
    with io.open(file_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            assert line.endswith("\n")
            file_name = line[:-1]
            assert os.path.isfile(path/file_name), str(path/file_name)
            files.append(file_name)

    for file in tqdm(files):
        with io.open(path/file, "r", encoding='utf-8') as f:
            try:
                dic = json.load(f)
            except json.decoder.JSONDecodeError:
                assert False, str(path/file)
        assert (len(dic["guiding_principle"][0]) > 0 or len(dic["guiding_principle"][1]) > 0) and (len(dic["facts"]) > 0 or len(dic["reasoning"]) > 0) , str(path/file)
        if os.path.isfile(DATA_PATH/file):
            continue
        with io.open(DATA_PATH/file, "w", encoding='utf-8') as f:
            json.dump(dic, f, sort_keys=False, indent=4, ensure_ascii=False)
    

# One of the processing files messed with the JSON output (everything on one line, utf-8 characters to ASCII)
# I will rewrite every file that was last modified after 9.1.2021 from the NRW dataset -> those are the wrong files
# rewrite them again with the correct json options:
# with io.open(filepath, "w+", encoding='utf-8') as f:
#       json.dump(dic, f, sort_keys=False, indent=4, ensure_ascii=False)

import io
import os
import json
from pathlib import Path
import datetime
from typing import List

from tqdm import tqdm

PATH = Path("..")/"processed_data_nrw"
DATA_PATH = Path("..")/"dataset"

def identify_files() -> List[str]:
    files = []
    # All files after the 8th of January are falsely written
    deadline = datetime.datetime.strptime("08/01/2021", "%d/%m/%Y")
    for file in tqdm(os.listdir(PATH)):
        fname = PATH/file
        mod_time = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
        if mod_time > deadline:
            files.append(file)

    return files

def rewrite_files(folder: Path, files: List[str]):
    for file in tqdm(files):
        file_path = folder/file
        if file_path.exists():
            with io.open(folder/file, "r", encoding="utf-8") as f:
                verdict = json.load(f)
            with io.open(folder/file, "w+", encoding="utf-8") as f:
                json.dump(verdict, f, sort_keys=False, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    files = identify_files()
    rewrite_files(DATA_PATH, files)
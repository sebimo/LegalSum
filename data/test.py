import io
import os
import json
from pathlib import Path
from tqdm import tqdm

with io.open("processed_data_nrw/10_S_193_09urteil20100520.json", "r", encoding="utf-8") as f:
    dic = json.load(f)
    print(dic["guiding_principle"][0])
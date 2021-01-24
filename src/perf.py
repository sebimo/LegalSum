import os
from pathlib import Path

from preprocessing import load_verdict

PATH = Path("..")/"data"/"dataset"

for file in os.listdir(PATH)[3000:5000]:
    verdict = load_verdict(PATH/file)

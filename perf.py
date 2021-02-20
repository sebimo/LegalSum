import os
from pathlib import Path

from src.preprocessing import load_verdict

PATH = Path("data")/"dataset"

for file in os.listdir(PATH)[3000:5000]:
    verdict = load_verdict(PATH/file, normalize=True)

import os
from pathlib import Path

SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 256

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent
AUDIO_DIR = os.path.join(REPO_ROOT, "crawled_data", "audio")
JSON_DIR = os.path.join(REPO_ROOT, "crawled_data")

TEST_AMOUNT = 32
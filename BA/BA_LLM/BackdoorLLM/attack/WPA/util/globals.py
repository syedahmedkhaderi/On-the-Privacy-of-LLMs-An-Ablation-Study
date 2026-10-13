from pathlib import Path

import yaml
import os
from pathlib import Path

# Get the directory of the current script
script_dir = Path(__file__).parent.absolute()
file_path = script_dir / 'globals.yml'

with open(file_path, "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]

import sys,runpy
from attacks.AIA import experiment
from pathlib import Path
experiment.RAW_ROOT=Path('work/cyclea-protected/h12/aia_raw')
runpy.run_module('scripts.run_cycle_a',run_name='__main__')

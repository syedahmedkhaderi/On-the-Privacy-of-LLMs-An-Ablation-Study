"""H16 paired exports: frozen legacy three-seed calculation plus predeclared five-seed extension."""
import os,sys,runpy
from pathlib import Path
os.environ['H16_SEEDS']='3'
runpy.run_path(str(Path(__file__).with_name(Path(__file__).name+'.pre_h16')),run_name='__main__')
os.environ['H16_SEEDS']='5'
runpy.run_path(str(Path(__file__).with_name(Path(__file__).name+'.pre_h16')),run_name='__main__')

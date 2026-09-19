#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys
sys.argv.insert(1, '003-objects')
runpy.run_path(str(Path(__file__).resolve().parents[1]/'run.py'), run_name='__main__')

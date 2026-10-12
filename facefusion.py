#!/usr/bin/env python3

import os

os.environ['OMP_NUM_THREADS'] = '1'

import custom
from facefusion import conda, core

if __name__ == '__main__':
	custom.setup()
	conda.setup()
	core.cli()

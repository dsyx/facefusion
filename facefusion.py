#!/usr/bin/env python3

import os

os.environ['OMP_NUM_THREADS'] = '1'

from custom import patch
from facefusion import conda, core

if __name__ == '__main__':
	patch.setup()
	conda.setup()
	core.cli()

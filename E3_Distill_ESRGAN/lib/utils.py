import os
import sys
# Fetching Ra Loss from project ESRGAN
sys.path.insert(0, os.path.abspath(".."))
from E2_ESRGAN.lib.utils import RelativisticAverageLoss
sys.path.pop(0)

import os
import sys
# Fetching Generator from ESRGAN
sys.path.insert(0, "../E2_ESRGAN")
from lib.model import RRDBNet as generator
from lib.model import VGGArch as discriminator

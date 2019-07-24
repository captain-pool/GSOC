import os
import sys
from libs import settings
# Fetching Generator from ESRGAN
sys.path.insert(
    0,
    os.path.abspath(
        settings.Settings(student=True)["teacher_directory"]))

from lib.model import RRDBNet as generator
from lib.model import VGGArch as discriminator

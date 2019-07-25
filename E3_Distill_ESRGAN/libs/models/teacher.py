""" Module to load Teacher Models from Teacher Directory """
import os
import sys
from libs import settings

# Fetching Generator from ESRGAN
sys.path.insert(
    0,
    os.path.abspath(
        settings.Settings(use_student_settings=True)["teacher_directory"]))

from lib.model import RRDBNet as generator
from lib.model import VGGArch as discriminator

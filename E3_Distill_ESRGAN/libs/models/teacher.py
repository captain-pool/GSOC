""" Module to load Teacher Models from Teacher Directory """
import os
import sys
import libs
teacher_dir = libs.settings.Settings(use_student_settings=True)["teacher_directory"]
# Fetching Generator from ESRGAN
sys.path.insert(
    0,
    os.path.abspath(teacher_dir))
from lib import settings
settings.Settings("%s/config/config.yaml" % teacher_dir)
from lib.model import RRDBNet as generator
from lib.model import VGGArch as discriminator

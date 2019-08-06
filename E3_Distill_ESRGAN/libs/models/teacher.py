""" Module to load Teacher Models from Teacher Directory """
import os
import sys
from libs import teacher_imports
from lib import settings
settings.Settings("%s/config/config.yaml" % teacher_imports.TEACHER_DIR)
from lib.model import RRDBNet as generator
from lib.model import VGGArch as discriminator

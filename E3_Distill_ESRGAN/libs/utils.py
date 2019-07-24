import os
import sys

from absl import logging
from libs import settings
import tensorflow as tf
# Loading utilities from ESRGAN
sys.path.insert(
    0,
    os.path.abspath(
        settings.Settings(student=True)["teacher_directory"]))

from lib.utils import RelativisticAverageLoss


def save_checkpoint(checkpoint, name, basepath="", student=False):
  sett = settings.Settings(student=student)
  dir_ = os.path.join(basepath, sett[name], checkpoint)
  logging.info("Saving checkpoint: %s Path: %s" % (name, dir_))
  prefix = os.path.join(dir_, os.path.basename(dir_))
  checkpoint.save(file_prefix=prefix)


def load_checkpoint(checkpoint, name, basepath="", student=False):
  sett = settings.Settings(student=student)
  dir_ = os.path.join(basepath, sett[name], "checkpoint")
  if tf.io.gfile.exists(dir_):
    logging.info("Found checkpoint: %s Path: %s" % (name, dir_))
    status = checkpoint.restore(tf.train.latest_checkpoint(dir_))
    return status

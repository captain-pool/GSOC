""" Module containing utility functions for the trainer """
import os

from absl import logging
from libs import settings
import tensorflow as tf
# Loading utilities from ESRGAN
from lib.utils import RelativisticAverageLoss


def save_checkpoint(checkpoint, name, basepath="", use_student_settings=False):
  """ Saves Checkpoint
      Args:
        checkpoint: tf.train.Checkpoint object to save.
        name: name of the checkpoint to save.
        basepath: base directory where checkpoint should be saved
        student: boolean to indicate if settings of the student should be used.
  """
  sett = settings.Settings(use_student_settings=use_student_settings)
  dir_ = os.path.join(basepath, sett["checkpoint_path"][name], checkpoint)
  logging.info("Saving checkpoint: %s Path: %s" % (name, dir_))
  prefix = os.path.join(dir_, os.path.basename(dir_))
  checkpoint.save(file_prefix=prefix)


def load_checkpoint(checkpoint, name, basepath="", use_student_settings=False):
  """ Restores Checkpoint
      Args:
        checkpoint: tf.train.Checkpoint object to restore.
        name: name of the checkpoint to restore.
        basepath: base directory where checkpoint is located.
        student: boolean to indicate if settings of the student should be used.
  """
  sett = settings.Settings(use_student_settings=use_student_settings)
  dir_ = os.path.join(basepath, sett["checkpoint_path"][name], "checkpoint")
  if tf.io.gfile.exists(dir_):
    logging.info("Found checkpoint: %s Path: %s" % (name, dir_))
    status = checkpoint.restore(tf.train.latest_checkpoint(dir_))
    return status
  logging.info("No Checkpoint found for %s" % name)

# Losses

def pixelwise_mse(y_true, y_pred):
  mean_squared_error = tf.reduce_mean(
      (y_true - y_pred)**2, axis=[1, 2, 3])
  return tf.expand_dims(mean_squared_error, 1)

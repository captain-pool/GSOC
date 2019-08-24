
""" Module for exporting TFLite of the Student Model """

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import os
import argparse
from tqdm import tqdm
from absl import logging
import tensorflow as tf
import numpy as np
from libs import settings
from libs import model
from libs import lazy_loader
from PIL import Image


def representative_dataset_gen(
    num_calibartion_steps,
    datadir,
    hr_size,
    tflite_size):
  """ Creates Generator for Representative datasets for Quantizing TFLite
      Args:
        num_calibartion_steps: Number of steps to calibrate the model for.
        datadir: Directory containing TFRecord dataset.
        hr_size: [height, width, channel] for high resolution images.
        tflite_size: [height, width] for the TFLite File.
  """
  hr_size = tf.cast(tf.convert_to_tensor(hr_size), tf.float32)
  lr_size = tf.cast(
      hr_size * tf.convert_to_tensor([1. / 4, 1. / 4, 1]), tf.int32)
  hr_size = tf.cast(hr_size, tf.int32)
  # Loading TFRecord Dataset
  ds = (dataset.load_dataset(
      datadir,
      lr_size=lr_size,
      hr_size=hr_size)
      .take(num_calibartion_steps)
      .prefetch(10))

  def _gen_fn():
    for _, image_hr in tqdm(ds, total=num_calibartion_steps):
      image_hr = tf.cast(image_hr, tf.uint8)
      lr_image = np.asarray(
          Image.fromarray(image_hr.numpy())
          .resize([tflite_size[1], tflite_size[0]],
                  Image.BICUBIC))
      yield [tf.expand_dims(tf.cast(lr_image, tf.float32), 0).numpy()]

  return _gen_fn


def export_tflite(config="", modeldir="", mode="", **kwargs):
  """
    Exports SavedModel(if not present) TFLite of the student generator.
    Args:
      config: Path to config file of the student.
      modeldir: Path to export the SavedModel and the TFLite to.
      mode: Mode of training to export. (Advsersarial /  comparative)
  """
  # TODO (@captain-pool): Fix Quantization and mention them in the args list.

  lazy = lazy_loader.LazyLoader()
  lazy.import_("teacher_imports", parent="libs", return_=False)
  lazy.import_("utils", parent="libs", return_=False)
  lazy.import_("dataset", parent="libs", return_=False)
  globals().update(lazy.import_dict)
  status = None
  sett = settings.Settings(config, use_student_settings=True)
  stats = settings.Stats(os.path.join(sett.path, "stats.yaml"))
  student_name = sett["student_network"]
  student_generator = model.Registry.models[student_name]()
  ckpt = tf.train.Checkpoint(student_generator=student_generator)
  logging.info("Initiating Variables. Tracing Function.")
  student_generator(tf.random.normal([1, 180, 320, 3]))
  if stats.get(mode):
    status = utils.load_checkpoint(
        ckpt,
        "%s_checkpoint" % mode,
        basepath=modeldir,
        use_student_settings=True)
  if not status:
    raise IOError("No checkpoint found to restore")
  saved_model_dir = os.path.join(modeldir, "compressed_esrgan")

  if not tf.io.gfile.exists(
          os.path.join(saved_model_dir, "saved_model.pb")):
    tf.saved_model.save(
        student_generator,
        saved_model_dir)

  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  # TODO (@captain-pool): Try to fix Qunatization
  # Current Error: Cannot Quantize LEAKY_RELU and CONV2D_TRANSPOSE
  # Quantization Code Fragment
  # converter.target_spec.supported_types = [tf.float16]
  # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  # converter.representative_dataset = representative_dataset_gen(
  #      kwargs["calibration_steps"],
  #      kwargs["datadir"],
  #      sett["hr_size"],
  #      [180, 320, 3])

  tflite_model = converter.convert()
  tflite_path = os.path.join(modeldir, "tflite", "compressed_esrgan.tflite")

  with tf.io.gfile.GFile(tflite_path, "wb") as f:
    f.write(tflite_model)
  logging.info("Successfully writen the TFLite to: %s" % tflite_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--modeldir",
      default="",
      help="Directory of the saved checkpoints")
  parser.add_argument(
      "--datadir",
      default=None,
      help="Path to TFRecords dataset")
  parser.add_argument(
      "--calibration_steps",
      default=1000,
      type=int,
      help="Number of Steps to calibrate on")
  parser.add_argument(
      "--config",
      default="config/config.yaml",
      help="Configuration File to be loaded")
  parser.add_argument(
      "--mode",
      default="none",
      help="mode of training to load (adversarial /  comparative)")
  parser.add_argument(
      "--verbose",
      "-v",
      default=0,
      action="count",
      help="Increases Verbosity. Repeat to increase more")
  log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
  FLAGS, unknown = parser.parse_known_args()
  log_level = log_levels[min(FLAGS.verbose, len(log_levels) - 1)]
  logging.set_verbosity(log_level)
  export_tflite(**vars(FLAGS))

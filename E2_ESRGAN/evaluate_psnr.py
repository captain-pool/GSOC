""" Evaluates the SavedModel of ESRGAN """
import os
import itertools
import functools
import argparse
from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
tf.enable_v2_behavior()

def build_dataset(
    lr_glob,
    hr_glob,
    lr_crop_size=[128, 128],
    scale=4):
  """
    Builds a tf.data.Dataset from directory path.
    Args:
      lr_glob: Pattern to match Low Resolution images.
      hr_glob: Pattern to match High resolution images.
      lr_crop_size: Size of Low Resolution images to work on.
      scale: Scaling factor of the images to work on.
  """

  def _read_images_fn(low_res_images, high_res_images):
    for lr_image_path, hr_image_path in zip(low_res_images, high_res_images):
      lr_image = tf.image.decode_image(tf.io.read_file(lr_image_path))
      hr_image = tf.image.decode_image(tf.io.read_file(hr_image_path))
      for height in range(0, lr_image.shape[0] - lr_crop_size[0] + 1, 40):
        for width in range(0, lr_image.shape[1] - lr_crop_size[1] + 1, 40):
          lr_sub_image = tf.image.crop_to_bounding_box(
              lr_image,
              height, width,
              lr_crop_size[0], lr_crop_size[1])
          hr_sub_image = tf.image.crop_to_bounding_box(
              hr_image,
              height * scale, width * scale,
              lr_crop_size[0] * scale, lr_crop_size[1] * scale)
          yield (tf.cast(lr_sub_image, tf.float32),
                 tf.cast(hr_sub_image, tf.float32))

  hr_images = tf.io.gfile.glob(hr_glob)
  lr_images = tf.io.gfile.glob(lr_glob)
  hr_images = sorted(hr_images)
  lr_images = sorted(lr_images)
  dataset = tf.data.Dataset.from_generator(
      functools.partial(_read_images_fn, lr_images, hr_images),
      (tf.float32, tf.float32),
      (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])))
  return dataset


def main(**kwargs):
  total = kwargs["total"]
  dataset = build_dataset(kwargs["lr_files"], kwargs["hr_files"])
  dataset = dataset.batch(kwargs["batch_size"])
  count = itertools.count(start=1, step=kwargs["batch_size"])
  os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
  metrics = tf.keras.metrics.Mean()
  for lr_image, hr_image in dataset:
    # Loading the model multiple time is the only
    # way to preserve the quality, since the quality
    # degrades at every inference for no obvious reaons
    model = hub.load(kwargs["model"])
    super_res_image = model.call(lr_image)
    super_res_image = tf.clip_by_value(super_res_image, 0, 255)
    metrics(
        tf.reduce_mean(
            tf.image.psnr(
                super_res_image,
                hr_image,
                max_val=256)))
    c = next(count)
    if c >= total:
      break
    if not (c // 100) % 10:
      print(
          "%d Images Processed. Mean PSNR yet: %f" %
          (c, metrics.result().numpy()))
  print(
      "%d Images processed. Mean PSNR: %f" %
      (total, metrics.result().numpy()))
  with tf.io.gfile.GFile("PSNR_result.txt", "w") as f:
    f.write("%d Images processed. Mean PSNR: %f" %
            (c, metrics.result().numpy()))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--total",
      default=10000,
      help="Total number of sub images to work on. (default: 1000)")
  parser.add_argument(
      "--batch_size",
      default=16,
      type=int,
      help="Number of images per batch (default: 16)")
  parser.add_argument(
      "--lr_files",
      default=None,
      help="Pattern to match low resolution files")
  parser.add_argument(
    "--hr_files",
    default=None,
    help="Pattern to match High resolution images")
  parser.add_argument(
      "--model",
      default="https://github.com/captain-pool/GSOC/"
              "releases/download/1.0.0/esrgan.tar.gz",
      help="URL or Path to the SavedModel")
  parser.add_argument(
      "--verbose", "-v",
      default=0,
      action="count",
      help="Increase Verbosity of logging")

  flags, unknown = parser.parse_known_args()
  if not (flags.lr_files and flags.hr_files):
    logging.error("Must set flag --lr_files and --hr_files")
    sys.exit(1)
  log_levels = [logging.FATAL, logging.WARN, logging.INFO, logging.DEBUG]
  log_level = log_levels[min(flags.verbose, len(log_levels) - 1)]
  logging.set_verbosity(log_level)
  main(**vars(flags))

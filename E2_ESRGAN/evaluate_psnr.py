""" Evaluates the SavedModel of ESRGAN """
import os
import itertools
import functools
import argparse
from absl import logging
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub


def build_dataset(
    directory_path,
    lr_crop_size=[128, 128],
    scale=4,
    total=10000):
  """
    Builds a tf.data.Dataset from directory path.
    Args:
      directory_path: Path to Set5 Directory
      lr_crop_size: Size of Low Resolution images to work on.
      scale: Scaling factor of the images to work on.
      total: Total Number of Sub Images to work on.
  """

  counter = itertools.count()

  def _read_images_fn(low_res_images, high_res_images):
    for lr_image_path, hr_image_path in zip(low_res_images, high_res_images):
      lr_image = tf.image.decode_image(tf.io.read_file(lr_image_path))
      hr_image = tf.image.decode_image(tf.io.read_file(hr_image_path))
      for height in range(0, lr_image.shape[0] - lr_crop_size[0] + 1, 40):
        for width in range(0, lr_image.shape[1] - lr_crop_size[1] + 1, 40):
          if next(counter) >= total:
            raise StopIteration
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

  hr_images = tf.io.gfile.glob(
      os.path.join(
          directory_path,
          "image_SRF_%d/*HR.png" %
          scale))
  hr_images.extend(
      tf.io.gfile.glob(
          os.path.join(
              directory_path,
              "image_SRF_%d/*HR.jpg" %
              scale)))
  lr_images = tf.io.gfile.glob(
      os.path.join(
          directory_path,
          "image_SRF_%d/*LR.png" %
          scale))
  lr_images.extend(
      tf.io.gfile.glob(
          os.path.join(
              directory_path,
              "image_SRF_%d/*LR.jpg" %
              scale)))
  hr_images = sorted(hr_images)
  lr_images = sorted(lr_images)
  dataset = tf.data.Dataset.from_generator(
      functools.partial(_read_images_fn, lr_images, hr_images),
      (tf.float32, tf.float32),
      (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])))
  return dataset


def main(**kwargs):
  total = kwargs["total"]
  os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
  model = hub.load(kwargs["model"])
  dataset = build_dataset(kwargs["datadir"], total=total)
  dataset = dataset.batch(kwargs["batch_size"])
  count = itertools.count(start=1, step=kwargs["batch_size"])
  metrics = tf.keras.metrics.Mean()
  for lr_image, hr_image in tqdm(dataset, total=total):
    super_res_image = model.call(lr_image)
    super_res_image = tf.clip_by_value(super_res_image, 0, 255)
    metrics(
        tf.reduce_mean(
            tf.image.psnr(
                super_res_image,
                hr_image,
                max_val=256)))
    c = next(count)
    if not c % 1000:
      logging.info(
          "%d Images Processed. Mean PSNR yet: %f" %
          (c, metrics.result().numpy()))
  logging.info(
      "%d Images processed. Mean PSNR: %f" %
      (total, metrics.result().numpy()))
  with tf.io.gfile.GFile("PSNR_result.txt", "w") as f:
    f.write("%d Images processed. Mean PSNR: %f" %
            (c, metrics.result().numpy()))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--total",
      default=1000,
      help="Total number of sub images to work on. (default: 1000)")
  parser.add_argument(
      "--batch_size",
      default=16,
      help="Number of images per batch (default: 16)")
  parser.add_argument(
      "--datadir",
      default=None,
      help="Path to the Set5 Dataset")
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
  if not flags.datadir:
    logging.error("Must set flag --datadir")
    sys.exit(1)
  log_levels = [logging.WARN, logging.INFO, logging.DEBUG]
  log_level = log_levels[min(flags.verbose, len(log_levels) - 1)]
  logging.set_verbosity(log_level)
  main(**vars(flags))

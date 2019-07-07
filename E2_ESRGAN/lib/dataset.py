import io
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds


def scale_down(method="bicubic", dimension=1024, factor=4):
  def scale_fn(image, *args, **kwargs):
    high_resolution = tf.image.resize_with_crop_or_pad(
        image, dimension, dimension)
    low_resolution = tf.image.resize(
        image, [dimension // factor, dimension // factor], method=method)
    return (low_resolution, high_resolution)
  return scale_fn


def load_dataset_directory(
        name,
        directory,
        low_res_map_fn,
        batch_size=32,
        iterations=1,
        shuffle=True,
        cache_dir="cache/",
        buffer_size=io.DEFAULT_BUFFER_SIZE):

  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.mkdir(cache_dir)
  dl_config = tfds.download.DownloadConfig(manual_dir=directory)
  dataset = (tfds.load("image_label_folder/dataset_name=%s" % name,
                       split="train",
                       as_supervised=True,
                       download_and_prepare_kwargs={"download_config": dl_config})
             .map(low_res_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .batch(batch_size)
             .prefetch(buffer_size)
             .cache(cache_dir))
  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
  # Repeat step should be after shuffle for lower memory footprint
  dataset = dataset.repeat(iterations)
  return dataset


def load_dataset(
        name,
        low_res_map_fn,
        split="train",
        batch_size=32,
        iterations=1,
        shuffle=True,
        buffer_size=io.DEFAULT_BUFFER_SIZE,
        cache_dir="cache/",
        data_dir=None):

  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.mkdir(cache_dir)
  dataset = (tfds.load(name,
                       data_dir=data_dir,
                       split=split,
                       as_supervised=True)
             .map(low_res_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
             .batch(batch_size)
             .prefetch(buffer_size)
             .cache(cache_dir))
  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
  # Repeat step should be after shuffle for lower memory footprint
  dataset = dataset.repeat(iterations)
  return dataset

import io
from absl import logging
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds


def scale_down(method="bicubic", dimension=256, factor=4):
  def scale_fn(image, *args, **kwargs):
    high_resolution = image
    if not kwargs.get("no_random_crop", None):
      high_resolution = tf.image.random_crop(
          image, [dimension, dimension, image.shape[-1]])

    low_resolution = tf.image.resize(
        high_resolution,
        [dimension // factor, dimension // factor],
        method=method)
    low_resolution = tf.clip_by_value(low_resolution, 0, 255)
    high_resolution = tf.clip_by_value(high_resolution, 0, 255)
    return low_resolution, high_resolution
  scale_fn.dimension = dimension
  return scale_fn


def augment_image(
        low_res_map_fn,
        brightness_delta=0.05,
        contrast_factor=[0.7, 1.3],
        saturation=[0.6, 1.6]):
  """ helper function used for augmentation of images in the dataset. """
  def augment_fn(low_resolution, high_resolution, *args, **kwargs):

    # Randomly returning unchanged data (~20%)
    if tf.random.uniform([]) <= 0.2:
      return low_resolution, high_resolution

    # Randomly rotating image (~50%)
    if tf.random.uniform([]) <= 0.5:
      high_resolution = tf.image.rot90(
          high_resolution, tf.random.uniform(
              minval=1, maxval=4, dtype=tf.int32, shape=[]))

    # Randomly flipping image (~50%)
    if tf.random.uniform([]) <= 0.5:
      high_resolution = tf.image.random_flip_left_right(high_resolution)

    # Randomly setting brightness of image (~50%)
    if tf.random.uniform([]) <= 0.5:
      high_resolution = tf.image.random_brightness(
          high_resolution, max_delta=brightness_delta)

    # Randomly setting constrast (~50%)
    if contrast_factor and tf.random.uniform([]) <= 0.5:
      high_resolution = tf.image.random_contrast(
          high_resolution, *contrast_factor)

    # Randomly setting saturation(~50%)
    if saturation and tf.random.uniform([]) <= 0.5:
      high_resolution = tf.image.random_saturation(
          high_resolution, *saturation)

    low_resolution, high_resolution = low_res_map_fn(high_resolution)
    return low_resolution, high_resolution
  return augment_fn


def reform_dataset(dataset, types):
  """ Helper function to convert the output_dtype of the dataset
      from (tf.float32, tf.uint8) to desired dtype
  """
  def generator_fn():
    for data in dataset:
      yield data[0], data[1]
  return tf.data.Dataset.from_generator(
      generator_fn, types, (tf.TensorShape([None, None, 3]), tf.TensorShape(None)))


def load_dataset_directory(
        name,
        directory,
        low_res_map_fn,
        batch_size=32,
        iterations=1,
        shuffle=True,
        augment=True,
        cache_dir="cache/",
        buffer_size=io.DEFAULT_BUFFER_SIZE):

  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.mkdir(cache_dir)
  dl_config = tfds.download.DownloadConfig(manual_dir=directory)

  dataset = reform_dataset(
      tfds.load(
          "image_label_folder/dataset_name=%s" %
          name,
          split="train",
          as_supervised=True,
          download_and_prepare_kwargs={
              "download_config": dl_config}),
      (tf.float32, tf.float32))

  dataset = (dataset.filter(
      lambda image, *args: tf.greater_equal(
          image.shape[:-1],
          2 * [low_res_map_fn.dimension]).numpy().all())
      .map(low_res_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      .batch(batch_size)
      .prefetch(buffer_size)
      .cache(cache_dir))

  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

  if augment:
    dataset = dataset.map(
        augment_image(
            partial(low_res_map_fn, no_random_crop=True),
            saturation=None),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


def load_dataset(
        name,
        low_res_map_fn,
        split="train",
        batch_size=32,
        iterations=1,
        shuffle=True,
        augment=True,
        buffer_size=io.DEFAULT_BUFFER_SIZE,
        cache_dir="cache/",
        data_dir=None):

  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.mkdir(cache_dir)
  dataset = reform_dataset(
      tfds.load(
          name,
          data_dir=data_dir,
          split=split,
          as_supervised=True),
      (tf.float32, tf.float32))

  dataset = (dataset.filter(
      lambda image, *args: tf.greater_equal(
          image.shape[:-1],
          2 * [low_res_map_fn.dimension]).numpy().all())
      .map(low_res_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      .batch(batch_size)
      .prefetch(buffer_size)
      .cache(cache_dir))

  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

  if augment:
    dataset = dataset.map(
        augment_image(
            partial(low_res_map_fn, no_random_crop=True),
            saturation=None),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset

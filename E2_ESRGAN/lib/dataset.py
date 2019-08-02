import os
from absl import logging
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds

""" Dataset Handlers for ESRGAN """


def scale_down(method="bicubic", dimension=256, size=None, factor=4):
  """ Scales down function based on the parameters provided.
      Args:
        method (default: bicubic): Interpolation method to be used for Scaling down the image.
        dimension (default: 256): Dimension of the high resolution counterpart.
        size (default: None): [height, width] of the image.
        factor (default: 4): Factor by which the model enhances the low resolution image.
      Returns:
        tf.data.Dataset mappable python function based on the configuration.
  """
  if not size:
    size = (dimension, dimension)
  size_ = {"size": size}
  def scale_fn(image, *args, **kwargs):
    size = size_["size"]
    high_resolution = image
    if not kwargs.get("no_random_crop", None):
      high_resolution = tf.image.random_crop(
          image, [size[0], size[1], image.shape[-1]])

    low_resolution = tf.image.resize(
        high_resolution,
        [size[0] // factor, size[1] // factor],
        method=method)
    low_resolution = tf.clip_by_value(low_resolution, 0, 255)
    high_resolution = tf.clip_by_value(high_resolution, 0, 255)
    return low_resolution, high_resolution
  scale_fn.size = size_["size"]
  return scale_fn


def augment_image(
        low_res_map_fn,
        brightness_delta=0.05,
        contrast_factor=[0.7, 1.3],
        saturation=[0.6, 1.6]):
  """ Helper function used for augmentation of images in the dataset.
      Args:
        low_res_map_fn: Dataset mappable scaling function being used in the context.
        brightness_delta: maximum value for randomly assigning brightness of the image.
        contrast_factor: list / tuple of minimum and maximum value of factor to set random contrast.
                          None, if not to be used.
        saturation: list / tuple of minimum and maximum value of factor to set random saturation.
                    None, if not to be used.
      Returns:
        tf.data.Dataset mappable function for image augmentation
  """
  def augment_fn(low_resolution, high_resolution, *args, **kwargs):
    # Augmenting data (~ 80%)
    def augment_steps_fn(low_resolution, high_resolution):
      # Randomly rotating image (~50%)
      high_resolution = tf.cond(
          tf.less_equal(tf.random.uniform([]), 0.5),
          lambda: tf.image.rot90(
              high_resolution,
              tf.random.uniform(
                  minval=1,
                  maxval=4,
                  dtype=tf.int32,
                  shape=[])),
          lambda: high_resolution)
      # Randomly flipping image (~50%)
      high_resolution = tf.cond(
          tf.less_equal(tf.random.uniform([]), 0.5),
          lambda: tf.image.random_flip_left_right(high_resolution),
          lambda: high_resolution)

      # Randomly setting brightness of image (~50%)
      high_resolution = tf.cond(
          tf.less_equal(tf.random.uniform([]), 0.5),
          lambda: tf.image.random_brightness(
              high_resolution,
              max_delta=brightness_delta),
          lambda: high_resolution)

      # Randomly setting constrast (~50%)
      if contrast_factor:
        high_resolution = tf.cond(
            tf.less_equal(tf.random.uniform([]), 0.5),
            lambda: tf.image.random_contrast(
                high_resolution, *contrast_factor),
            lambda: high_resolution)

      # Randomly setting saturation(~50%)
      if saturation:
        high_resolution = tf.cond(
            tf.less_equal(tf.random.uniform([]), 0.5),
            lambda: tf.image.random_saturation(
                high_resolution, *saturation),
            lambda: high_resolution)

      return low_res_map_fn(high_resolution)

    # Randomly returning unchanged data (~20%)
    return tf.cond(
        tf.less_equal(tf.random.uniform([]), 0.2),
        lambda: (low_resolution, high_resolution),
        partial(augment_steps_fn, low_resolution, high_resolution))

  return augment_fn


def reform_dataset(dataset, types, size):
  """ Helper function to convert the output_dtype of the dataset
      from (tf.float32, tf.uint8) to desired dtype
      Args:
        dataset: Source dataset(image-label dataset) to convert.
        types: tuple / list of target datatype.
        size: [height, width] threshold of the images.
      Returns:
        tf.data.Dataset with the images of dimension >= Args.size and types = Args.types
  """
  def generator_fn():
    for data in dataset:
      if data[0].shape[0] >= size[0] and data[0].shape[1] >= size[1]:
        yield data[0], data[1]
      else:
        continue
  return tf.data.Dataset.from_generator(
      generator_fn, types, (tf.TensorShape([None, None, 3]), tf.TensorShape(None)))


def load_dataset_directory(
        name,
        directory,
        low_res_map_fn,
        batch_size=None,
        shuffle=False,
        augment=False,
        cache_dir="cache/",
        buffer_size=3 * 32,
        options=None):
  """ Loads image_label dataset from a local directory:
      Structure of the local directory should be:

      dataset_name
      |__ label1
      |   |__ image1
      |   |__ image2
      |
      |__ label2
          |__ image1
          |__ image2

      Args:
          name: Name of the dataset.
          directory: Location where the manual directory is located
          low_res_map_fn: tf.data.Dataset mappable function to generate
                          (low_resolution, high_resolution) pair
          batch_size: Size of batch to create
          shuffle: Boolean to indicate if data is to be shuffled.
          augment: Boolean to indicate if data is to augmented.
          cache_dir: Cache directory to save the data to.
          buffer_size: size of shuffle buffer to use.
      Returns:
          A tf.data.Dataset having data as (low_resolution, high_resoltion)
  """
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
      (tf.float32, tf.float32),
      size=low_res_map_fn.size)
  if options:
    dataset.with_options(options)
  dataset = dataset.map(low_res_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if batch_size:
    dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size)
            #.cache(cache_dir))

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
        batch_size=None,
        shuffle=True,
        augment=True,
        buffer_size=3 * 32,
        cache_dir="cache/",
        data_dir=None,
        options=None):
  """ Helper function to load a dataset from tensorflow_datasets
      Args:
          name: Name of the dataset builder from tensorflow_datasets to load the data.
          low_res_map_fn: tf.data.Dataset mappable function to generate
                          (low_resolution, high_resolution) pair.
          split: split of the dataset to return.
          batch_size: Size of batch to create
          shuffle: Boolean to indicate if data is to be shuffled.
          augment: Boolean to indicate if data is to augmented.
          buffer_size: size of shuffle buffer to use.
          cache_dir: Cache directory to save the data to.
          data_dir: Directory to save the downloaded dataset to.
      Returns:
          A tf.data.Dataset having data as (low_resolution, high_resoltion)

  """
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.mkdir(cache_dir)
  dataset = reform_dataset(
      tfds.load(
          name,
          data_dir=data_dir,
          split=split,
          as_supervised=True),
      (tf.float32, tf.float32),
      size=low_res_map_fn.size)
  if options:
    dataset.with_options(options)
  dataset = dataset.map(low_res_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if batch_size:
    dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size)
             #.cache(cache_dir))

  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

  if augment:
    dataset = dataset.map(
        augment_image(
            partial(low_res_map_fn, no_random_crop=True),
            saturation=None),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset

def load_tfrecord_dataset(tfrecord_path, lr_size, hr_size):
  def _parse_tf_record(serialized_example):
    features = {
        "low_res_image": tf.io.FixedLenFeature([], dtype=tf.string),
        "high_res_image": tf.io.FixedLenFeature([], dtype=tf.string)}
    example = tf.io.parse_single_example(serialized_example, features)
    lr_image = tf.io.parse_tensor(
        example["low_res_image"],
        out_type=tf.float32)
    lr_image = tf.reshape(lr_image, lr_size)
    hr_image = tf.io.parse_tensor(
        example["high_res_image"],
        out_type=tf.float32)
    hr_image = tf.reshape(hr_image, hr_size)
    return lr_image, hr_image
  files = tf.io.gfile.glob(
          os.path.join(tfrecord_path, "*.tfrecord"))
  if len(files) == 0:
    raise ValueError("Path Doesn't contain any file")
  ds = tf.data.TFRecordDataset(files).map(_parse_tf_record)
  if len(files) == 1:
    option = tf.data.Options()
    option.auto_shard = False
    ds.with_options(ds)
  return ds

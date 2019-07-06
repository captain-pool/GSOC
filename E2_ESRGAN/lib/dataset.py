import io
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
        buffer_size=io.DEFAULT_BUFFER_SIZE):

  dl_config = tfds.download.DownloadConfig(manual_dir=directory)
  dataset = (tfds.load("image_label_folder/dataset_name=%s" % name,
                       split="train",
                       as_supervised=True,
                       download_and_prepare_kwargs={"download_config": dl_config})
             .batch(batch_size)
             .repeat(iterations)
             .map(low_res_map_fn))
  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
  return dataset


def load_dataset(
        name,
        low_res_map_fn,
        split="train",
        batch_size=32,
        iterations=1,
        shuffle=True,
        buffer_size=io.DEFAULT_BUFFER_SIZE,
        data_dir=None):

  dataset = (tfds.load(name,
                       data_dir=data_dir,
                       split=split,
                       as_supervised=True)
             .batch(batch_size)
             .repeat(iterations)
             .map(low_res_map_fn))
  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
  return dataset

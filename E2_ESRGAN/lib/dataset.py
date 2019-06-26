import io
import tensorflow as tf
import tensorflow_datasets as tfds
from settings import settings


def scale_down(image, method="bicubic", dimension=1024, factor=4, **kwargs):
  high_resolution = tf.image.resize_with_crop_or_pad(
      image, dimension, dimension)
  low_resolution = tf.image.resize(
      image, [dimension // factor, dimension // factor], method=method)
  return (low_resolution, high_resolution)


def load_dataset(
        name,
        low_res_map_fn,
        splits="train",
        batch_size=32,
        iterations=1,
        shuffle=True,
        buffer_size=io.DEFAULT_BUFFER_SIZE,
        data_dir=None):

  dataset = (tfds.load(name,
                       data_dir=data_dir,
                       splits=splits,
                       as_supervised=True)
             .shuffle(buffer_size,
                      reshuffle_each_iterations=True)
             .batch(batch_size)
             .repeat(iterations)
             .map(low_res_map_fn))
  return dataset

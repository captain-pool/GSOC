# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" TensorFlow Sample for running TPU Training """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import argparse
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tf.enable_v2_behavior()
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

PRETRAINED_KERAS_LAYER = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
BATCH_SIZE = 32  # In case of TPU, Must be a multiple of 8


class SingleDeviceStrategy(object):
  """ Dummy Class to mimic tf.distribute.Strategy for Single Devices """

  def __enter__(self, *args, **kwargs):
    pass

  def __exit__(self, *args, **kwargs):
    pass

  def scope(self):
    return self

  def experimental_distribute_dataset(self, dataset):
    return dataset

  def experimental_run_v2(self, func, args, kwargs):
    return func(*args, **kwargs)

  def reduce(self, reduction_type, distributed_data, axis):  # pylint: disable=unused-argument
    return distributed_data


class Model(tf.keras.layers.Model):
  """ Keras Model class for Image Retraining """

  def __init__(self, num_classes):
    self._pretrained_layer = hub.KerasLayer(
        PRETRAINED_KERAS_LAYER,
        output_shape=[2048],
        trainable=False)
    self._dense_1 = tf.keras.layers.Dense(num_classes, activation="sigmoid")

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              shape=[None, None, 3],
              dtype=tf.float32)])
  def call(self, inputs):
    return self.unsigned_call(inputs)

  def unsigned_call(self, inputs):
    intermediate = self._pretrained_layer(inputs)
    return self._dense_1(intermediate)


def connect_to_tpu(tpu=None):
  if tpu:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu)
    tf.config.experimental_connect_to_host(cluster_resolver.get_master())
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    return strategy, "/task:1"
  return SingleDeviceStrategy(), ""


def load_dataset(name, datadir, batch_size=32, shuffle=None):
  """
    Loads and preprocesses dataset from TensorFlow dataset.
    Args:
      name: Name of the dataset to load
      datadir: Directory to the dataset in.
      batch_size: size of each minibatch. Must be a multiple of 8.
      shuffle: size of shuffle buffer to use. Not shuffled if set to None.
  """
  dataset, info = tfds.load(
      name,
      try_gcs=True,
      data_dir=datadir,
      as_supervised=True,
      with_info=True)
  num_classes = info.features["label"].num_classes

  def _scale_fn(image, label):
    image = tf.cast(image, tf.float32)
    label = tf.cast(image, tf.float32)
    image = image / 127.5
    image -= 1.
    label = tf.one_hot(label, num_classes)
    return image, label

  options = tf.data.Options()
  if not hasattr(tf.data.Options.auto_shard):
    options.experimental_distribute.auto_shard = False
  else:
    options.auto_shard = False

  dataset = (
      dataset.map(
          _scale_fn,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      .with_options(options)
      .batch(batch_size, drop_remainder=True))
  if shuffle:
    dataset = dataset.shuffle(shuffle, reshuffle_each_iteration=True)
  return dataset.repeat(), num_classes


def train_and_export(**kwargs):
  """
    Trains the model and exports as SavedModel.
    Args:
      tpu: Name or GRPC address of the TPU to use.
      logdir: Path to a bucket or directory to store TensorBoard logs.
      modeldir: Path to a bucket or directory to store the model.
      datadir: Path to store the downloaded datasets to.
      dataset: Name of the dataset to load from TensorFlow Datasets.
      num_steps: Number of steps to train the model for.
  """
  if kwargs["tpu"]:
    # For TPU Training the Files must be stored in
    # Cloud Buckets for the TPU to access
    if not kwargs["logdir"].startswith("gs://"):
      raise ValueError("To train on TPU. `logdir` must be cloud bucket")
    if not kwargs["modeldir"].startswith("gs://"):
      raise ValueError("To train on TPU. `modeldir` must be cloud bucket")
    if kwargs["datadir"]:
      if not kwargs["datadir"].startswith("gs://"):
        raise ValueError("To train on TPU. `datadir` must be a cloud bucket")

  os.environ["TFHUB_CACHE_DIR"] = os.path.join(
      kwargs["modeldir"], "tfhub_cache")

  strategy, device = connect_to_tpu(kwargs["tpu"])

  with tf.device(device), strategy.scope():
    summary_writer = tf.summary.create_file_writer(kwargs["logdir"])
    dataset, num_classes = load_dataset(
        kwargs["dataset"],
        kwargs["datadir"],
        shuffle=3 * 32,
        batch_size=BATCH_SIZE)
    dataset = iter(strategy.experimental_distribute_dataset(dataset))
    model = Model(num_classes)
    loss_metric = tf.keras.metrics.Mean()
    optimizer = tf.keras.optimizers.Adam()

    def distributed_step(images, labels):
      with tf.GradientTape() as tape:
        predictions = model.unsigned_call(images)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, predictions)
        loss_metric(loss)
        loss = loss * (1.0 / BATCH_SIZE)
      gradient = tape.gradient(loss, model.trainable_variables)
      train_op = optimizer.apply_gradients(gradient, model.trainable_variables)
      with tf.control_dependencies([train_op]):
        return tf.cast(optimizer.iterations, tf.float32)

    @tf.function
    def train_step(image, label):
      distributed_metric = strategy.experimental_run_v2(
          distributed_step, args=[image, label])
      step = strategy.reduce(
          tf.distribute.ReduceOp.MEAN, distributed_metric, axis=None)
      return step

    while True:
      image, label = next(dataset)
      step = train_step(image, label)
      with summary_writer.as_default():
        tf.summary.scalar(loss_metric.result(), step=optimizer.iterations)
      if step % 100:
        logging.info("Step: #%f\tLoss: %f" % (step, loss_metric.result()))
      if step % kwargs["num_steps"]:
        break

  logging.info("Exporting Saved Model")
  export_path = (kwargs["export_path"]
                 or os.path.join(kwargs["modeldir"], "model"))
  tf.saved_model.save(model, export_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "dataset",
      default=None,
      help="Name of the Dataset to use")
  parser.add_argument(
      "datadir",
      default=None,
      help="Directory to store the downloaded Dataset")
  parser.add_argument(
      "modeldir",
      default=None,
      help="Directory to store the SavedModel to")
  parser.add_argument(
      "logdir",
      default=None,
      help="Directory to store the Tensorboard logs")
  parser.add_argument(
      "tpu",
      default=None,
      help="name or GRPC address of the TPU")
  parser.add_argument(
      "num_steps",
      default=1000,
      type=int,
      help="Number of Steps to train the model for")
  parser.add_argument(
      "export_path",
      default=None,
      help="Explicitly specify the export path of the model."
      "Else `modeldir/model` wil be used.")
  parser.add_argument(
      "--verbose",
      "-v",
      default=0,
      type=int,
      action="count",
      help="increase verbosity. multiple tags to increase more")
  flags, unknown = parser.parse_known_args()
  log_levels = [logging.FATAL, logging.WARNING, logging.INFO, logging.DEBUG]
  log_level = log_levels[min(flags.verbose, len(log_levels) - 1)]
  if not flags.modeldir:
    logging.fatal("`--modeldir` must be specified")
    sys.exit(1)
  logging.set_verbosity(log_level)
  train_and_export(**vars(flags))

import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from absl import app, flags
""" Create a Sample TF-Hub Module using SavedModel v2.0
The module has as a single signature which loads MNIST Dataset from TFDS and train a simple Neural Network for classifying the digits. The model is built and trained using Tewnsorlfow
"""
FLAGS = flags.FLAGS
flags.DEFINE_string("path", "/tmp/tfhub_modules/mnist/digits/1", "Path to export the module")
flags.DEFINE_string("data_dir", None, "Path to Custom TFDS Data Directory")
flags.DEFINE_integer("buffer_size", 1000, "Buffer Size to Use while Shuffling the Dataset")
flags.DEFINE_integer("batch_size", 32, "Size of each batch")
flags.DEFINE_integer("epoch", 10, "Number of iterations")

class MNIST(tf.keras.models.Model):
  def __init__(self, output_activation="softmax"):
    super(MNIST, self).__init__()
    self.layer_1 = tf.keras.layers.Dense(64)
    self.layer_2 = tf.keras.layers.Dense(10, activation=output_activation)

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              shape=[
                  None,
                  28,
                  28,
                  1],
              dtype=tf.uint8)])
  def call(self, inputs):
    casted = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, tf.float32))(inputs)
    flatten = tf.keras.layers.Flatten()(casted)
    normalize = tf.keras.layers.Lambda(
        lambda x: x / tf.reduce_max(tf.gather(x, 0)))(flatten)
    x = self.layer_1(normalize)
    output = self.layer_2(x)
    return output


@tf.function
def train_step(model, loss_fn, optimizer_fn, metric, image, label):
  with tf.GradientTape() as tape:
    preds = model(image)
    label_onehot = tf.one_hot(label, 10)
    loss_ = loss_fn(label_onehot, preds)
  grads = tape.gradient(loss_, model.trainable_variables)
  optimizer_fn.apply_gradients(zip(grads, model.trainable_variables))
  metric(loss_)

def main(_):
  model = MNIST()
  kwargs = {}
  if FLAGS.data_dir:
    kwargs = {"data_dir": FLAGS.data_dir}
  train = tfds.load("mnist", split="train", **kwargs)
  optimizer_fn = tf.optimizers.Adam(learning_rate=1e-3)
  loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  metric = tf.keras.metrics.Mean()
  model.compile(optimizer_fn, loss=loss_fn)
  train = train.shuffle(FLAGS.buffer_size, reshuffle_each_iteration=True).batch(FLAGS.batch_size)
  # Training Loop
  for epoch in range(FLAGS.epoch):
    for step, data in enumerate(train):
      train_step(
          model,
          loss_fn,
          optimizer_fn,
          metric,
          data['image'],
          data['label'])
      sys.stdout.write("\rEpoch: #{}\tStep: #{}\tLoss: {}".format(
            epoch, step, metric.result().numpy()))
  # Exporting Model as SavedModel 2.0
  tf.saved_model.save(model, FLAGS.path)


if __name__ == "__main__":
  app.run(main)

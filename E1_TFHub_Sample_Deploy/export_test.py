import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import unittest
import argparse

FLAGS = None

class TFHubMNISTTest(tf.test.TestCase):
  def setUp(self):
    kwargs = {}
    if FLAGS.data_dir:
      kwargs["data_dir"] = FLAGS.data_dir
    self.dataset = tfds.load("mnist",
                             split="test",
                             **kwargs).shuffle(FLAGS.buffer_size,
                                               reshuffle_each_iteration=True).batch(FLAGS.batch_size)
    self.batch = FLAGS.batch_size
    self.accuracy_flag = FLAGS.check_accuracy
    self.model = hub.load(FLAGS.exported_path)

  def checkModelLoaded(self):
    output_ = self.model.call(
        tf.zeros([self.batch, 28, 28, 1], dtype=tf.uint8).numpy())
    self.assertEqual(output_.shape, [self.batch, 10])

  def checkModelWorking(self):
    sample = next(iter(self.dataset.take(1)))
    prediction = self.model.call(sample['image'])
    self.assertEqual(tf.argmax(prediction).numpy(), sample['label'])

  def checkAccuracy(self):
    accuracy = 0
    for data in self.dataset:
      preds = self.model.call(data['image'])
      preds = tf.argmax(preds, axis=1).numpy()
      accuracy += sum(preds == data['label'].numpy()) / self.batch
    print("Accuracy: %.2f" % (accuracy * 100))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--exported_path", type=str, default="/tmp/tfhub_modules/mnist/digits/1",help="Path to exported Module")
  parser.add_argument("--data_dir", type=str, default=None, help="Path to TFDS Dataset Directory")
  parser.add_argument("--check_accuracy", type=bool, default=False,help="Path to TFDS Dataset Directory")
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size for checking accuracy")
  parser.add_argument("--buffer_size", type=int, default=1000, help="Buffer size used for Data Shuffling")
  FLAGS, unk = parser.parse_known_args()
  tf.test.main()

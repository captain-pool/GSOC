import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib
matplotlib.use('agg')


def main():
  path = input("Path: ")
  module = hub.load(path)
  test_ds = tfds.load("mnist", split=tfds.Split.TEST)
  ax = plt.gca()
  for d in test_ds.take(1):
    test = d
  pred = tf.argmax(tf.gather(module.call(tf.expand_dims(test['image'], 0)), 0))
  ax.set_title("Prediction: " + str(pred.numpy()) + "\n" +
               "Truth: " + str(test['label'].numpy()), loc="center")
  plt.axis('off')
  plt.imshow(test['image'][:, ..., -1])
  print("Saving as: output.jpg  ....  ", end="")
  plt.savefig("output.jpg")
  print("SAVED!")


if __name__ == "__main__":
  main()

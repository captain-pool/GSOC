from functools import partial
import tensorflow as tf
from lib import utils

""" Keras Models for ESRGAN
    Classes:
      RRDBNet: Generator of ESRGAN. (Residual in Residual Network)
      VGGArch: VGG28 Architecture making the Discriminator ESRGAN
"""

class RRDBNet(tf.keras.Model):
  """ Residual in Residual Network consisting of:
      - Convolution Layers
      - Residual in Residual Block as the trunk of the model
      - Pixel Shuffler layers (tf.nn.depth_to_space)
      - Upscaling Convolutional Layers

      Args:
        out_channel: number of channels of the fake output image.
        num_features (default: 32): number of filters to use in the convolutional layers.
        trunk_size (default: 3): number of Residual in Residual Blocks to form the trunk.
        growth_channel (default: 32): number of filters to use in the internal convolutional layers.
        use_bias (default: True): boolean to indicate if bias is to be used in the conv layers.
  """
  def __init__(
          self,
          out_channel,
          num_features=32,
          trunk_size=3,
          growth_channel=32,
          use_bias=True):
    super(RRDBNet, self).__init__()
    self.rrdb_block = partial(utils.RRDB, growth_channel)
    conv = partial(tf.keras.layers.Conv2D, kernel_size=[3, 3],
                   strides=[1, 1],
                   padding="same",
                   use_bias=use_bias)
    self.conv_first = conv(filters=num_features)
    self.rdb_trunk = tf.keras.Sequential(
        [self.rrdb_block() for _ in range(trunk_size)])
    self.conv_trunk = conv(filters=num_features)
    # Upsample
    self.upsample1 = conv(filters=num_features)
    self.upsample2 = conv(filters=num_features)
    self.conv_last_1 = conv(filters=num_features)
    self.conv_last_2 = conv(filters=out_channel)

    self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

  @tf.function(
      input_signature=[
          tf.TensorSpec(shape=[None, None, None, 3],
                        dtype=tf.float32)])
  def call(self, input_):
    feature = self.conv_first(input_)
    trunk = self.conv_trunk(self.rdb_trunk(feature))
    feature = trunk + feature
    feature = self.lrelu(
        self.upsample1(
            tf.nn.depth_to_space(
                feature,
                block_size=2)))
    feature = self.lrelu(
        self.upsample2(
            tf.nn.depth_to_space(
                feature,
                block_size=2)))
    feature = self.lrelu(self.conv_last_1(feature))
    out = self.conv_last_2(feature)
    return out


class VGGArch(tf.keras.Model):
  """ Keras Model for VGG28 Architecture needed to form
      the discriminator of the architecture.
      Args:
        output_shape (default: 1): output_shape of the generator
        num_features (default: 64): number of features to be used in the convolutional layers
                                    a factor of 2**i will be multiplied as per the need
        use

  """
  def __init__(self, output_shape=1, num_features=64, use_bias=True):

    super(VGGArch, self).__init__()
    self.conv = lambda n, s, x: tf.keras.layers.Conv2D(
        n, kernel_size=[3, 3], strides=[s, s], use_bias=use_bias)(x)
    self.num_features = num_features
    self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
    self.batch_norm = lambda x: tf.keras.layers.BatchNormalization()(x)
    self.dense = tf.keras.layers.Dense
    self._output_shape = output_shape

  def call(self, input_):

    features = self.lrelu(self.conv(self.num_features, 1, input_))
    features = self.lrelu(
        self.batch_norm(
            self.conv(
                self.num_features,
                2,
                features)))
    # VGG Trunk
    for i in range(1, 4):
      for j in range(1, 3):
        features = self.lrelu(
            self.batch_norm(
                self.conv(2**i * self.num_features, j, features)))

    flattened = tf.keras.layers.Flatten()(features)
    dense = self.lrelu(self.dense(1024)(flattened))
    out = self.dense(self._output_shape)(dense)
    return out

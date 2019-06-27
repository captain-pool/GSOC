import tensorflow as tf
from lib.utils import *
from functools import partial


class RRDBNet(tf.keras.models.Model):
  def __init__(
          self,
          out_channel,
          num_features=64,
          trunk_size=3,
          growth_channel=32,
          use_bias=True):
    super(RRDBNet, self).__init__()
    self.RRDB_block = partial(RRDB, growth_channel)
    conv = partial(tf.keras.layers.Conv2D, kernel_size=[3, 3],
                   strides=[1, 1],
                   padding="same",
                   use_bias=use_bias)
    self.conv_first = conv(filters=num_features)
    self.RDB_trunk = tf.keras.Sequential(
        [self.RRDB_block() for _ in range(trunk_size)])
    self.conv_trunk = conv(filters=num_features)
    # Upsample
    self.upsample1 = conv(filters=num_features)
    self.upsample2 = conv(filters=num_features)
    self.conv_last_1 = conv(filters=num_features)
    self.conv_last_2 = conv(filters=out_channel)

    self.lrelu = tf.keras.layers.LeakyReLU()

  def call(self, input_):
    feature = self.conv_first(input_)
    trunk = self.conv_trunk(self.RDB_trunk(feature))
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

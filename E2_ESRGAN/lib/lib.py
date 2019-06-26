import tensorflow as tf
from settings import settings


class RDB(tf.keras.layers.Layer):
  """ Residual Dense Block """

  def __init__(self, out_features=32, bias=True):
    super(RDB, self).__init__()
    self.conv = lambda x: tf.keras.layers.Conv2D(
        out_features,
        kernel_size=[3, 3],
        strides=[1, 1], use_bias=bias)(x)
    self.lrelu = tf.keras.layers.LeakyReLU()
    self.beta = settings()["RDB"].get("residual_scale_beta", 0.2)

  def call(self, input_):
    x1 = self.lrelu(self.conv(input_))
    x2 = self.lrelu(self.conv(tf.concat([input_, x1], -1)))
    x3 = self.lrelu(self.conv(tf.concat([input_, x1, x2], -1)))
    x4 = self.lrelu(self.conv(tf.concat([input_, x1, x2, x3], -1)))
    x5 = self.conv5(tf.concat([input_, x1, x2, x3, x4], -1))
    return input_ + self.beta * x5


class RRDB(tf.keras.layers.Layer):
  """ Residual in Residual Block """

  def __init__(self, out_features=32):
    super(RRDB, self).__init__()
    self.RDB1 = RDB(out_features)
    self.RDB2 = RDB(out_features)
    self.RDB3 = RDB(out_features)
    self.beta = settings()["RDB"].get("residual_scale_beta", 0.2)
    
  def call(self, input_):
    out = self.RDB1(input_)
    trunk = input_ + out
    out = self.RDB2(trunk)
    trunk = trunk + out
    out = self.RDB3(trunk)
    trunk = trunk + out
    return input_ + self.beta * trunk

""" Module contains Residual in Residual student network and helper layers. """

from functools import partial
from libs.models import abstract
from libs import settings
import tensorflow as tf


class ResidualDenseBlock(tf.keras.layers.Layer):
  """
    Keras implmentation of Residual Dense Block.
    (https://arxiv.org/pdf/1809.00219.pdf)
  """

  def __init__(self):
    super(ResidualDenseBlock, self).__init__()
    self.settings = settings.Settings(use_student_settings=True)
    rdb_config = self.settings["student_config"]["rrdb_student"]["rdb_config"]
    depthwise_convolution = partial(
        tf.keras.layers.DepthwiseConv2D,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same")
    self._conv_layers = {
        "conv_%d" % index: depthwise_convolution()
        for index in range(1, rdb_config["depth"])}
    self._lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
    self._beta = rdb_config["residual_scale_beta"]

  def call(self, inputs):
    intermediates = [inputs]
    for layer_name in self._conv_layers:
      residual_junction = tf.math.add_n(intermediates)
      raw_intermediate = self._conv_layers[layer_name](residual_junction)
      activated_intermediate = self._lrelu(raw_intermediate)
      intermediates.append(activated_intermediate)
    return inputs + self._beta * raw_intermediate


class ResidualInResidualBlock(tf.keras.layers.Layer):
  """
    Keras implmentation of Residual in Residual Block.
    (https://arxiv.org/pdf/1809.00219.pdf)
  """

  def __init__(self):
    super(ResidualInResidualBlock, self).__init__()
    self.settings = settings.Settings(use_student_settings=True)
    rrdb_config = self.settings["student_config"]["rrdb_student"]["rrdb_config"]
    self._rdb_layers = {
        "rdb_%d" % index: ResidualDenseBlock()
        for index in range(1, rrdb_config["rdb_units"])}
    self._beta = rrdb_config["residual_scale_beta"]

  def call(self, inputs):
    intermediate = inputs
    for layer_name in self._rdb_layers:
      intermediate += self._rdb_layers[layer_name](intermediate)
    return inputs + self._beta * intermediate


class RRDBStudent(abstract.Model):
  """
    Keras implmentation of Residual in Residual Student Network.
    (https://arxiv.org/pdf/1809.00219.pdf)
  """

  def init(self):
    self.settings = settings.Settings(use_student_settings=True)
    rrdb_student_config = self.settings["student_config"]["rrdb_student"]
    rrdb_block = partial(ResidualInResidualBlock)
    depthwise_convolution = partial(
        tf.keras.layers.DepthwiseConv2D,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same")
    convolution = partial(
        tf.keras.layers.Conv2D,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same")
    self.rrdb_trunk = tf.keras.Sequential(
        [rrdb_block() for _ in range(rrdb_student_config["trunk_size"])])
    self.first_conv = depthwise_convolution()
    self._conv_pre_last = convolution(rrdb_student_config["growth_channels"])
    self._upsample1 = depthwise_convolution()
    self._upsample2 = depthwise_convolution()
    self._conv_last = convolution(3)
    self._lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              shape=[None, 180, 270, 3],    # 720x1080 Images
              dtype=tf.float32)])
  def call(self, inputs):
    residual_start = self.first_conv(inputs)
    intermediate = residual_start + self.rrdb_trunk(residual_start)
    intermediate = self._conv_pre_last(intermediate)
    intermediate = self._lrelu(
        self._upsample1(
            tf.nn.depth_to_space(intermediate, 2)))
    intermediate = self._lrelu(
        self._upsample2(
            tf.nn.depth_to_space(intermediate, 2)))
    out = self._conv_last(intermediate)
    return out

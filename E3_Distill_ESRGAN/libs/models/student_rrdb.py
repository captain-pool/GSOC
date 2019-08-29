""" Module contains Residual in Residual student network and helper layers. """

from functools import partial
from libs.models import abstract
from libs import settings
import tensorflow as tf
from absl import logging


class ResidualDenseBlock(tf.keras.layers.Layer):
  """
    Keras implmentation of Residual Dense Block.
    (https://arxiv.org/pdf/1809.00219.pdf)
  """

  def __init__(self, first_call=True):
    super(ResidualDenseBlock, self).__init__()
    self.settings = settings.Settings(use_student_settings=True)
    rdb_config = self.settings["student_config"]["rrdb_student"]["rdb_config"]
    convolution = partial(
        tf.keras.layers.DepthwiseConv2D,
        depthwise_initializer="he_normal",
        bias_initializer="he_normal",
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same")
    self._first_call = first_call
    self._conv_layers = {
        "conv_%d" % index: convolution()
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
    if self._first_call:
      logging.debug("Initializing Layers with 0.1 x MSRA")
      for _, layer in self._conv_layers.items():
        for weight in layer.trainable_variables:
          weight.assign(0.1 * weight)
      self._first_call = False
    return inputs + self._beta * raw_intermediate


class ResidualInResidualBlock(tf.keras.layers.Layer):
  """
    Keras implmentation of Residual in Residual Block.
    (https://arxiv.org/pdf/1809.00219.pdf)
  """

  def __init__(self, first_call=True):
    super(ResidualInResidualBlock, self).__init__()
    self.settings = settings.Settings(use_student_settings=True)
    rrdb_config = self.settings["student_config"]["rrdb_student"]["rrdb_config"]
    self._rdb_layers = {
        "rdb_%d" % index: ResidualDenseBlock(first_call=first_call)
        for index in range(1, rrdb_config["rdb_units"])}
    self._beta = rrdb_config["residual_scale_beta"]

  def call(self, inputs):
    intermediate = inputs
    for layer_name in self._rdb_layers:
      intermediate = self._rdb_layers[layer_name](intermediate)
    return inputs + self._beta * intermediate


class RRDBStudent(abstract.Model):
  """
    Keras implmentation of Residual in Residual Student Network.
    (https://arxiv.org/pdf/1809.00219.pdf)
  """

  def init(self, first_call=True):
    self.settings = settings.Settings(use_student_settings=True)
    self._scale_factor = self.settings["scale_factor"]
    self._scale_value = self.settings["scale_value"]
    rrdb_student_config = self.settings["student_config"]["rrdb_student"]
    rrdb_block = partial(ResidualInResidualBlock, first_call=first_call)
    growth_channels = rrdb_student_config["growth_channels"]
    depthwise_conv = partial(
        tf.keras.layers.DepthwiseConv2D,
        kernel_size=[3, 3],
        strides=[1, 1],
        use_bias=True,
        padding="same")
    convolution = partial(
        tf.keras.layers.Conv2D,
        kernel_size=[3, 3],
        use_bias=True,
        strides=[1, 1],
        padding="same")
    conv_transpose = partial(
        tf.keras.layers.Conv2DTranspose,
        kernel_size=[3, 3],
        use_bias=True,
        strides=self._scale_value,
        padding="same")
    self._rrdb_trunk = tf.keras.Sequential(
        [rrdb_block() for _ in range(rrdb_student_config["trunk_size"])])
    self._first_conv = convolution(filters=64)
    self._upsample_layers = {
        "upsample_%d" % index: conv_transpose(filters=growth_channels)
        for index in range(1, self._scale_factor)}
    key = "upsample_%d" % self._scale_factor
    self._upsample_layers[key] = conv_transpose(filters=3)
    self._conv_last = depthwise_conv()
    self._lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

  def call(self, inputs):
    return self.unsigned_call(inputs)

  def unsigned_call(self, inputs):
    residual_start = self._first_conv(inputs)
    intermediate = residual_start + self._rrdb_trunk(residual_start)
    for layer_name in self._upsample_layers:
      intermediate = self._lrelu(
          self._upsample_layers[layer_name](intermediate))
    out = self._conv_last(intermediate)
    return out

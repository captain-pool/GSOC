""" Module containing ResNet architecture as a student Network """
import tensorflow as tf
from libs.models import abstract
from libs import settings
from functools import partial


class ResidualStudent(abstract.Model):
  """ ResNet Student Network Model """

  def init(self):
    self.student_settings = settings.Settings(use_student_settings=True)
    self._scale_factor = self.student_settings["scale_factor"]
    self._scale_value = self.student_settings["scale_value"]
    model_args = self.student_settings["student_config"]["residual_student"]
    depth = model_args["trunk_depth"]
    depthwise_convolution = partial(
        tf.keras.layers.DepthwiseConv2D,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        use_bias=model_args["use_bias"])
    convolution = partial(
        tf.keras.layers.Conv2D,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding="same",
        use_bias=model_args["use_bias"])
    conv_transpose = partial(
        tf.keras.layers.Conv2DTranspose,
        kernel_size=[3, 3],
        strides=self._scale_value,
        padding="same")
    self._lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
    self._residual_scale = model_args["residual_scale_beta"]
    self._conv_layers = {
        "conv_%d" % index:
        depthwise_convolution() for index in range(1, depth + 1)}
    self._upscale_layers = {
        "upscale_%d" % index:
        conv_transpose(filters=32) for index in range(1, self._scale_factor)}
    self._last_layer = conv_transpose(filters=3)

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              shape=[None, None, None, 3],  # 720x1080 Images
              dtype=tf.float32)])
  def call(self, inputs):
    return self.unsigned_call(inputs)

  def usigned_call(self, inputs):
    intermediate = inputs
    for layer_name in self._conv_layers:
      intermediate += self._lrelu(self._conv_layers[layer_name](intermediate))
    intermediate = inputs + self._residual_scale * intermediate  # Residual Trunk
    # Upsampling
    for layer_name in self._upscale_layers:
      intermediate = self._lrelu(
          self._upscale_layers[layer_name](intermediate))
    return self._last_layer(intermediate)

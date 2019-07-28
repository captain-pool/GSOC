""" Student Network with VGG Architecture """
import tensorflow as tf
from libs import settings
from libs.models import abstract
from functools import partial


class VGGStudent(abstract.Model):
  def init(self):
    sett = settings.Settings(use_student_settings=True)
    model_args = sett["student_config"]["vgg_student"]
    self._scale_factor = sett["scale_factor"]
    self._scale_value = sett["scale_value"]
    depth = model_args["trunk_depth"]  # Minimum 2 for scale factor of 4
    depthwise_convolution = partial(
        tf.keras.layers.DepthwiseConv2D,
        kernel_size=[3, 3],
        padding="same",
        use_bias=model_args["use_bias"])
    conv_transpose = partial(
        tf.keras.layers.Conv2DTranspose,
        kernel_size=[3, 3],
        strides=self._scale_value,
        padding="same")
    convolution = partial(
        tf.keras.layers.Conv2D,
        kernel_size=[3, 3],
        strides=[1, 1],
        use_bias=model_args["use_bias"],
        padding="same")
    trunk_depth = depth - self._scale_factor
    self._conv_layers = {
        "conv_%d" % index: depthwise_convolution()
        for index in range(1, trunk_depth + 1)}

    self._upsample_layers = {
        "upsample_%d" % index: conv_transpose(filters=32)
        for index in range(1, self._scale_factor)}
    self._last_layer = conv_transpose(filters=3)

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              shape=[None, None, None, 3],  # For 720x1080 images
              dtype=tf.float32)])
  def call(self, inputs):
    intermediate = inputs
    for layer_name in self._conv_layers:
      intermediate = self._conv_layers[layer_name](intermediate)
      intermediate = tf.keras.layers.LeakyReLU(alpha=0.2)(intermediate)
    for layer_name in self._upsample_layers:
      intermediate = self._upsample_layers[layer_name](intermediate)
      intermediate = tf.keras.layers.LeakyReLU(alpha=0.2)(intermediate)
    return self._last_layer(intermediate)

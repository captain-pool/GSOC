""" Student Network with VGG Architecture """
import tensorflow as tf
from libs import settings
from libs.models import abstract
from functools import partial


class VGGStudent01(abstract.Model):
  def init(self):
    sett = settings.Settings(student=True)
    model_args = sett["student_config"]["vgg_student01"]
    self.scale_factor = sett["scale_factor"]
    depth = model_args["trunk_depth"]  # Minimum 2 for scale factor of 4
    convolution = partial(
        tf.keras.layers.DepthwiseConv2D,
        kernel_size=[3, 3],
        padding="same",
        use_bias=model_args["use_bias"])
    self._conv_layers = {
        "conv_%d" % index: convolution() for index in range(1, depth + 1)}
    self._last_layer = convolution()

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              shape=[None, 180, 270, 3],  # For 720x1080 images
              dtype=tf.float32)])
  def call(self, inputs):
    intermediate = inputs
    for layer_name in list(self._conv_layers.keys())[:-self.scale_factor]:
      intermediate = self._conv_layers[layer_name](intermediate)

    for layer_name in list(self._conv_layers.keys())[-self.scale_factor:]:
      if not layer_name.endswith("_1"):
        intermediate = tf.keras.layers.LeakyReLU(alpha=0.2)(intermediate)
      pixel_shuffle = tf.nn.depth_to_space(intermediate, self.scale_factor)
      intermediate = self._conv_layers[layer_name](intermediate)

    return self._last_layer(intermediate)

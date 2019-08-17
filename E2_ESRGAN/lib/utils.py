import os
from functools import partial
import tensorflow as tf
from absl import logging
from lib import settings

""" Utility functions needed for training ESRGAN model. """

# Checkpoint Utilities


def save_checkpoint(checkpoint, training_phase, basepath=""):
  """ Saves checkpoint.
      Args:
        checkpoint: tf.train.Checkpoint object
        training_phase: The training phase of the model to load/store the checkpoint for.
                        can be one of the two "phase_1" or "phase_2"
        basepath: Base path to load checkpoints from.
  """
  dir_ = settings.Settings()["checkpoint_path"][training_phase]
  if basepath:
    dir_ = os.path.join(basepath, dir_)
  dir_ = os.path.join(dir_, os.path.basename(dir_))
  checkpoint.save(file_prefix=dir_)
  logging.debug("Prefix: %s. checkpoint saved successfully!" % dir_)


def load_checkpoint(checkpoint, training_phase, basepath=""):
  """ Saves checkpoint.
      Args:
        checkpoint: tf.train.Checkpoint object
        training_phase: The training phase of the model to load/store the checkpoint for.
                        can be one of the two "phase_1" or "phase_2"
        basepath: Base Path to load checkpoints from.
  """
  logging.info("Loading check point for: %s" % training_phase)
  dir_ = settings.Settings()["checkpoint_path"][training_phase]
  if basepath:
    dir_ = os.path.join(basepath, dir_)
  if tf.io.gfile.exists(os.path.join(dir_, "checkpoint")):
    logging.info("Found checkpoint at: %s" % dir_)
    status = checkpoint.restore(tf.train.latest_checkpoint(dir_))
    return status

# Network Interpolation utility


def interpolate_generator(
        generator_fn,
        discriminator,
        alpha,
        dimension,
        factor=4,
        basepath=""):
  """ Interpolates between the weights of the PSNR model and GAN model

       Refer to Section 3.4 of https://arxiv.org/pdf/1809.00219.pdf (Xintao et. al.)

       Args:
         generator_fn: function which returns the keras model the generator used.
         discriminiator: Keras model of the discriminator.
         alpha: interpolation parameter between both the weights of both the models.
         dimension: dimension of the high resolution image
         factor: scale factor of the model
         basepath: Base directory to load checkpoints from.
       Returns:
         Keras model of a generator with weights interpolated between the PSNR and GAN model.
  """
  assert 0 <= alpha <= 1
  size = dimension
  if not tf.nest.is_nested(dimension):
    size = [dimension, dimension]
  logging.debug("Interpolating generator. Alpha: %f" % alpha)
  optimizer = partial(tf.optimizers.Adam)
  gan_generator = generator_fn()
  psnr_generator = generator_fn()
  # building generators
  gan_generator(tf.random.normal(
      [1, size[0] // factor, size[1] // factor, 3]))
  psnr_generator(tf.random.normal(
      [1, size[0] // factor, size[1] // factor, 3]))

  phase_1_ckpt = tf.train.Checkpoint(
      G=psnr_generator, G_optimizer=optimizer())
  phase_2_ckpt = tf.train.Checkpoint(
      G=gan_generator,
      G_optimizer=optimizer(),
      D=discriminator,
      D_optimizer=optimizer())
  load_checkpoint(phase_1_ckpt, "phase_1", basepath)
  load_checkpoint(phase_2_ckpt, "phase_2", basepath)

  # Consuming Checkpoint
  gan_generator(tf.random.normal(
      [1, size[0] // factor, size[1] // factor, 3]))
  psnr_generator(tf.random.normal(
      [1, size[0] // factor, size[1] // factor, 3]))

  for variables_1, variables_2 in zip(
          gan_generator.trainable_variables, psnr_generator.trainable_variables):
    variables_1.assign((1 - alpha) * variables_2 + alpha * variables_1)

  return gan_generator

# Losses
def preprocess_input(image):
  image = image[...,::-1]
  mean = -tf.constant([103.939, 116.779, 123.68])
  return tf.nn.bias_add(image, mean)

def PerceptualLoss(weights=None, input_shape=None, loss_type="L1"):
  """ Perceptual Loss using VGG19
      Args:
        weights: Weights to be loaded.
        input_shape: Shape of input image.
        loss_type: Loss type for features. (L1 / L2)
  """
  vgg_model = tf.keras.applications.vgg19.VGG19(
      input_shape=input_shape, weights=weights, include_top=False)
  for layer in vgg_model.layers:
    layer.trainable = False
  # Removing Activation Function
  vgg_model.get_layer("block5_conv4").activation = lambda x: x
  phi = tf.keras.Model(
      inputs=[vgg_model.input],
      outputs=[
          vgg_model.get_layer("block5_conv4").output])

  def loss(y_true, y_pred):
    if loss_type.lower() == "l1":
      return tf.compat.v1.losses.absolute_difference(phi(y_true), phi(y_pred))

    if loss_type.lower() == "l2":
      return tf.reduce_mean(
          tf.reduce_mean(
              (phi(y_true) - phi(y_pred))**2,
              axis=0))
    raise ValueError(
        "Loss Function: \"%s\" not defined for Perceptual Loss" %
        loss_type)
  return loss


def pixel_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.reduce_mean(tf.abs(y_true - y_pred), axis=0))


def RelativisticAverageLoss(non_transformed_disc, type_="G"):
  """ Relativistic Average Loss based on RaGAN
      Args:
      non_transformed_disc: non activated discriminator Model
      type_: type of loss to Ra loss to produce.
             'G': Relativistic average loss for generator
             'D': Relativistic average loss for discriminator
  """
  loss = None

  def D_Ra(x, y):
    return non_transformed_disc(
        x) - tf.reduce_mean(non_transformed_disc(y))

  def loss_D(y_true, y_pred):
    """
      Relativistic Average Loss for Discriminator
      Args:
        y_true: Real Image
        y_pred: Generated Image
    """
    real_logits = D_Ra(y_true, y_pred)
    fake_logits = D_Ra(y_pred, y_true)
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(real_logits), logits=real_logits))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(fake_logits), logits=fake_logits))
    return real_loss + fake_loss

  def loss_G(y_true, y_pred):
    """
     Relativistic Average Loss for Generator
     Args:
       y_true: Real Image
       y_pred: Generated Image
    """
    real_logits = D_Ra(y_true, y_pred)
    fake_logits = D_Ra(y_pred, y_true)
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(real_logits), logits=real_logits)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(fake_logits), logits=fake_logits)
    return real_loss + fake_loss
  if type_ == "G":
    loss = loss_G
  elif type_ == "D":
    loss = loss_D
  return loss


# Strategy Utils

def assign_to_worker(use_tpu):
  if use_tpu:
    return "/job:worker"
  return ""


class SingleDeviceStrategy(object):
  """ Dummy Strategy when Outside TPU """

  def __enter__(self, *args, **kwargs):
    pass

  def __exit__(self, *args, **kwargs):
    pass

  def experimental_distribute_dataset(self, dataset, *args, **kwargs):
    return dataset

  def experimental_run_v2(self, fn, args, kwargs):
    return fn(*args, **kwargs)

  def reduce(reduction_type, distributed_data, axis):
    return distributed_data

  def scope(self):
    return self


# Model Utils


class RDB(tf.keras.layers.Layer):
  """ Residual Dense Block Layer """

  def __init__(self, out_features=32, bias=True):
    super(RDB, self).__init__()
    _create_conv2d = partial(
        tf.keras.layers.Conv2D,
        out_features,
        kernel_size=[3, 3],
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        strides=[1, 1], padding="same", use_bias=bias)
    self._conv2d_layers = {
        "conv_1": _create_conv2d(),
        "conv_2": _create_conv2d(),
        "conv_3": _create_conv2d(),
        "conv_4": _create_conv2d(),
        "conv_5": _create_conv2d()}
    self._lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
    self._beta = settings.Settings()["RDB"].get("residual_scale_beta", 0.2)
    self._first_call = True
  def call(self, input_):
    x1 = self._lrelu(self._conv2d_layers["conv_1"](input_))
    x2 = self._lrelu(self._conv2d_layers["conv_2"](
        tf.concat([input_, x1], -1)))
    x3 = self._lrelu(self._conv2d_layers["conv_3"](
        tf.concat([input_, x1, x2], -1)))
    x4 = self._lrelu(self._conv2d_layers["conv_4"](
        tf.concat([input_, x1, x2, x3], -1)))
    x5 = self._conv2d_layers["conv_5"](tf.concat([input_, x1, x2, x3, x4], -1))
    if self._first_call:
      logging.debug("Initializing with MSRA")
      for _, layer in self._conv2d_layers.items():
        for variable in layer.trainable_variables:
          variable.assign(0.1 * variable)
      self._first_call = False
    return input_ + self._beta * x5


class RRDB(tf.keras.layers.Layer):
  """ Residual in Residual Block Layer """

  def __init__(self, out_features=32):
    super(RRDB, self).__init__()
    self.RDB1 = RDB(out_features)
    self.RDB2 = RDB(out_features)
    self.RDB3 = RDB(out_features)
    self.beta = settings.Settings()["RDB"].get("residual_scale_beta", 0.2)

  def call(self, input_):
    out = self.RDB1(input_)
    out = self.RDB2(out)
    out = self.RDB3(out)
    return input_ + self.beta * out

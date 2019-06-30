import tensorflow as tf
from lib.settings import settings


def checkpoint(checkpoint, training_phase, load=False, assert_consumed=True):
  """ Saves or Loads checkpoint.
      Args:
        checkpoint: tf.train.Checkpoint object
        training_phase: The training phase of the model to load/store the checkpoint for.
                        can be one of the two "phase_1" or "phase_2"
        load: boolean to specify to load or store checkpoint.
        assert_consumed: boolean to check whether or not all the restored variables are loaded.
  """
  dir_ = settings()["checkpoint_path"][training_phase]
  if not load:
    checkpoint.save(file_prefix=dir_)
  else:
    status = checkpoint.restore(tf.train.latest_checkpoint(dir_))
    if assert_consumed:
      satus.assert_consumed()


def PerceptualLoss(**kwargs):
  """ Perceptual Loss using VGG19
      Args:
        weights: Weights to be loaded.
        input_shape: Shape of input image.
  """
  vgg_model = tf.keras.applications.VGG19(**kwargs, include_top=False)
  for layer in vgg_model.layers:
    layer.trainable = False
  phi = tf.keras.Model(
      inputs=[vgg_model.input],
      outputs=[
          vgg_model.get_layer("block5_conv4").output])

  def loss(y_true, y_pred):
    return tf.compat.v1.losses.absolute_difference(
        phi(y_true), phi(y_pred), reduction="weighted_mean")
  return loss


def pixel_loss(y_true, y_pred):
  return tf.reduce_mean(tf.abs(y_true - y_pred))


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


class RDB(tf.keras.layers.Layer):
  """ Residual Dense Block Layer"""

  def __init__(self, out_features=32, bias=True):
    super(RDB, self).__init__()
    self.conv = lambda x: tf.keras.layers.Conv2D(
        out_features,
        kernel_size=[3, 3],
        strides=[1, 1], padding="same", use_bias=bias)(x)
    self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
    self.beta = settings()["RDB"].get("residual_scale_beta", 0.2)

  def call(self, input_):
    x1 = self.lrelu(self.conv(input_))
    x2 = self.lrelu(self.conv(tf.concat([input_, x1], -1)))
    x3 = self.lrelu(self.conv(tf.concat([input_, x1, x2], -1)))
    x4 = self.lrelu(self.conv(tf.concat([input_, x1, x2, x3], -1)))
    x5 = self.conv(tf.concat([input_, x1, x2, x3, x4], -1))
    return input_ + self.beta * x5


class RRDB(tf.keras.layers.Layer):
  """ Residual in Residual Block Layer"""

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

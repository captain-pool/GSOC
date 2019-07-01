import logging
import itertools
from functools import partial
import tensorflow as tf
from lib import settings, utils, model, dataset


class Model(object):
  def __init__(self, **kwargs):
  """ Training on L1 Loss to warmup the Generator.

  Minimizing the L1 Loss will reduce the Peak Signal to Noise Ratio (PSNR)
  of the generated image from the generator.
  This trained generator is then used to bootstrap the training of the
  GAN, creating better image inputs instead of random noises.
  
  Args:
      data_dir: Path to store the downloaded dataset from tfds.
      summary_writer: SummaryWriter object to write summaries for Tensorboard
"""
  settings = settings.settings()
  self._warmup_num_iter = settings.get("warmup_num_iter", None)
  dataset_args = settings["dataset"]
  self._phase_args = settings["train_psnr"]
  self._dataset = dataset.load_dataset(
      dataset_args["name"],
      dataset.scale_down(
          method=dataset_args["scale_method"],
          dimension=dataset_args["dimension"]),
      batch_size=settings["batch_size"],
      data_dir=kwargs["data_dir"])
  self.G = model.RRDBNet(out_channel=3)
  self._iterations = settings["iterations"]
  self.G_optimizer = tf.optimizers.Adam(
      learning_rate=self._phase_args["adam"]["initial_lr"],
      beta_0=self._phase_args["adam"]["beta_0"],
      beta_1=self._phase_args["adam"]["beta_1"])
  self._checkpoint = tf.train.Checkpoint(
      G=self.G,
      G_optimizer=self.G_optimizer)
  self._summary_writer = kwargs["summary_writer"]

  def train(self):
    """ Train PSNR Model"""

    utils.load_checkpoint(self._checkpoint, "phase_1")
    metric = tf.keras.metrics.Mean()
    previous_loss = float("inf")
    decay_params = self._phase_args["adam"]["decay"]
    decay_step = decay_params["step"]
    decay_factor = decay_params["factor"]
    num_steps = itertools.count(1)
    for epoch in range(self._iterations):
      metric.reset_states()
      for lr, hr in self._dataset:
        step = next(num_steps)
        if step % (decay_step - 1):  # Decay Learning Rate
          self.G_optimizer.learning_rate.assign(
              self.G_optimizer.learning_rate * decay_factor)
        if self._warmup_num_iter and step % self._warmup_num_iter:
          return
        with tf.GradientTape() as tape:
          fake = self.G(lr)
          loss = self.pixel_loss(hr, fake)
        gradient = tape.gradient(fake, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(
            *zip(gradient, self.G.trainable_variables))
        mean_loss = metric(loss)
        with self._summary_writer.as_default():
          tf.summary.scalar("warmup_loss", mean_loss)
        if not step % 100:
          logging.info(
              "[WARMUP] Epoch: %d\tBatch: %d\tGenerator Loss: %f" %
              (epoch, step, mean_loss.numpy()))
          if mean_loss < previous_loss:
            utils.save_checkpoint(self._checkpoint, "phase_1")
          previous_loss = mean_loss

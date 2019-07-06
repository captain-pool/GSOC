import time
from absl import logging
import itertools
from functools import partial
import tensorflow as tf
from lib import utils, dataset

class Trainer(object):
  """ Trainer class for ESRGAN """
  def __init__(self, summary_writer, settings, data_dir=None, manual=False):
    """ Setup the values and variables for Training.
        Args:
          summary_writer: tf.summary.SummaryWriter object to write summaries for Tensorboard
          settings: settings object for fetching data from config files
          data_dir (default: None): path where the data downloaded should be stored / accessed
    """
    self.settings = settings
    self.summary_writer = summary_writer
    self.iterations = self.settings["iterations"]
    dataset_args = self.settings["dataset"]
    if not manual:
      self.dataset = dataset.load_dataset(
          dataset_args["name"],
          dataset.scale_down(
              method=dataset_args["scale_method"],
              dimension=dataset_args["hr_dimension"]),
          batch_size=settings["batch_size"],
          data_dir=data_dir)
    else:
      self.dataset = dataset.load_dataset_directory(
          dataset_args["name"],
          data_dir,
          dataset.scale_down(
              method=dataset_args["scale_method"],
              dimension=dataset_args["hr_dimension"]),
          batch_size=settings["batch_size"])
  def warmup_generator(self, generator):
    """ Training on L1 Loss to warmup the Generator.

    Minimizing the L1 Loss will reduce the Peak Signal to Noise Ratio (PSNR)
    of the generated image from the generator.
    This trained generator is then used to bootstrap the training of the
    GAN, creating better image inputs instead of random noises.
    Args:
      generator: Model Object for the Generator
    """
    # Loading up phase parameters
    warmup_num_iter = self.settings.get("warmup_num_iter", None)
    phase_args = self.settings["train_psnr"]
    decay_params = phase_args["adam"]["decay"]
    decay_step = decay_params["step"]
    decay_factor = decay_params["factor"]

    metric = tf.keras.metrics.Mean()
    num_steps = itertools.count(1)

    # Generator Optimizer
    G_optimizer = tf.optimizers.Adam(
        learning_rate=phase_args["adam"]["initial_lr"],
        beta_1=phase_args["adam"]["beta_1"],
        beta_2=phase_args["adam"]["beta_2"])

    checkpoint = tf.train.Checkpoint(
        G=generator,
        G_optimizer=G_optimizer)

    utils.load_checkpoint(checkpoint, "phase_1")
    previous_loss = float("inf")
    start_time = time.time()
    # Training starts
    for epoch in range(self.iterations):
      metric.reset_states()
      for image_lr, image_hr in self.dataset:
        step = next(num_steps)

        if step % (decay_step - 1):  # Decay Learning Rate
          G_optimizer.learning_rate.assign(
              G_optimizer.learning_rate * decay_factor)

        if warmup_num_iter and step % warmup_num_iter:
          return

        with tf.GradientTape() as tape:
          fake = generator(image_lr)
          loss = utils.pixel_loss(image_hr, fake)
        gradient = tape.gradient(fake, generator.trainable_variables)
        G_optimizer.apply_gradients(
            *zip(gradient, generator.trainable_variables))
        mean_loss = metric(loss)

        with self.summary_writer.as_default():
          tf.summary.scalar("warmup_loss", mean_loss)

        if not step % 100:
          logging.info(
              "[WARMUP] Epoch: {}\tBatch: {}\tGenerator Loss: {}\tTime Taken: {} sec".format(
                  epoch, step // (epoch + 1),
                  mean_loss.numpy(), time.time() - start_time))
          if mean_loss < previous_loss:
            utils.save_checkpoint(checkpoint, "phase_1")
          previous_loss = mean_loss
          start_time = time.time()

  def train_gan(self, generator, discriminator):
    """ Implements Training routine for ESRGAN
        Args:
          generator: Model object for the Generator
          discriminator: Model object for the Discriminator
    """
    phase_args = self.settings["train_combined"]
    decay_args = phase_args["adam"]["decay"]
    decay_factor = decay_args["factor"]
    decay_steps = decay_args["step"]
    lambda_ = phase_args["lambda"]
    eta = phase_args["eta"]
    num_steps = itertools.count(1)

    optimizer = partial(
        tf.optimizers.Adam,
        learning_rate=phase_args["adam"]["initial_lr"],
        beta_1=phase_args["adam"]["beta_1"],
        beta_2=phase_args["adam"]["beta_2"])

    G_optimizer = optimizer()
    D_optimizer = optimizer()

    perceptual_loss = utils.PerceptualLoss(
        weights="imagenet",
        input_shape=[
            self.settings["dataset"]["hr_dimension"],
            self.settings["dataset"]["hr_dimension"]])
    ra_gen = utils.RelativisticAverageLoss(discriminator, type_="G")
    ra_disc = utils.RelativisticAverageLoss(discriminator, type_="D")
    
    # The weights of generator trained during Phase #1
    # is used to initialize or "hot start" the generator
    # for phase #2 of training
    hot_start = tf.train.Checkpoint(G=generator, G_optimizer=G_optimizer)
    utils.load_checkpoint(hot_start, "train_psnr")

    checkpoint = tf.train.Checkpoint(
        G=generator,
        G_optimizer=G_optimizer,
        D=discriminator,
        D_optimizer=D_optimizer)

    utils.load_checkpoint(checkpoint, "train_combined")

    gen_metric = tf.keras.metrics.Mean()
    disc_metric = tf.keras.metrics.Mean()

    for epoch in range(self.iterations):
      # Resetting Metrics
      gen_metric.reset_states()
      disc_metric.reset_states()
      start = time.time()
      for (image_lr, image_hr) in self.dataset:

        step = next(num_steps)
        # Decaying Learning Rate
        for _step in decay_steps.copy():
          if step >= _step:
            decay_steps.pop()
            G_optimizer.learning_rate.assign(
                G_optimizer.learning_rate * decay_factor)
            D_optimizer.learning_rate.assign(
                D_optimizer.learning_rate * decay_factor)

       # Calculating Loss applying gradients
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          fake = generator(image_lr)
          percep_loss = perceptual_loss(image_hr, fake)
          l1_loss = utils.pixel_loss(image_hr, fake)
          loss_RaG = ra_gen(image_hr, fake)
          disc_loss = ra_disc(image_hr, fake)
          gen_loss = percep_loss + lambda_ * loss_RaG + eta * l1_loss
          disc_metric(disc_loss)
          gen_metric(gen_loss)
        disc_grad = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)
        gen_grad = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        D_optimizer.apply_gradients(
            *zip(disc_grad, discriminator.trainable_variables))
        G_optimizer.apply_gradients(
            *zip(gen_grad, generator.trainable_variables))

        # Writing Summary
        with self.summary_writer.as_default():
          tf.summary.scalar("gen_loss", gen_metric)
          tf.summary.scalar("disc_loss", disc_metric)
          tf.summary.image("lr_image", image_lr)
          tf.summary.image("hr_image", fake)
        # Logging and Checkpointing
        if not step % 100:
          logging.info("Epoch: {}\tBatch: {}\tGen Loss: {}\tDisc Loss: {}\t Time Taken: {} sec".format(
              (epoch + 1), step // (epoch + 1),
              gen_metric.result().numpy(),
              disc_metric.result().numpy(), time.time() - start))
          utils.save_checkpoint(checkpoint, "train_combined")
          start = time.time()

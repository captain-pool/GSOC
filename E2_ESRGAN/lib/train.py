import os
import time
import itertools
from functools import partial
from absl import logging
import tensorflow as tf
from lib import utils, dataset


class Trainer(object):
  """ Trainer class for ESRGAN """

  def __init__(self, summary_writer, settings, data_dir=None, manual=False):
    """ Setup the values and variables for Training.
        Args:
          summary_writer: tf.summary.SummaryWriter object to write summaries for Tensorboard.
          settings: settings object for fetching data from config files.
          data_dir (default: None): path where the data downloaded should be stored / accessed.
          manual (default: False): boolean to represent if data_dir is a manual dir.
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
    psnr_metric = tf.keras.metrics.Mean()
    tf.summary.experimental.set_step(tf.Variable(0, dtype=tf.int64))
    # Generator Optimizer
    G_optimizer = tf.optimizers.Adam(
        learning_rate=phase_args["adam"]["initial_lr"],
        beta_1=phase_args["adam"]["beta_1"],
        beta_2=phase_args["adam"]["beta_2"])
    checkpoint = tf.train.Checkpoint(
        G=generator,
        G_optimizer=G_optimizer,
        summary_step=tf.summary.experimental.get_step())

    status = utils.load_checkpoint(checkpoint, "phase_1")
    logging.debug("phase_1 status object: {}".format(status))
    previous_loss = float("inf")
    start_time = time.time()
    # Training starts
    for epoch in range(self.iterations):
      metric.reset_states()
      psnr_metric.reset_states()
      for image_lr, image_hr in self.dataset:
        step = tf.summary.experimental.get_step()
        if warmup_num_iter and step % warmup_num_iter:
          return

        with tf.GradientTape() as tape:
          fake = generator(image_lr)
          loss = utils.pixel_loss(image_hr, fake)
        psnr = psnr_metric(
            tf.reduce_mean(
                tf.image.psnr(
                    fake,
                    image_hr,
                    max_val=256.0)))
        gradient = tape.gradient(loss, generator.trainable_variables)
        G_optimizer.apply_gradients(
            zip(gradient, generator.trainable_variables))
        mean_loss = metric(loss)

        if status:
          status.assert_consumed()
          logging.info(
              "consumed checkpoint for phase_1 successfully")
          status = None

        if step % decay_step:  # Decay Learning Rate
          logging.debug("Learning Rate: %f" % G_optimizer.learning_rate.numpy())
          G_optimizer.learning_rate.assign(
              G_optimizer.learning_rate * decay_factor)
          logging.debug(
                  "Decayed Learning Rate by %f. Current Learning Rate %f" % (
                  decay_factor, G_optimizer.learning_rate.numpy()))
        with self.summary_writer.as_default():
          tf.summary.scalar(
              "warmup_loss", mean_loss, step=step)
          tf.summary.scalar("mean_psnr", psnr, step=step)
          step.assign_add(1)

        if not step % self.settings["print_step"]:
          with self.summary_writer.as_default():
            tf.summary.image("fake_image", tf.cast(tf.clip_by_value(
                fake[:1], 0, 255), tf.uint8), step=step)
            tf.summary.image("hr_image",
                             tf.cast(image_hr[:1], tf.uint8),
                             step=step)

          logging.info(
              "[WARMUP] Epoch: {}\tBatch: {}\tGenerator Loss: {}\tPSNR: {}\tTime Taken: {} sec".format(
                  epoch,
                  step //
                  epoch,
                  mean_loss.numpy(),
                  psnr.numpy(),
                  time.time() -
                  start_time))
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
    tf.summary.experimental.set_step(tf.Variable(0, dtype=tf.int64))
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
            self.settings["dataset"]["hr_dimension"], 3])
    ra_gen = utils.RelativisticAverageLoss(discriminator, type_="G")
    ra_disc = utils.RelativisticAverageLoss(discriminator, type_="D")

    # The weights of generator trained during Phase #1
    # is used to initialize or "hot start" the generator
    # for phase #2 of training
    status = None
    if not tf.io.gfile.exists(
        os.path.join(
            self.settings["checkpoint_path"]["phase_2"],
            "checkpoint")):
      hot_start = tf.train.Checkpoint(
          G=generator,
          G_optimizer=G_optimizer,
          summary_step=tf.summary.experimental.get_step())
      status = utils.load_checkpoint(hot_start, "phase_1")
      # consuming variable from checkpoint
      tf.summary.experimental.get_step()

      tf.summary.experimental.set_step(tf.Variable(1, tf.int64))
    else:
      checkpoint = tf.train.Checkpoint(
          G=generator,
          G_optimizer=G_optimizer,
          D=discriminator,
          D_optimizer=D_optimizer,
          summary_step=tf.summary.experimental.get_step())
      status = utils.load_checkpoint(checkpoint, "phase_2")

    logging.debug("phase status object: {}".format(status))

    gen_metric = tf.keras.metrics.Mean()
    disc_metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()
    for epoch in range(self.iterations):
      # Resetting Metrics
      gen_metric.reset_states()
      disc_metric.reset_states()
      psnr_metric.reset_states()
      start = time.time()
      for (image_lr, image_hr) in self.dataset:
        step = tf.summary.experimental.get_step()

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
        psnr = psnr_metric(
            tf.reduce_mean(
                tf.image.psnr(
                    fake,
                    image_hr,
                    max_val=256.0)))
        disc_grad = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)
        gen_grad = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        D_optimizer.apply_gradients(
            zip(disc_grad, discriminator.trainable_variables))
        G_optimizer.apply_gradients(
            zip(gen_grad, generator.trainable_variables))

        if status:
          status.assert_consumed()
          logging.info("consumed checkpoint successfully!")
          status = None

        # Decaying Learning Rate
        for _step in decay_steps.copy():
          if (step - 1) >= _step:
            decay_steps.pop()
            logging.debug(
                "[Phase 2] Decayed Learing Rate by %f." % decay_factor)
            G_optimizer.learning_rate.assign(
                G_optimizer.learning_rate * decay_factor)
            D_optimizer.learning_rate.assign(
                D_optimizer.learning_rate * decay_factor)

        # Writing Summary
        with self.summary_writer.as_default():
          tf.summary.scalar(
              "gen_loss", gen_metric, step=step)
          tf.summary.scalar(
              "disc_loss", disc_metric, step=step)
          tf.summary.scalar("mean_psnr", psnr, step=step)
          step.assign_add(1)

        # Logging and Checkpointing
        if not step % self.settings["print_step"]:
          with self.summary_writer.as_default():
            tf.summary.image("fake_image", tf.cast(tf.clip_by_value(
                fake[:1], 0, 255), tf.uint8), step=step)
            tf.summary.image("hr_image",
                             tf.cast(image_hr[:1], tf.uint8),
                             step=step)
          logging.info(
              "Epoch: {}\tBatch: {}\tGen Loss: {}\tDisc Loss: {}\tPSNR: {}\tTime Taken: {} sec".format(
                  (epoch + 1), step.numpy() // (epoch + 1),
                  gen_metric.result().numpy(),
                  disc_metric.result().numpy(), psnr.numpy(),
                  time.time() - start))
          utils.save_checkpoint(checkpoint, "train_combined")
          start = time.time()

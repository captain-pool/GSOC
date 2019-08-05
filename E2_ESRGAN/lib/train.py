import os
import time
from functools import partial
from absl import logging
import tensorflow as tf
from lib import utils, dataset


class Trainer(object):
  """ Trainer class for ESRGAN """

  def __init__(
          self,
          summary_writer,
          settings,
          model_dir="",
          data_dir=None,
          manual=False,
          strategy=None):
    """ Setup the values and variables for Training.
        Args:
          summary_writer: tf.summary.SummaryWriter object to write summaries for Tensorboard.
          settings: settings object for fetching data from config files.
          data_dir (default: None): path where the data downloaded should be stored / accessed.
          manual (default: False): boolean to represent if data_dir is a manual dir.
    """
    self.settings = settings
    self.model_dir = model_dir
    self.summary_writer = summary_writer
    self.iterations = self.settings["iterations"]
    self.strategy = strategy
    dataset_args = self.settings["dataset"]
    self.batch_size = self.settings["batch_size"]
    hr_size = tf.convert_to_tensor(
        [dataset_args["hr_dimension"],
        dataset_args["hr_dimension"], 3])

    lr_size = tf.cast(hr_size, tf.float32) * tf.convert_to_tensor([1/4, 1/4, 1], tf.float32)
    lr_size = tf.cast(lr_size, tf.int32)
    if isinstance(strategy, tf.distribute.Strategy):
      self.dataset = dataset.load_tfrecord_dataset(
          tfrecord_path=data_dir,
          lr_size=lr_size,
          hr_size=hr_size).batch(self.batch_size, drop_remainder=True)
      self.dataset = strategy.experimental_distribute_dataset(self.dataset)
    else:
      if not manual:
        self.dataset = dataset.load_dataset(
            dataset_args["name"],
            dataset.scale_down(
                method=dataset_args["scale_method"],
                dimension=dataset_args["hr_dimension"]),
            batch_size=settings["batch_size"],
            data_dir=data_dir,
            augment=True,
            shuffle=True)
      else:
        self.dataset = dataset.load_dataset_directory(
            dataset_args["name"],
            data_dir,
            dataset.scale_down(
                method=dataset_args["scale_method"],
                dimension=dataset_args["hr_dimension"]),
            batch_size=settings["batch_size"],
            augment=True,
            shuffle=True)

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

    status = utils.load_checkpoint(checkpoint, "phase_1", self.model_dir)
    logging.debug("phase_1 status object: {}".format(status))
    previous_loss = float("inf")
    start_time = time.time()
    # Training starts

    def _step_fn(image_lr, image_hr):
      with tf.GradientTape() as tape:
        fake = generator.unsigned_call(image_lr)
        loss = utils.pixel_loss(image_hr, fake) * (1.0 / self.batch_size)
      psnr_metric(
          tf.reduce_mean(
              tf.image.psnr(
                  fake,
                  image_hr,
                  max_val=256.0)))
      gen_vars = list(set(generator.trainable_variables))
      gradient = tape.gradient(loss, gen_vars)
      G_optimizer.apply_gradients(
          zip(gradient, gen_vars))
      mean_loss = metric(loss)

    @tf.function
    def train_step(image_lr, image_hr):
      self.strategy.experimental_run_v2(_step_fn, args=[image_lr, image_hr])

    for epoch in range(1, self.iterations + 1):
      for image_lr, image_hr in self.dataset:
        step = tf.summary.experimental.get_step()
        if warmup_num_iter and step % warmup_num_iter:
          return
        train_step(image_lr, image_hr)
        if status:
          status.assert_consumed()
          logging.info(
              "consumed checkpoint for phase_1 successfully")
          status = None

        if not step % decay_step and step:  # Decay Learning Rate
          logging.debug(
              "Learning Rate: %s" %
              G_optimizer.learning_rate.numpy)
          G_optimizer.learning_rate.assign(
              G_optimizer.learning_rate * decay_factor)
          logging.debug(
              "Decayed Learning Rate by %f. Current Learning Rate %s" % (
                  decay_factor, G_optimizer.learning_rate))
        with self.summary_writer.as_default():
          tf.summary.scalar(
              "warmup_loss", metric.result(), step=step)
          tf.summary.scalar("mean_psnr", psnr_metric.result(), step=step)
          step.assign_add(1)

        if not step % self.settings["print_step"]:
          logging.info(
              "[WARMUP] Epoch: {}\tBatch: {}\tGenerator Loss: {}\tPSNR: {}\tTime Taken: {} sec".format(
                  epoch,
                  step //
                  epoch,
                  metric,
                  psnr_metric,
                  time.time() -
                  start_time))
          if metric.result() < previous_loss:
            utils.save_checkpoint(checkpoint, "phase_1", self.model_dir)
          previous_loss = metric.result()
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
    hr_dimension = self.settings["dataset"]["hr_dimension"]
    eta = phase_args["eta"]
    tf.summary.experimental.set_step(tf.Variable(0, dtype=tf.int64))
    optimizer = partial(
        tf.optimizers.Adam,
        learning_rate=phase_args["adam"]["initial_lr"],
        beta_1=phase_args["adam"]["beta_1"],
        beta_2=phase_args["adam"]["beta_2"])

    G_optimizer = optimizer()
    D_optimizer = optimizer()

    ra_gen = utils.RelativisticAverageLoss(discriminator, type_="G")
    ra_disc = utils.RelativisticAverageLoss(discriminator, type_="D")

    # The weights of generator trained during Phase #1
    # is used to initialize or "hot start" the generator
    # for phase #2 of training
    status = None
    checkpoint = tf.train.Checkpoint(
        G=generator,
        G_optimizer=G_optimizer,
        D=discriminator,
        D_optimizer=D_optimizer,
        summary_step=tf.summary.experimental.get_step())

    if not tf.io.gfile.exists(
        os.path.join(
            self.model_dir,
            os.path.join(
            self.settings["checkpoint_path"]["phase_2"],
            "checkpoint"))):
      hot_start = tf.train.Checkpoint(
          G=generator,
          G_optimizer=G_optimizer,
          summary_step=tf.summary.experimental.get_step())
      status = utils.load_checkpoint(hot_start, "phase_1", self.model_dir)
      # consuming variable from checkpoint
      tf.summary.experimental.get_step()

      tf.summary.experimental.set_step(tf.Variable(0, dtype=tf.int64))
    else:
      status = utils.load_checkpoint(checkpoint, "phase_2", self.model_dir)

    logging.debug("phase status object: {}".format(status))

    gen_metric = tf.keras.metrics.Mean()
    disc_metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()
    perceptual_loss = utils.PerceptualLoss(
        weights="imagenet",
        input_shape=[hr_dimension, hr_dimension, 3],
        loss_type=phase_args["perceptual_loss_type"])

    def _step_fn(image_lr, image_hr):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake = generator.unsigned_call(image_lr)
        percep_loss = perceptual_loss(image_hr, fake)
        l1_loss = utils.pixel_loss(image_hr, fake)
        loss_RaG = ra_gen(image_hr, fake)
        disc_loss = ra_disc(image_hr, fake)
        gen_loss = percep_loss + lambda_ * loss_RaG + eta * l1_loss
        gen_loss = gen_loss * (1.0 / self.batch_size)
        disc_loss = disc_loss * (1.0 / self.batch_size)
        disc_metric(disc_loss)
        gen_metric(gen_loss)
      psnr = psnr_metric(
          tf.reduce_mean(
              tf.image.psnr(
                  fake,
                  image_hr,
                  max_val=256.0)))
      gen_vars = list(set(generator.trainable_variables))
      disc_vars = list(set(discriminator.trainable_variables))
      disc_grad = disc_tape.gradient(
          disc_loss, disc_vars)
      gen_grad = gen_tape.gradient(
          gen_loss, gen_vars)
      G_optimizer.apply_gradients(
          zip(gen_grad, gen_vars))     
      D_optimizer.apply_gradients(
          zip(disc_grad, disc_vars))


    @tf.function
    def train_step(image_lr, image_hr):

      self.strategy.experimental_run_v2(_step_fn, args=(image_lr, image_hr))

    for epoch in range(1, self.iterations + 1):
      # Resetting Metrics
      start = time.time()
      for (image_lr, image_hr) in self.dataset:
        step = tf.summary.experimental.get_step()
        train_step(image_lr, image_hr)
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
              "gen_loss", gen_metric.result(), step=step)
          tf.summary.scalar(
              "disc_loss", disc_metric.result(), step=step)
          tf.summary.scalar("mean_psnr", psnr_metric.result(), step=step)
          step.assign_add(1)

        # Logging and Checkpointing
        if not step % self.settings["print_step"]:
          logging.info(
              "Epoch: {}\tBatch: {}\tGen Loss: {}\tDisc Loss: {}\tPSNR: {}\tTime Taken: {} sec".format(
                  (epoch + 1), step // (epoch + 1),
                  gen_metric.result(),
                  disc_metric.result(), psnr_metric,
                  time.time() - start))
          utils.save_checkpoint(checkpoint, "phase_2", self.model_dir)
          start = time.time()

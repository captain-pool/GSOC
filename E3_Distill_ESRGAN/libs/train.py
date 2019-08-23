""" Trainer class to train student network to compress ESRGAN """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from libs import dataset
from libs import settings
from libs import utils
import tensorflow as tf


class Trainer(object):
  """Trainer Class for Knowledge Distillation of ESRGAN"""

  def __init__(
          self,
          teacher,
          discriminator,
          summary_writer,
          summary_writer_2=None,
          model_dir="",
          data_dir="",
          strategy=None):
    """
      Args:
        teacher: Keras Model of pre-trained teacher generator.
                 (Generator of ESRGAN)
        discriminator: Keras Model of pre-trained teacher discriminator.
                       (Discriminator of ESRGAN)
        summary_writer: tf.summary.SummaryWriter object for writing
                         summary for Tensorboard.
        data_dir: Location of the stored dataset.
        raw_data: Indicate if data_dir contains Raw Data or TFRecords.
        model_dir: Location to store checkpoints and SavedModel directory.
    """
    self.teacher_generator = teacher
    self.teacher_discriminator = discriminator
    self.teacher_settings = settings.Settings(use_student_settings=False)
    self.student_settings = settings.Settings(use_student_settings=True)
    self.model_dir = model_dir
    self.strategy = strategy
    self.train_args = self.student_settings["train"]
    self.batch_size = self.teacher_settings["batch_size"]
    self.hr_size = self.student_settings["hr_size"]
    self.lr_size = tf.unstack(self.hr_size)[:-1]
    self.lr_size.append(tf.gather(self.hr_size, len(self.hr_size) - 1) * 4)
    self.lr_size = tf.stack(self.lr_size) // 4
    self.summary_writer = summary_writer
    self.summary_writer_2 = summary_writer_2
    # Loading TFRecord Dataset
    self.dataset = dataset.load_dataset(
        data_dir,
        lr_size=self.lr_size,
        hr_size=self.hr_size)
    self.dataset = (
        self.dataset.repeat()
        .batch(self.batch_size, drop_remainder=True)
        .prefetch(1024))
    self.dataset = iter(self.strategy.experimental_distribute_dataset(
        self.dataset))
    # Reloading Checkpoint from Phase 2 Training of ESRGAN
    checkpoint = tf.train.Checkpoint(
      G=self.teacher_generator,
      D=self.teacher_discriminator)
    utils.load_checkpoint(
        checkpoint,
        "phase_2",
        basepath=model_dir,
        use_student_settings=False)

  def train_comparative(self, student, export_only=False):
    """
      Trains the student using a comparative loss function (Mean Squared Error)
      based on the output of Teacher.
      Args:
        student: Keras model of the student.
    """
    train_args = self.train_args["comparative"]
    total_steps = train_args["num_steps"]
    decay_rate = train_args["decay_rate"]
    decay_steps = train_args["decay_steps"]
    optimizer = tf.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(
        student_generator=student,
        student_optimizer=optimizer)
    status = utils.load_checkpoint(
        checkpoint,
        "comparative_checkpoint",
        basepath=self.model_dir,
        use_student_settings=True)
    if export_only:
      return
    loss_fn = tf.keras.losses.MeanSquaredError(reduction="none")
    metric_fn = tf.keras.metrics.Mean()
    student_psnr = tf.keras.metrics.Mean()
    teacher_psnr = tf.keras.metrics.Mean()

    def step_fn(image_lr, image_hr):
      """
        Function to be replicated among the worker nodes
        Args:
          image_lr: Distributed Batch of Low Resolution Images
          image_hr: Distributed Batch of High Resolution Images
      """
      with tf.GradientTape() as tape:
        teacher_fake = self.teacher_generator.unsigned_call(image_lr)
        student_fake = student.unsigned_call(image_lr)
        student_fake = tf.clip_by_value(student_fake, 0, 255)
        teacher_fake = tf.clip_by_value(teacher_fake, 0, 255)
        image_hr = tf.clip_by_value(image_hr, 0, 255)
        student_psnr(tf.reduce_mean(tf.image.psnr(student_fake, image_hr, max_val=256.0)))
        teacher_psnr(tf.reduce_mean(tf.image.psnr(teacher_fake, image_hr, max_val=256.0)))
        loss = utils.pixelwise_mse(teacher_fake, student_fake)
        loss = tf.reduce_mean(loss) * (1.0 / self.batch_size)
        metric_fn(loss)
      student_vars = list(set(student.trainable_variables))
      gradient = tape.gradient(loss, student_vars)
      train_op = optimizer.apply_gradients(
          zip(gradient, student_vars))
      with tf.control_dependencies([train_op]):
        return tf.cast(optimizer.iterations, tf.float32)

    @tf.function
    def train_step(image_lr, image_hr):
      """
        In Graph Function to assign trainer function to
        replicate among worker nodes.
        Args:
          image_lr: Distributed batch of Low Resolution Images
          image_hr: Distributed batch of High Resolution Images
      """
      distributed_metric = self.strategy.experimental_run_v2(
          step_fn, args=(image_lr, image_hr))
      mean_metric = self.strategy.reduce(
          tf.distribute.ReduceOp.MEAN, distributed_metric, axis=None)
      return mean_metric
    logging.info("Starting comparative loss training")
    while True:
      image_lr, image_hr = next(self.dataset)
      step = train_step(image_lr, image_hr)
      if step >= total_steps:
        return
      for _step in decay_steps.copy():
        if step >= _step:
          decay_steps.pop(0)
          logging.debug("Reducing Learning Rate by: %f" % decay_rate)
          optimizer.learning_rate.assign(optimizer.learning_rate * decay_rate)
      if status:
        status.assert_consumed()
        logging.info("Checkpoint loaded successfully")
        status = None
      # Writing Summary
      with self.summary_writer.as_default():
        tf.summary.scalar("student_loss", metric_fn.result(), step=optimizer.iterations)
        tf.summary.scalar("mean_psnr", student_psnr.result(), step=optimizer.iterations)
      if self.summary_writer_2:
        with self.summary_writer_2.as_default():
          tf.summary.scalar("mean_psnr", teacher_psnr.result(), step=optimizer.iterations)

      if not step % train_args["print_step"]:
        logging.info("[COMPARATIVE LOSS] Step: %s\tLoss: %s" %
                     (step, metric_fn.result()))
      # Saving Checkpoint
      if not step % train_args["checkpoint_step"]:
        utils.save_checkpoint(
            checkpoint,
            "comparative_checkpoint",
            basepath=self.model_dir,
            use_student_settings=True)

  def train_adversarial(self, student, export_only=False):
    """
      Train the student adversarially using a joint loss between teacher discriminator
      and mean squared error between the output of the student-teacher generator pair.
      Args:
        student: Keras model of the student to train.
    """
    train_args = self.train_args["adversarial"]
    total_steps = train_args["num_steps"]
    decay_steps = train_args["decay_steps"]
    decay_rate = train_args["decay_rate"]

    lambda_ = train_args["lambda"]
    alpha = train_args["alpha"]

    generator_metric = tf.keras.metrics.Mean()
    discriminator_metric = tf.keras.metrics.Mean()
    generator_optimizer = tf.optimizers.Adam(learning_rate=train_args["initial_lr"])
    dummy_optimizer = tf.optimizers.Adam()
    discriminator_optimizer = tf.optimizers.Adam(learning_rate=train_args["initial_lr"])
    checkpoint = tf.train.Checkpoint(
        student_generator=student,
        student_optimizer=generator_optimizer,
        teacher_optimizer=discriminator_optimizer,
        teacher_generator=self.teacher_generator,
        teacher_discriminator=self.teacher_discriminator)

    status = None
    if not utils.checkpoint_exists(
        names="adversarial_checkpoint",
        basepath=self.model_dir,
        use_student_settings=True):
      if export_only:
        raise ValueError("Adversarial checkpoints not found")
    else:
      status = utils.load_checkpoint(
          checkpoint,
          "adversarial_checkpoint",
          basepath=self.model_dir,
        use_student_settings=True)
    if export_only:
      if not status:
        raise ValueError("No Checkpoint Loaded!")
      return
    ra_generator = utils.RelativisticAverageLoss(
        self.teacher_discriminator, type_="G")
    ra_discriminator = utils.RelativisticAverageLoss(
        self.teacher_discriminator, type_="D")
    perceptual_loss = utils.PerceptualLoss(
        weights="imagenet",
        input_shape=self.hr_size,
        loss_type="L2")
    student_psnr = tf.keras.metrics.Mean()
    teacher_psnr = tf.keras.metrics.Mean()

    def step_fn(image_lr, image_hr):
      """
        Function to be replicated among the worker nodes
        Args:
          image_lr: Distributed Batch of Low Resolution Images
          image_hr: Distributed Batch of High Resolution Images
      """
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        teacher_fake = self.teacher_generator.unsigned_call(image_lr)
        logging.debug("Fetched Fake: Teacher")
        teacher_fake = tf.clip_by_value(teacher_fake, 0, 255)
        student_fake = student.unsigned_call(image_lr)
        logging.debug("Fetched Fake: Student")
        student_fake = tf.clip_by_value(student_fake, 0, 255)
        image_hr = tf.clip_by_value(image_hr, 0, 255)
        psnr = tf.image.psnr(student_fake, image_hr, max_val=255.0)
        student_psnr(tf.reduce_mean(psnr))
        psnr = tf.image.psnr(teacher_fake, image_hr, max_val=255.0)
        teacher_psnr(tf.reduce_mean(psnr))
        mse_loss = utils.pixelwise_mse(teacher_fake, student_fake)

        image_lr = utils.preprocess_input(image_lr)
        image_hr = utils.preprocess_input(image_hr)
        student_fake = utils.preprocess_input(student_fake)
        teacher_fake = utils.preprocess_input(teacher_fake)

        student_ra_loss = ra_generator(teacher_fake, student_fake)
        logging.debug("Relativistic Average Loss: Student")
        discriminator_loss = ra_discriminator(teacher_fake, student_fake)
        discriminator_metric(discriminator_loss)
        discriminator_loss = tf.reduce_mean(
            discriminator_loss) * (1.0 / self.batch_size)
        logging.debug("Relativistic Average Loss: Teacher")
        percep_loss = perceptual_loss(image_hr, student_fake)
        generator_loss = lambda_ * percep_loss + alpha * student_ra_loss + (1 - alpha) * mse_loss
        generator_metric(generator_loss)
        logging.debug("Calculated Joint Loss for Generator")
        generator_loss = tf.reduce_mean(
            generator_loss) * (1.0 / self.batch_size)
      generator_gradient = gen_tape.gradient(
          generator_loss, student.trainable_variables)
      logging.debug("calculated gradient: generator")
      discriminator_gradient = disc_tape.gradient(
          discriminator_loss, self.teacher_discriminator.trainable_variables)
      logging.debug("calculated gradient: discriminator")
      generator_op = generator_optimizer.apply_gradients(
          zip(generator_gradient, student.trainable_variables))
      logging.debug("applied generator gradients")
      discriminator_op = discriminator_optimizer.apply_gradients(
          zip(discriminator_gradient, self.teacher_discriminator.trainable_variables))
      logging.debug("applied discriminator gradients")

      with tf.control_dependencies(
              [generator_op, discriminator_op]):
        return tf.cast(discriminator_optimizer.iterations, tf.float32)

    @tf.function
    def train_step(image_lr, image_hr):
      """
        In Graph Function to assign trainer function to
        replicate among worker nodes.
        Args:
          image_lr: Distributed batch of Low Resolution Images
          image_hr: Distributed batch of High Resolution Images
      """
      distributed_metric = self.strategy.experimental_run_v2(
          step_fn,
          args=(image_lr, image_hr))
      mean_metric = self.strategy.reduce(
          tf.distribute.ReduceOp.MEAN,
          distributed_metric, axis=None)
      return mean_metric

    logging.info("Starting Adversarial Training")

    while True:
      image_lr, image_hr = next(self.dataset)
      step = train_step(image_lr, image_hr)
      if status:
        status.assert_consumed()
        logging.info("Loaded Checkpoint successfully!")
        status = None
      if not isinstance(decay_steps, list):
        if not step % decay_steps:
          logging.debug("Decaying Learning Rate by: %s" % decay_rate)
          generator_optimizer.learning_rate.assign(
              generator_optimizer.learning_rate * decay_rate)
          discriminator_optimizer.learning_rate.assign(
              discriminator_optimizer.learning_rate * decay_rate)
      else:
        for decay_step in decay_steps.copy():
          if decay_step <= step:
            decay_steps.pop(0)
            logging.debug("Decaying Learning Rate by: %s" % decay_rate)
            generator_optimizer.learning_rate.assign(
                generator_optimizer.learning_rate * decay_rate)
            discriminator_optimizer.learning_rate.assign(
                discriminator_optimizer.learning_rate * decay_rate)
      # Setting Up Logging
      with self.summary_writer.as_default():
        tf.summary.scalar(
            "student_loss",
            generator_metric.result(),
            step=discriminator_optimizer.iterations)
        tf.summary.scalar(
            "teacher_discriminator_loss",
            discriminator_metric.result(),
            step=discriminator_optimizer.iterations)
        tf.summary.scalar(
            "mean_psnr",
            student_psnr.result(),
            step=discriminator_optimizer.iterations)
      if self.summary_writer_2:
        with self.summary_writer_2.as_default():
          tf.summary.scalar(
              "mean_psnr",
              teacher_psnr.result(),
              step=discriminator_optimizer.iterations)
      if not step % train_args["print_step"]:
        logging.info(
            "[ADVERSARIAL] Step: %s\tStudent Loss: %s\t"
            "Discriminator Loss: %s" %
            (step, generator_metric.result(),
             discriminator_metric.result()))
      # Setting Up Checkpoint
      if not step % train_args["checkpoint_step"]:
        utils.save_checkpoint(
            checkpoint,
            "adversarial_checkpoint",
            basepath=self.model_dir,
            use_student_settings=True)
      if step >= total_steps:
        return

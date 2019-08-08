""" Trainer class to train student network to compress ESRGAN """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
from functools import partial
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
    assert(isinstance(strategy, tf.distribute.Strategy))
    self.teacher_generator = teacher
    self.teacher_discriminator = discriminator
    self.teacher_settings = settings.Settings(use_student_settings=False)
    self.student_settings = settings.Settings(use_student_settings=True)
    self.model_dir = model_dir
    self.strategy = strategy
    self.train_args = self.student_settings["train"]
    self.batch_size = self.teacher_settings["batch_size"]
    self.summary_writer = summary_writer
    self.summary_writer_2 = summary_writer_2
    # Loading TFRecord Dataset
    self.dataset = dataset.load_dataset(
        data_dir,
        lr_size=[64, 64, 3],
        hr_size=[256, 256, 3])
    self.dataset = (
        self.dataset.repeat()
        .batch(self.batch_size, drop_remainder=True)
        .prefetch(1024))
    self.dataset = iter(self.strategy.experimental_distribute_dataset(
        self.dataset))
    # Reloading Checkpoint from Phase 2 Training of ESRGAN
    if not utils.checkpoint_exists(
            names=[
                "comparative_checkpoint",
                "adversarial_checkpoint"],
            basepath=self.model_dir,
            use_student_settings=True):
      checkpoint = tf.train.Checkpoint(
          G=self.teacher_generator,
          D=self.teacher_discriminator)
      utils.load_checkpoint(
          checkpoint,
          "phase_2",
          basepath=model_dir,
          use_student_settings=False)

  def train_comparative(self, student):
    """
      Trains the student using a comparative loss function (Mean Squared Error)
      based on the output of Teacher.
      Args:
        student: Keras model of the student.
    """
    total_steps = self.train_args["num_steps"]
    if not tf.summary.experimental.get_step():
      tf.summary.experimental.set_step(tf.Variable(0, dtype=tf.int64))
    optimizer = tf.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(
        student_generator=student,
        student_optimizer=optimizer,
        summary_step=tf.summary.experimental.get_step())
    status = utils.load_checkpoint(
        checkpoint,
        "comparative_checkpoint",
        basepath=self.model_dir,
        use_student_settings=True)
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
        teacher_fake = tf.clip_by_value(teacher_fake, 0, 255)
        student_fake = student.unsigned_call(image_lr)
        student_fake = tf.clip_by_value(student_fake, 0, 255)
        psnr = tf.image.psnr(student_fake, image_hr, max_val=255.0)
        student_psnr(tf.reduce_mean(psnr))
        psnr = tf.image.psnr(teacher_fake, image_hr, max_val=255.0)
        teacher_psnr(tf.reduce_mean(psnr))
        loss = loss_fn(teacher_fake, student_fake)
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
      step = tf.summary.experimental.get_step()
      num_steps = train_step(image_lr, image_hr)
      if num_steps >= total_steps:
        return
      if status:
        status.assert_consumed()
        logging.info("Checkpoint loaded successfully")
        status = None
      # Writing Summary
      with self.summary_writer.as_default():
        tf.summary.scalar("student_loss", metric_fn.result(), step=step)
        tf.summary.scalar("psnr", student_psnr.result(), step=step)
      if self.summary_writer_2:
        with self.summary_writer_2.as_default():
          tf.summary.scalar("psnr", teacher_psnr.result(), step=step)

      if not step % self.train_args["print_step"]:
        logging.info("[COMPARATIVE LOSS] Step: %s\tLoss: %s" %
                     (num_steps, metric_fn.result()))
      # Saving Checkpoint
      if not step % self.train_args["checkpoint_step"]:
        utils.save_checkpoint(
            checkpoint,
            "comparative_checkpoint",
            basepath=self.model_dir,
            use_student_settings=True)
      step.assign_add(1)

  def train_adversarial(self, student):
    """
      Train the student adversarially using a joint loss between teacher discriminator
      and mean squared error between the output of the student-teacher generator pair.
      Args:
        student: Keras model of the student to train.
    """
    total_steps = self.train_args["num_steps"]
    if not tf.summary.experimental.get_step():
      tf.summary.experimental.set_step(tf.Variable(0, dtype=tf.int64))
    ra_generator = utils.RelativisticAverageLoss(
        self.teacher_discriminator, type_="G")
    ra_discriminator = utils.RelativisticAverageLoss(
        self.teacher_discriminator, type_="D")
    alpha = self.train_args["balance_factor"]
    generator_metric = tf.keras.metrics.Mean()
    discriminator_metric = tf.keras.metrics.Mean()
    generator_optimizer = tf.optimizers.Adam()
    discriminator_optimizer = tf.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(
        student_generator=student,
        student_optimizer=generator_optimizer,
        teacher_optimizer=discriminator_optimizer,
        teacher_generator=self.teacher_generator,
        teacher_discriminator=self.teacher_discriminator,
        summary_step=tf.summary.experimental.get_step())
    status = utils.load_checkpoint(
        checkpoint,
        "adversarial_checkpoint",
        basepath=self.model_dir,
        use_student_settings=True)
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
        student_fake = student.unsigned_call(image_lr)
        student_fake = tf.clip_by_value(student_fake, 0, 255)
        psnr = tf.image.psnr(image_hr, student_fake, max_val=255)
        student_psnr(psnr)
        teacher_fake = self.teacher_generator.unsigned_call(image_lr)
        teacher_fake = tf.clip_by_value(teacher_fake, 0, 255)
        psnr = tf.image.psnr(image_hr, teacher_fake, max_val=255)
        teacher_psnr(psnr)
        student_ra_loss = ra_generator(image_hr, student_fake)
        discriminator_loss = ra_discriminator(image_hr, student_fake)
        discriminator_loss = tf.reduce_mean(
            discriminator_loss) * (1.0 / self.batch_size)
        mse_loss = utils.pixelwise_mse(teacher_fake, student_fake)
        generator_loss = alpha * student_ra_loss + (1 - alpha) * mse_loss
        generator_loss = tf.reduce_mean(
            generator_loss) * (1.0 / self.batch_size)
      student_vars = list(set(student.trainable_variables))
      generator_gradient = gen_tape.gradient(
          generator_loss, student_vars)
      teacher_vars = list(set(self.teacher_discriminator.trainable_variables))
      discriminator_gradient = disc_tape.gradient(
          discriminator_loss, teacher_vars)
      generator_op = generator_optimizer.apply_gradients(
          zip(generator_gradient, student_vars))
      discriminator_op = discriminator_optimizer.apply_gradients(
          zip(discriminator_gradient, teacher_vars))
      generator_metric(generator_loss)
      discriminator_metric(discriminator_loss)
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
      mean_metric = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         distributed_metric, axis=None)
      return mean_metric

    logging.info("Starting Adversarial Training")

    while True:
      image_lr, image_hr = next(self.dataset)
      step = tf.summary.experimental.get_step()
      num_steps = train_step(image_lr, image_hr)
      if num_steps >= total_steps:
        return
      if status:
        status.assert_consumed()
        status = None
      # Setting Up Logging
      with self.summary_writer.as_default():
        tf.summary.scalar(
            "student_loss",
            generator_metric.result(),
            step=step)
        tf.summary.scalar(
            "teacher_discriminator_loss",
            discriminator_metric.result(),
            step=step)
        tf.summary.scalar("psnr", student_psnr.result(), step=step)
      if self.summary_writer_2:
        with self.summary_writer_2.as_default():
          tf.summary.scalar("psnr", teacher_psnr.result(), step=step)
      if not step % self.train_args["print_step"]:
        logging.info(
            "[ADVERSARIAL] Step: %s\tStudent Loss: %s\t"
            "Discriminator Loss: %s" %
            (num_steps, generator_metric.result(),
            discriminator_metric.result()))
      step.assign_add(1)
      # Setting Up Checkpoint
      if not step % self.train_args["checkpoint_step"]:
        utils.save_checkpoint(
            checkpoint,
            "adversarial_checkpoint",
            basepath=self.model_dir,
            use_student_settings=True)

""" Trainer class to train student network to compress ESRGAN """

from __future__ import absoulte_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
from lib import dataset
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
      data_dir="",
      model_dir="",
      raw_data=False):
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
    dataset_args = self.teacher_settings["dataset"]
    self.train_args = self.student_settings["train"]

    if raw_data:
      self.dataset = dataset.load_dataset_directory(
          dataset_args["name"],
          data_dir,
          dataset.scale_down(
              method=dataset_args["scale_method"],
              size=self.student_settings["hr_size"]),
          batch_size=self.teacher_settings["batch_size"])
    else:
      self.dataset = dataset.load_dataset(
          dataset_args["name"],
          dataset.scale_down(
              method=dataset_args["scale_args"],
              size=self.student_settings["hr_size"]),
          batch_size=self.teacher_settings["batch_size"],
          data_dir=data_dir)
    self.summary_writer = summary_writer
    self.summary_writer_2 = summary_writer_2
    # Reloading Checkpoint from Phase 2 Training of ESRGAN
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
    tf.summary.experimental.set_step(tf.Variable(0, tf.int64))
    optimizer = tf.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(
        generator=student,
        optimizer=optimizer,
        summary_step=tf.summary.experimental.get_step())
    logging.info("Starting Training using Comparative Loss")
    status = utils.load_checkpoint(
        checkpoint,
        "mse_checkpoint",
        base_path=self.model_dir,
        use_student_settings=True)
    loss_fn = tf.keras.losses.MeanSquaredError()
    metric_fn = tf.keras.metrics.Mean()
    logging.info("Starting comparative loss training")
    for epoch in range(1, self.train_args["iterations"] + 1):
      metric_fn.reset_states()
      for image_lr, image_hr in self.dataset:
        step = tf.summary.experimental.get_step()
        with tf.GradientTape() as tape:
          teacher_fake = self.teacher_generator(image_lr)
          student_fake = student(image_lr)
          student_psnr = tf.image.psnr(image_hr, student_fake, maxval=255.0)
          teacher_psnr = tf.image.psnr(image_hr, teacher_fake, maxval=255.0)
          loss = loss_fn(teacher_fake, student_fake)
          metric_fn(loss)
        gradient = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(gradient, student.trainable_variables))
        if status:
          status.assert_consumed()
          logging.info("Checkpoint loaded successfully")
          status = None
        # Writing Summary
        with self.summary_writer.as_default():
          tf.summary.scalar("loss", metric_fn.result(), step=step)
          tf.summary.scalar("psnr", student_psnr, step=step)
        if self.summary_writer_2:
          with self.summary_writer_2.as_default():
            tf.summary.scalar("psnr", teacher_psnr, step=step)

        if step % self.train_args["print_step"]:
          with self.summary_writer.as_default():
            tf.summary.image("low_res", tf.cast(
                tf.clip_by_value(image_lr[:1], 0, 255), tf.uint8), step=step)
            tf.summary.image("teacher_fake", tf.cast(
                tf.clip_by_value(teacher_fake[:1], 0, 255), tf.uint8), step=step)
            tf.summary.image("student_fake", tf.cast(
                tf.clip_by_value(student_fake[:1], 0, 255), tf.uint8), step=step)
            tf.summary.image("high_res", tf.cast(
                image_hr[:1], tf.uint8), step=step)
          logging.info("[COMPARATIVE LOSS] Epoch: %d\tBatch: %d\tLoss: %f" %
                       (epoch, step // epoch, metric_fn.result().numpy()))
        # Saving Checkpoint
        if step % self.train_args["checkpoint_step"]:
          utils.save_checkpoint(
              checkpoint,
              "mse_checkpoint",
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
    checkpoint = tf.train.Checkpoint(
        student_generator=student,
        teacher_generator=self.teacher_generator,
        teacher_discriminator=self.teacher_discriminator,
        summary_step=tf.summary.experimental.get_step())
    status = utils.load_checkpoint(
        checkpoint,
        "adversarial_checkpoint",
        basepath=self.model_dir,
        use_student_settings=True)
    if not tf.summary.experimental.get_step():
      tf.summary.experimental.set_step(tf.Variable(0, dtype=tf.int64))

    ra_generator = utils.RelativisticAverageLoss(
        self.teacher_discriminator, type_="G")
    ra_discriminator = utils.RelativisticAverageLoss(
        self.teacher_discriminator, type_="D")
    alpha = self.training_args["balance_factor"]
    generator_optimizer = tf.optimizers.Adam()
    discriminator_optimizer = tf.optimizers.Adam()
    generator_metric = tf.keras.metrics.Mean()
    discriminator_metric = tf.keras.metrics.Mean()

    for epoch in range(1, self.student_settings["iterations"] + 1):
      generator_metric.reset_states()
      discriminator_metric.reset_states()
      for image_lr, image_hr in self.dataset:
        step = tf.summary.expermental.get_step()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          student_fake = student(image_lr)
          student_psnr = tf.image.psnr(image_hr, student_fake, maxval=255)
          teacher_fake = self.teacher_generator(image_lr)
          teacher_psnr = tf.image.psnr(image_hr, teacher_fake, maxval=255)
          student_ra_loss = ra_generator(image_hr, student_fake)
          discriminator_loss = ra_discriminator(image_hr, student_fake)
          discriminator_metric(discriminator_loss)
          mse_loss = tf.keras.losses.mean_squared_error(
              teacher_fake, student_fake)
          generator_loss = alpha * student_ra_loss + (1 - alpha) * mse_loss
          generator_metric(generator_loss)
        generator_gradient = gen_tape.gradient(
            generator_loss, student.trainable_variables)
        discriminator_gradient = disc_tape.gradient(
            discriminator_loss, self.teacher_generator.trainable_variables)
        generator_optimizer.apply_gradients(
            zip(generator_gradient, student.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradient, self.teacher_discriminator.trainable_variables))
        if status:
          status.assert_consumed()
          logging.info("Checkpoint consumed successfully")
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
          tf.summary.scalar("psnr", student_psnr, step=step)
        if self.summary_writer_2:
          with self.summary_writer_2.as_default():
            tf.summary.scalar("psnr", teacher_psnr, step=step)

        if step % self.student_settings["print_step"]:
          with self.summary_writer.as_default():
            tf.summary.image("low_res", tf.cast(
                tf.clip_by_value(image_lr[:1], 0, 255), tf.uint8), step=step)
            tf.summary.image("student_fake", tf.cast(
                tf.clip_by_value(student_fake[:1], 0, 255), tf.uint8), step=step)
            tf.summary.image("teacher_fake", tf.cast(
                tf.clip_by_value(teacher_fake[:1], 0, 255), tf.uint8), step=step)
            tf.summary.image("high_res", tf.cast(
                image_hr[:1], tf.uint8), step=step)
          logging.info(
              "[ADVERSARIAL] Epoch: %d\tBatch: %d\tStudent Loss: %f" %
              (epoch, step // epoch, loss))

        # Setting Up Checkpoint
        if step % self.student_settings["checkpoint_step"]:
          utils.save_checkpoint(
              checkpoint,
              "adversarial_checkpoint",
              basepath=self.modelpath,
              use_student_settings=True)

        step.assign_add(1)

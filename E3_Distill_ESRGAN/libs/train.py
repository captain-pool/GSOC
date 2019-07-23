import os
import sys
sys.path.insert(0, os.path.abspath("../E2_ESRGAN"))

from absl import logging
from lib import dataset
from libs import settings
from libs import utils
import tensorflow as tf


class Trainer(object):
  def __init__(
          self,
          data_dir,
          teacher,
          discriminator,
          summary_writer,
          manual=False,
          model_dir=""):

    self.teacher_generator = teacher
    self.teacher_discriminator = discriminator
    self.teacher_settings = settings.Settings(student=False)
    self.student_settings = settings.Settings(student=True)
    dataset_args = self.teacher_settings["dataset"]
    self.train_args = self.student_settings["train"]

    if manual:
      self.dataset = dataset.load_dataset_directory(
          dataset_args["name"],
          data_dir,
          dataset.scale_down(
              method=dataset_args["scale_method"],
              dimension=dataset_args["hr_dimension"]),
          batch_size=self.teacher_settings["batch_size"])
    else:
      self.dataset = dataset.load_dataset(
          dataset_args["name"],
          dataset.scale_down(
              method=dataset_args["scale_args"],
              dimension=dataset_args["hr_dimension"]),
          batch_size=self.teacher_settings["batch_size"],
          data_dir=data_dir)
    self.summary_writer = summary_writer
    # Reloading Checkpoint from Phase 2 Training of ESRGAN
    checkpoint = tf.train.Checkpoint(
        G=self.teacher_generator,
        D=self.teacher_discriminator)
    utils.load_checkpoint(
        checkpoint,
        "phase_2",
        basepath=model_dir,
        student=False)

  def train_comparative_loss(self, student):

    tf.summary.experimental.set_step(tf.Variable(0, tf.int64))
    optimizer = tf.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(
        generator=student,
        optimizer=optimizer,
        summary_step=tf.summary.experimental.get_step())
    logging.info("Starting Training using Comparative Loss")
    status = None
    if tf.io.gfile.exists(
        os.path.join(
            self.model_dir,
            os.path.join(
                self.student_settings["checkpoint_path"],
                "checkpoint"))):
      logging.info(
          "Found checkpoint at: %s" %
          self.student_settings["checkpoint_path"])
      status = utils.load_checkpoint(
          checkpoint,
          "checkpoint_path",
          base_path=self.model_dir,
          student=True)
    loss_fn = tf.keras.losses.MeanSquaredError()
    metric_fn = tf.keras.losses.Mean()

    for epoch in range(1, self.train_args["iterations"] + 1):
      metric_fn.reset_states()
      for image_lr, _ in self.dataset:
        step = tf.summary.experimental.get_step()
        with tf.GradientTape() as tape:
          teacher_fake = self.teacher_generator(image_lr)
          student_fake = student(image_lr)
          psnr = tf.image.psnr(teacher_fake, student_fake, maxval=255.0)
          loss = loss_fn(teacher_fake, student_fake)
          metric_fn(loss)
        gradient = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(gradient, student.trainable_variables))
        with self.summary_writer.as_default():
          tf.summary.scalar("loss", metric_fn.result(), step=step)
          tf.summary.scalar("psnr", psnr, step=step)
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

          logging.info("Epoch: %d\tBatch: %d\tLoss: %f" %
                       (epoch, step // epoch, metric_fn.result().numpy()))
        if step % self.train_args["checkpoint_step"]:
          utils.save_checkpoint(
              checkpoint,
              "checkpoint_path",
              basepath=self.model_dir,
              student=True)

        tf.summary.experimental.set_step(step.assign_add(1))

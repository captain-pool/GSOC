import logging
import itertools
from functools import partial
import tensorflow as tf
from lib import utils, model, dataset

# TODO (@captain-pool): Merge phase_1 and phase_2 in one file.


def warmup_generator(generator, summary_writer, settings, data_dir=None):
  """ Training on L1 Loss to warmup the Generator.

  Minimizing the L1 Loss will reduce the Peak Signal to Noise Ratio (PSNR)
  of the generated image from the generator.
  This trained generator is then used to bootstrap the training of the
  GAN, creating better image inputs instead of random noises.

  Args:
      generator: Generator Model
      data_dir: Path to store the downloaded dataset from tfds.
      summary_writer: SummaryWriter object to write summaries for Tensorboard
  """
  # Loadng up the settings and parameters
  warmup_num_iter = settings.get("warmup_num_iter", None)
  dataset_args = settings["dataset"]
  phase_args = settings["train_psnr"]
  decay_params = phase_args["adam"]["decay"]
  decay_step = decay_params["step"]
  decay_factor = decay_params["factor"]
  iterations = settings["iterations"]

  metric = tf.keras.metrics.Mean()
  num_steps = itertools.count(1)

  dataset = dataset.load_dataset(
      dataset_args["name"],
      dataset.scale_down(
          method=dataset_args["scale_method"],
          dimension=dataset_args["dimension"]),
      batch_size=settings["batch_size"],
      data_dir=data_dir)
  # Generator Optimizer
  G_optimizer = tf.optimizers.Adam(
      learning_rate=phase_args["adam"]["initial_lr"],
      beta_0=phase_args["adam"]["beta_0"],
      beta_1=phase_args["adam"]["beta_1"])

  checkpoint = tf.train.Checkpoint(
      G=generator,
      G_optimizer=G_optimizer)

  utils.load_checkpoint(checkpoint, "phase_1")
  previous_loss = float("inf")
  # Training starts
  for epoch in range(iterations):
    metric.reset_states()
    for lr, hr in dataset:
      step = next(num_steps)

      if step % (decay_step - 1):  # Decay Learning Rate
        G_optimizer.learning_rate.assign(
            G_optimizer.learning_rate * decay_factor)

      if warmup_num_iter and step % warmup_num_iter:
        return

      with tf.GradientTape() as tape:
        fake = generator(lr)
        loss = pixel_loss(hr, fake)
      gradient = tape.gradient(fake, generator.trainable_variables)
      G_optimizer.apply_gradients(
          *zip(gradient, generator.trainable_variables))
      mean_loss = metric(loss)

      with summary_writer.as_default():
        tf.summary.scalar("warmup_loss", mean_loss)

      if not step % 100:
        logging.info(
            "[WARMUP] Epoch: %d\tBatch: %d\tGenerator Loss: %f" %
            (epoch, step // (epoch + 1), mean_loss.numpy()))
        if mean_loss < previous_loss:
          utils.save_checkpoint(checkpoint, "phase_1")
        previous_loss = mean_loss

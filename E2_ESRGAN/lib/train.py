import time
import logging
import itertools
from functools import partial
import tensorflow as tf
from lib import utils, model, dataset


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
  start_time = time.time()
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
            "[WARMUP] Epoch: %d\tBatch: %d\tGenerator Loss: %f\tTime Taken: %f" %
            (epoch, step // (epoch + 1), mean_loss.numpy(), time.time() - start_time))
        if mean_loss < previous_loss:
          utils.save_checkpoint(checkpoint, "phase_1")
        previous_loss = mean_loss
        start_time = time.time()


def train_gan(G, D, summary_writer, settings, data_dir=None):
  """ Implements ESRGAN
      Args:
        G: Generator model.
        D: Discriminator model with non transformed output.
        summary_writer: tf.summary.SummaryWriter to write summaries for Tensorboard.
        settings: settings object
        data_dir: Path to download the dataset from tfds.
  """
  dataset_args = settings["dataset"]
  phase_args = settings["train_combined"]
  decay_args = phase_args["adam"]["decay"]
  num_steps = itertools.count(1)
  iterations = settings["iterations"]

  decay_factor = decay_args["factor"]
  decay_steps = decay_args["step"]
  lambda_ = phase_args["lambda"]
  eta = phase_args["eta"]

  dataset = dataset.load_dataset(
      dataset_args["name"],
      dataset.scale_down(
          method=dataset_args["scale_method"],
          dimension=dataset_args["dimension"]),
      batch_size=settings["batch_size"],
      data_dir=data_dir)

  optimizer = partial(
      tf.optimizers.Adam,
      learning_rate=phase_args["adam"]["initial_lr"],
      beta_0=phase_args["adam"]["beta_0"],
      beta_1=phase_args["adam"]["beta_1"])

  G_optimizer = optimizer()
  D_optimizer = optimizer()

  perceptual_loss = utils.PerceptualLoss(
      weights="imagenet",
      input_shape=[dataset_args["dimensions"], dataset_args["dimension"]])
  Ra_G = utils.RelativisticAverageLoss(D, type_="G")
  Ra_D = utils.RelativisticAverageLoss(D, type_="D")

  hot_start = tf.train.Checkpoint(G=G, G_optimizer=G_optimizer)
  utils.load_checkpoint(hot_start, "train_psnr")

  checkpoint = tf.train.Checkpoint(
      G=G,
      G_optimizer=G_optimizer,
      D=D,
      D_optimizer=D_optimizer)

  utils.load_checkpoint(checkpoint, "train_combined")

  gen_metric = tf.keras.metrics.Mean()
  disc_metric = tf.keras.metrics.Mean()

  for epoch in range(iterations):
    # Resetting Metrics
    gen_metric.reset_states()
    disc_metric.reset_states()
    start = time.time()
    for (image_lr, image_hr) in dataset:

      step = next(num_steps)
      # Decaying Learning Rate
      for _step in decay_steps.copy():
        if step >= _step:
          decay_step.pop()
          G_optimizer.learning_rate.assign(
              G_optimizer.learning_rate * decay_factor)
          D_optimizer.learning_rate.assign(
              D_optimizer.learning_rate * decay_factor)

     # Calculating Loss applying gradients
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake = G(image_lr)
        L_percep = perceptual_loss(image_hr, fake)
        L1 = utils.pixel_loss(image_hr, fake)
        L_RaG = Ra_G(image_hr, fake)
        disc_loss = Ra_D(image_hr, fake)
        gen_loss = L_percep + lambda_ * L_RaG + eta * L1
        disc_metric(disc_loss)
        gen_metric(gen_loss)
      disc_grad = disc_tape.gradient(disc_loss, D.trainable_variables)
      gen_grad = gen_tape.gradient(gen_loss, G.trainable_variables)
      D_optimizer.apply_gradients(
          *zip(disc_grad, D.trainable_variables))
      G_optimizer.apply_gradients(
          *zip(gen_grad, G.trainable_variables))

      # Writing Summary
      with summary_writer.as_default():
        tf.summary.scalar("gen_loss", gen_metric)
        tf.summary.scalar("disc_loss", disc_metric)
        tf.summary.image("lr_image", image_lr)
        tf.summary.image("hr_image", fake)
      # Logging and Checkpointing
      if not step % 100:
        logging.info("Epoch: %d\tBatch: %d\tGen Loss: %f\tDisc Loss: %f\t Time Taken: %f sec" % (
            (epoch + 1), steps // (epoch + 1),
            gen_metric.result().numpy(),
            disc_metric.result().numpy(), time.time() - start))
        utils.save_checkpoint(checkpoint, "train_combined")
        start = time.time()

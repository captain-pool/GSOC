import logging
from functools import partial
import itertools
import tensorflow as tf
from lib import settings, model, utils, dataset


class Model:
  def __init__(self, **kwargs):
    """ Implements ESRGAN
        Args:
          data_dir: Path to download the dataset from tfds
          summary_writer: tf.summary.SummaryWriter to write summaries for Tensorboard
    """
    sett = settings.settings()
    dataset_args = sett["dataset"]
    self.phase_args = sett["train_combined"]
    self.dataset = dataset.load_dataset(
        dataset_args["name"],
        dataset.scale_down(
            method=dataset_args["scale_method"],
            dimension=dataset_args["dimension"]),
        batch_size=sett["batch_size"],
        data_dir=kwargs["data_dir"])
    self.iterations = sett["iterations"]
    self.G = model.RRDBNet(out_channel=3)
    self.D = model.VGGArch()

    optimizer = partial(
        tf.optimizers.Adam,
        learning_rate=self.phase_args["adam"]["initial_lr"],
        beta_0=self.phase_args["adam"]["beta_0"],
        beta_1=self.phase_args["adam"]["beta_1"])
    self.G_optimizer = optimizer()
    self.D_optimizer = optimizer()
    self.perceptual_loss = utils.PerceptualLoss(
        weights="imagenet",
        input_shape=[dataset_args["dimensions"], dataset_args["dimension"]])
    self.Ra_G = utils.RelativisticAverageLoss(self.D, type_="G")
    self.Ra_D = utils.RelativisticAverageLoss(self.D, type_="D")
    hot_start = tf.train.Checkpoint(G=self.G, G_optimizer=self.G_optimizer)
    utils.checkpoint(hot_start, "train_psnr", load=True)
    self.checkpoint = tf.train.Checkpoint(
        G=self.G,
        G_optimizer=self.G_optimizer,
        D=self.D,
        D_optimizer=self.D_optimizer)
    self.summary_writer = kwargs["summary_writer"]

  def train(self):
    """ Train the ESRGAN Model """
    utils.checkpoint(self.checkpoint, "train_combined", load=True)
    gen_metric = tf.keras.metrics.Mean()
    disc_metric = tf.keras.metrics.Mean()
    num_steps = itertools.count(1)
    decay_args = self.phase_args["adam"]["decay"]
    decay_factor = decay_args["factor"]
    decay_steps = decay_args["step"]
    lambda_ = self.phase_args["lambda"]
    eta = self.phase_args["eta"]
    
    for epoch in range(self.iterations):
      # Resetting Metrics
      gen_metric.reset_states()
      disc_metric.reset_states()

      for (image_lr, image_hr) in self.dataset:
        
        step = next(num_steps)
        
        # Decaying Learning Rate
        for _step in decay_steps.copy():
          if step >= _step:
            decay_step.pop()
            self.G_optimizer.learning_rate.assign(
                self.G_optimizer.learning_rate * decay_factor)
            self.D_optimizer.learning_rate.assign(
                self.D_optimizer.learning_rate * decay_factor)
       
       # Calculating Loss applying gradients
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          fake = self.G(image_lr)
          L_percep = self.perceptual_loss(image_hr, fake)
          L1 = utils.pixel_loss(image_hr, fake)
          L_RaG = self.Ra_G(image_hr, fake)
          disc_loss = self.Ra_D(image_hr, fake)
          gen_loss = L_percep + lambda_ * L_RaG + eta * L1
          disc_metric(disc_loss)
          gen_metric(gen_loss)
        disc_grad = disc_tape.gradient(disc_loss, self.D.trainable_variables)
        gen_grad = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        self.D_optimizer.apply_gradients(
            *zip(disc_grad, self.D.trainable_variables))
        self.G_optimizer.apply_gradients(
            *zip(gen_grad, self.G.trainable_variables))
       
        # Writing Summary
        with self.summary_writer.as_default():
          tf.summary.scalar("gen_loss", gen_metric)
          tf.summary.scalar("disc_loss", disc_metric)
          tf.summary.image("lr_image", image_lr)
          tf.summary.image("hr_image", fake)
        # Logging and Checkpointing
        if not step % 100:
          logging.info("Epoch: %d\tBatch: %d\tGen Loss: %f\tDisc Loss: %f" % (
              (epoch + 1), steps // (epoch + 1),
              gen_metric.result().numpy(), disc_metric.result().numpy()))
          utils.checkpoint(self.checkpoint, "train_combined")

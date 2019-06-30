from functools import partial
import tensorflow as tf
from lib import settings, utils, model, dataset

class Model:
  def __init__(self, **kwargs):
  """ PSNR model training on L1 Loss to warmup the Generator
      Args:
        data_dir: Path to store the downloaded dataset from tfds.
        summary_writer: SummaryWriter object to write summaries for Tensorboard
  """
    settings = settings.settings()
    dataset_args = settings["dataset"]
    self.phase_args = settings["train_psnr"]
    self.dataset = dataset.load_dataset(
        dataset_args["name"],
        dataset.scale_down(
            method=dataset_args["scale_method"],
            dimension=dataset_args["dimension"]),
        batch_size=settings["batch_size"],
        data_dir=kwargs["data_dir"])
    self.G = model.RRDBNet(out_channel=3)
    self.iterations = settings["iterations"]
    self.G_optimizer = tf.optimizers.Adam(
        learning_rate=self.phase_args["adam"]["initial_lr"],
        beta_0=self.phase_args["adam"]["beta_0"],
        beta_1=self.phase_args["adam"]["beta_1"])
    self.checkpoint = tf.train.Checkpoint(
        G=self.G,
        G_optimizer=self.G_optimizer)
    self.summary_writer = kwargs["summary_writer"]

  def train(self):
    utils.checkpoint(self.checkpoint, "phase_1", load=True)
    metric = tf.keras.metrics.Mean()
    previous_loss = float("inf")
    decay_params = self.phase_args["adam"]["decay"]
    decay_step = decay_params["step"]
    decay_factor = decay_params["factor"]
    for epoch in range(self.iterations):
      metric.reset_states()
      for idx, (lr, hr) in enumerate(self.dataset):

        if idx and idx % (decay_step - 1):  # Decay Learning Rate
          self.G_optimizer.learning_rate.assign(
              self.G_optimizer.learning_rate * decay_factor)

        # TODO (@captain-pool): Add Condition for Breaking Out of Warm up
        # for a given step count

        with tf.GradientTape() as tape:
          fake = self.G(lr)
          loss = self.pixel_loss(hr, fake)
        gradient = tape.gradient(fake, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(
            *zip(gradient, self.G.trainable_variables))
        mean_loss = metric(loss)
        with self.summary_writer.as_default():
          tf.summary.scalar("warmup_loss", mean_loss)
        if not idx % 100:
          logging.info(
              "[WARMUP] Epoch: %d\tBatch: %d\tGenerator Loss: %f" %
              (epoch, idx + 1, mean_loss.numpy()))
          if mean_loss < previous_loss:
            utils.checkpoint(self.checkpoint, "phase_1")
          previous_loss = mean_loss

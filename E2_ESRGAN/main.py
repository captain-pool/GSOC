import os
import argparse
import logging
from lib import settings, train, model
import tensorflow as tf


def main(**kwargs):
  sett = settings.Settings(kwargs["config"])
  Stats = settings.Stats(os.path.join(sett.path, "stats.yaml"))
  summary_writer = tf.summary.create_file_writer(kwargs["logdir"])
  generator = model.RDBNet(out_channel=3)
  discriminator = model.VGGArch()
  training = train.Trainer(
      summary_writer=summary_writer,
      settings=sett,
      data_dir=kwargs["data_dir"],
      manual=kwargs["manual"])

  if not Stats["train_step_1"]:
    training.warmup_generator(generator)
    Stats["train_step_1"] = True
  if not Stats["train_step_2"]:
    training.train_gan(generator, discriminator)
    Stats["train_step_2"] = True

  # TODO (@captain-pool): Implement generator saver for SavedModel2.0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--config",
      "config/config.yaml",
      "Path to configuration file.")
  parser.add_argument("--data_dir", None, "Directory to put the Data.")
  parser.add_argument("--manual", False, "data_dir is a manual directory", action="store_true")
  parser.add_argument("--model_dir", None, "Directory to put the model in.")
  parser.add_argument("--log_dir", None, "Directory to story Summaries.")
  parser.add_argument("-v", "--verbose", action="count", default=0)
  FLAGS, unparsed = parser.parse_known_args()
  log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
  log_level = log_levels[min(FLAGS.verbose, len(log_levels) - 1)]
  logging.basicConfig(
      level=log_level,
      format="%(asctime)s: %(levelname)s: %(message)s")
  main(**vars(FLAGS))

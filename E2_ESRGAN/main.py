import os
import argparse
from absl import logging
from lib import settings, train, model
from tensorflow.python.eager import profiler
import tensorflow as tf


def main(**kwargs):
  sett = settings.Settings(kwargs["config"])
  Stats = settings.Stats(os.path.join(sett.path, "stats.yaml"))
  summary_writer = tf.summary.create_file_writer(kwargs["log_dir"])
  profiler.start_profiler_server(6009)
  generator = model.RRDBNet(out_channel=3)
  discriminator = model.VGGArch()
  training = train.Trainer(
      summary_writer=summary_writer,
      settings=sett,
      data_dir=kwargs["data_dir"],
      manual=kwargs["manual"])
  phases = list(map(lambda x: x.strip(), kwargs["phases"].lower().split("_")))
  if not Stats["train_step_1"] and "phase1" in phases:
    training.warmup_generator(generator)
    Stats["train_step_1"] = True
  if not Stats["train_step_2"] and "phase2" in phases:
    training.train_gan(generator, discriminator)
    Stats["train_step_2"] = True

  # TODO (@captain-pool): Implement generator saver for SavedModel2.0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--config",
      default="config/config.yaml",
      help="path to configuration file. (default: %(default)s)")
  parser.add_argument(
      "--data_dir",
      default=None,
      help="directory to put the data. (default: %(default)s)")
  parser.add_argument(
      "--manual",
      default=False,
      help="specify if data_dir is a manual directory. (default: %(default)s)",
      action="store_true")
  parser.add_argument(
      "--model_dir",
      default=None,
      help="directory to put the model in.")
  parser.add_argument(
      "--log_dir",
      default=None,
      help="directory to story summaries for tensorboard.")
  parser.add_argument(
      "--phases",
      default="phase1_phase2",
      help="phases to train for seperated by '_'")
  parser.add_argument(
      "-v",
      "--verbose",
      action="count",
      default=0,
      help="each 'v' increases vebosity of logging.")
  FLAGS, unparsed = parser.parse_known_args()
  log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
  log_level = log_levels[min(FLAGS.verbose, len(log_levels) - 1)]
  logging.set_verbosity(log_level)
  main(**vars(FLAGS))

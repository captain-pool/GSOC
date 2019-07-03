import os
import argparse
import logging
from lib import settings, phase_1, phase_2, model
import tensorflow as tf

# TODO (@captain-pool): Merge phase_1 and phase_2 in one file.


def main(**kwargs):
  sett = settings.settings(kwargs["config"])
  stats = settings.stats(os.path.join(sett.path, "stats.yaml"))
  summary_writer = tf.summary.create_file_writer(kwargs["logdir"])
  G = model.RDBNet(out_channel=3)
  D = model.VGGArch()

  if not stats["train_step_1"]:
    phase_1.warmup_generator(
        generator=G,
        data_dir=kwargs["data_dir"],
        summary_writer=summary_writer,
        settings=sett)
    stats["train_step_1"] = True
  if not stats["train_step_2"]:
    phase_2.train_gan(
        G=G, D=D,
        summary_writer=summary_writer,
        sett=sett,
        data_dir=kwargs["data_dir"])
    stats["train_step_2"] = True

  # TODO (@captain-pool): Implement Generator saver for SavedModel2.0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", "config.yaml", "Path to configuration file.")
  parser.add_argument("--data_dir", None, "Directory to put the Data.")
  parser.add_argument("--model_dir", None, "Directory to put the model in.")
  parser.add_argument("--log_dir", None, "Directory to story Summaries.")
  parser.add_argument("-v", "--verbose", action="count", default=0)
  FLAGS, unparsed = parser.parse_known_args()
  levels = [logging.WARNING, logging.INFO, logging.DEBUG]
  level = levels[min(FLAGS.verbose, len(levels) - 1)]
  logging.basicConfig(
      level=level,
      format="%(asctime)s: %(levelname)s: %(message)s")
  main(**vars(FLAGS))

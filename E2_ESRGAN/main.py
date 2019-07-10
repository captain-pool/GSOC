import os
from functools import partial
import argparse
from absl import logging
from lib import settings, train, model, utils
from tensorflow.python.eager import profiler
import tensorflow as tf

""" Enhanced Super Resolution GAN.
    Citation:
      @article{DBLP:journals/corr/abs-1809-00219,
        author    = {Xintao Wang and
                     Ke Yu and
                     Shixiang Wu and
                     Jinjin Gu and
                     Yihao Liu and
                     Chao Dong and
                     Chen Change Loy and
                     Yu Qiao and
                     Xiaoou Tang},
        title     = {{ESRGAN:} Enhanced Super-Resolution Generative Adversarial Networks},
        journal   = {CoRR},
        volume    = {abs/1809.00219},
        year      = {2018},
        url       = {http://arxiv.org/abs/1809.00219},
        archivePrefix = {arXiv},
        eprint    = {1809.00219},
        timestamp = {Fri, 05 Oct 2018 11:34:52 +0200},
        biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1809-00219},
        bibsource = {dblp computer science bibliography, https://dblp.org}
      }
"""


def main(**kwargs):
  """ Main function for training ESRGAN model and exporting it as a SavedModel2.0
      Args:
        config: path to config yaml file.
        log_dir: directory to store summary for tensorboard.
        data_dir: directory to store / access the dataset.
        manual: boolean to denote if data_dir is a manual directory.
        model_dir: directory to store the model into.
  """
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
    logging.info("starting phase 1")
    training.warmup_generator(generator)
    Stats["train_step_1"] = True
  if not Stats["train_step_2"] and "phase2" in phases:
    logging.info("starting phase 2")
    training.train_gan(generator, discriminator)
    Stats["train_step_2"] = True

  elif Stats["train_step_1"]:
    # Attempting to save "Interpolated" Model as SavedModel2.0
    interpolated_generator = utils.interpolate_generator(
        partial(model.RRDBNet, out_channel=3),
        discriminator,
        sett["interpolation_parameter"],
        sett["dataset"]["hr_dimension"])
    tf.saved_model.save(interpolated_generator, kwargs["model_dir"])


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

from absl import logging
import argparse
from libs.models import teacher
from libs import model
from libs import utils
from libs import settings
import tensorflow as tf
"""
  Compressing GANs using Knowledge Distillation.
  Teacher GAN: ESRGAN (https://github.com/captain-pool/E2_ESRGAN)
  
	Citation:
		@article{DBLP:journals/corr/abs-1902-00159,
			author    = {Angeline Aguinaldo and
									 Ping{-}Yeh Chiang and
									 Alexander Gain and
									 Ameya Patil and
									 Kolten Pearson and
									 Soheil Feizi},
			title     = {Compressing GANs using Knowledge Distillation},
			journal   = {CoRR},
			volume    = {abs/1902.00159},
			year      = {2019},
			url       = {http://arxiv.org/abs/1902.00159},
			archivePrefix = {arXiv},
			eprint    = {1902.00159},
			timestamp = {Tue, 21 May 2019 18:03:39 +0200},
			biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1902-00159},
			bibsource = {dblp computer science bibliography, https://dblp.org}
		}
"""
def main(**kwargs):
  student_settings = settings.Settings("../E2_ESRGAN/config.yaml", student=True)
  teacher_settings = settings.Settings("config/config.yaml", student=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", default=None, help="Path to log directory")
  parser.add_argument("--modeldir", default=None, help="directory to store checkpoints and SavedModel")
  parser.add_argument("--verbose", "-v", action="count", default=0, help="Increases Verbosity. Repeat to increase more")
  FLAGS, unparsed = parser.parse_known_args()
  log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
  log_level = log_levels[min(FLAGS.verbose, len(log_levels)-1)]
  logging.set_verbosity(log_level)
  main(**vars(FLAGS))

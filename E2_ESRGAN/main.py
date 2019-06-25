import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", "config.yaml", "Path to configuration file")
  parser.add_argument("--data_dir", None, "Directory to put the Data")
  parser.add_argument("--model_dir", None, "Directory to put the model in")
  parser.add_argument("--ckpt_dir", "checkpoints/", "Directory to put the checkpoints in")
  FLAGS, unparsed = parser.parse_known_args()

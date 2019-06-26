import yaml


def singleton(cls):
  instances = {}

  def getinstance(*args, **kwargs):
    if cls not in instances:
      instances[cls] = cls(*args, **kwargs)
    return instances[cls]
  return getinstance

@singleton
class settings:
  def __init__(self, filename="config.yaml"):
    with open(filename, "r") as f:
      self.data = yaml.load(f.read())
  def __getitem__(self, index):
    return self.data[index]
  def get(self, index, default=None):
    return self.data.get(index, default)

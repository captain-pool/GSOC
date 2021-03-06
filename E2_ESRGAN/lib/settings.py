import os
import yaml


def singleton(cls):
  instances = {}

  def getinstance(*args, **kwargs):
    if cls not in instances:
      instances[cls] = cls(*args, **kwargs)
    return instances[cls]
  return getinstance


@singleton
class Settings(object):
  def __init__(self, filename="config.yaml"):
    self.__path = os.path.abspath(filename)

  @property
  def path(self):
    return os.path.dirname(self.__path)

  def __getitem__(self, index):
    with open(self.__path, "r") as file_:
      return yaml.load(file_.read(), Loader=yaml.FullLoader)[index]

  def get(self, index, default=None):
    with open(self.__path, "r") as file_:
      return yaml.load(
          file_.read(),
          Loader=yaml.FullLoader).get(
          index,
          default)


class Stats(object):
  def __init__(self, filename="stats.yaml"):
    if os.path.exists(filename):
      with open(filename, "r") as file_:
        self.__data = yaml.load(file_.read(), Loader=yaml.FullLoader)
    else:
      self.__data = {}
    self.file = filename

  def get(self, index, default=None):
    self.__data.get(index, default)

  def __getitem__(self, index):
    return self.__data[index]

  def __setitem__(self, index, data):
    self.__data[index] = data
    with open(self.file, "w") as file_:
      yaml.dump(self.__data, file_, default_flow_style=False)

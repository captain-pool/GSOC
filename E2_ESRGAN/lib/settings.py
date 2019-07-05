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
    with open(filename, "r") as file_:
      self.__data = yaml.load(file_.read())
      self.__path = os.path.abspath(os.path.dirname(filename))

  @property
  def path(self):
    return self.__path

  def __getitem__(self, index):
    return self.__data[index]

  def get(self, index, default=None):
    return self.__data.get(index, default)


class Stats(object):
  def __init__(self, filename="stats.yaml"):
    if os.path.exists(filename):
      with open(filename, "r") as file_:
        self.__data = yaml.load(file_.read())
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

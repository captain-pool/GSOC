import re
from tensorflow.python import keras
class Registry(type):
  models = {}
  def _convert_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
  def __init__(cls, name, bases, attrs):
    if Registry._convert_to_snake(name) != "model":
      Registry.models[Registry._convert_to_snake(cls.__name__)] = cls

# Abstract class for Auto Registration of Kernels
class Model(keras.models.Model, metaclass=Registry):
  def __init__(self, *args, **kwargs):
    super(Model, self).__init__()
    self.init(*args, **kwargs)
  def init(self, *args, **kwargs):
    pass

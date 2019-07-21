from tensorflow.python import keras

class Registry(type):
  models = {}
  def __init__(cls, name, bases, attrs):
    if name.lower() != "models":
      Registry.models[cls.__name__.lower()] = cls

class Models(keras.models.Model, metaclass=Registry):
  def __init__(self, *args, **kwargs):
    super(Models, self).__init__()
    self.init(*args, **kwargs)
  def init(self, *args, **kwargs):
    pass

import tensorflow as tf
from lib import settings, utils, model, dataset


class Phase1:
  def __init__(self, **kwargs):
    dataset_args = kwargs["dataset"]
    self.dataset = dataset.load_dataset(
        dataset_args["name"],
        dataset.scale_down(
            method=dataset_args["scale_method"],
            dimension=dataset_args["dimension"]),
        batch_size=kwargs["batch_size"],
        iterations=kwargs["iterations"])
    self.G = model.RRDBNet(out_channel=3)

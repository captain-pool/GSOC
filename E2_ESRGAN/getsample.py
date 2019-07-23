import os
import shutil
try:
  shutil.rmtree("cache/")
except BaseException:
  pass
from PIL import Image
import tensorflow as tf
from lib import model, dataset, utils, settings
settings.Settings("config/config.yaml")
model = model.RRDBNet(out_channel=3)
ds = dataset.load_dataset_directory("coco",
                                    os.path.expanduser("~/datadir"),
                                    dataset.scale_down(dimension=256),
                                    augment=False,
                                    shuffle=False,
                                    batch_size=1)
for lr, hr in ds:
  break


def save(tensor, name): return Image.fromarray(
    tf.cast(tensor, tf.uint8)[0].numpy()).save(name)


save(lr, os.path.expanduser("~/esrgan_samples/low_res.jpg"))
save(hr, os.path.expanduser("~/esrgan_samples/high_res.jpg"))
opt = tf.optimizers.Adam()
ckpt = tf.train.Checkpoint(G=model, G_optimizer=opt)
utils.load_checkpoint(ckpt, "phase_1")
# consuming
# print("consuming...")
#model(tf.expand_dims(lr[0], 0))
print("fake...")
fake = model(tf.expand_dims(lr[0], 0))
save(fake, os.path.expanduser("~/esrgan_samples/fake.jpg"))

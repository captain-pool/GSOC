from absl import logging
import tensorflow as tf
from libs import settings
from libs import model
from libs import utils


def export_tflite(config="", modeldir="", mode="", **kwargs):
  i = input("Don't forget to change the input_signature of the model.
            press q to quit or return to continue")
  if i.lower() is "q":
    return
  status = None
  sett = settings.Settings(config, use_student_settings=True)
  stats = settings.Stats(os.path.join(sett.path, "stats.yaml"))
  student_name = sett["student_network"]
  student_generator = model.Registry.models[student_name]()
  ckpt = tf.train.Checkpoint(student_generator=student_generator)
  if stats.get(mode, None):
    status = utils.load_checkpoint(
        ckpt,
        "%s_checkpoint" % mode,
        basepath=modeldir,
        use_student_settings=True)
  if not status:
    return
  saved_model_dir = os.path.join(modeldir, "signed_compressed_esrgan")
  tf.saved_model.save(
      student_generator,
      saved_model_dir)
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.target_spec.supported_ops = [
      tf.lite.OpSet.TFLITE_BUILTINS,
      tf.lite.OpSet.SELECT_TF_OPS]
  tflite_model = converter.convert()
  with tf.io.gfile.open(
      os.path.join(
          modeldir,
          "tflite",
          "compressed_esrgan.tflite"), "wb") as f:
    f.write(tflite_model)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--modeldir",
      default="",
      help="Directory of the saved checkpoints")
  parser.add_argument(
      "--config",
      default="config/config.yaml",
      help="Configuration File to be loaded")
  parser.add_argument(
      "--mode",
      default="none",
      help="mode of training to load (adversarial /  comparative")
  parser.add_argument(
      "--verbose",
      "-v",
      default=0,
      action="count",
      help="Increases Verbosity. Repeat to increase more")
  log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
  log_level = log_levels[min(FLAGS.verbose, len(log_levels) - 1)]
  logging.set_verbosity(log_level)
  FLAGS, unknown = parser.parse_known_args()
  export_tflite(**vars(FLAGS))

from absl import logging
import os
from lib import dataset
from libs import settings
import tensorflow as tf


def generate_tf_record(
        dataset_name,
        data_dir,
        raw_data=False,
        tfrecord_path="serialized_dataset",
        num_shards=8):

  teacher_sett = settings.Settings(use_student_settings=False)
  student_sett = settings.Settings(use_student_settings=True)
  dataset_args = teacher_sett["dataset"]
  if raw_data:
    ds = dataset.load_dataset_directory(
        dataset_args["name"],
        data_dir,
        dataset.scale_down(
            method=dataset_args["scale_method"],
            size=student_sett["hr_size"]))
  else:
    ds = dataset.load_dataset(
        dataset_args["name"],
        dataset.scale_down(
            method=dataset_args["scale_method"],
            size=student_sett["hr_size"]),
        data_dir=data_dir)
  to_tfrecord(ds, tfrecord_path, num_shards)


def load_dataset(tfrecord_path, lr_size, hr_size):
  def _parse_tf_record(serialized_example):
    features = {
        "low_res_image": tf.io.FixedLenFeature([], dtype=tf.string),
        "high_res_image": tf.io.FixedLenFeature([], dtype=tf.string)}
    example = tf.io.parse_single_example(serialized_example, features)
    lr_image = tf.io.parse_tensor(
        example["low_res_image"],
        out_type=tf.float32)
    lr_image = tf.reshape(lr_image, lr_size) 
    hr_image = tf.io.parse_tensor(
        example["high_res_image"],
        out_type=tf.float32)
    hr_image = tf.reshape(hr_image, hr_size)
    return lr_image, hr_image
  files = tf.io.gfile.glob(
          os.path.join(tfrecord_path, "*.tfrecord"))
  if len(files) == 0:
    raise ValueError("Path Doesn't contain any file")
  ds = tf.data.TFRecordDataset(files).map(_parse_tf_record)
  if len(files) == 1:
    option = tf.data.Options()
    option.auto_shard = False
    ds.with_options(ds)
  ds = ds.shuffle(128)
  return ds


def to_tfrecord(ds, tfrecord_path, NUM_SHARDS=8):
  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def serialize_to_string(image_lr, image_hr):
    features = {
        "low_res_image": _bytes_feature(
            tf.io.serialize_tensor(image_lr).numpy()),
        "high_res_image": _bytes_feature(
            tf.io.serialize_tensor(image_hr).numpy())}
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()

  def write_to_tfrecord(shard_id, ds):
    filename = tf.strings.join(
            [tfrecord_path, "/dataset.", tf.strings.as_string(shard_id),
                ".tfrecord"])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(ds.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)

  def map_serialize_to_string(image_lr, image_hr):
    map_fn = tf.py_function(
        serialize_to_string,
        (image_lr, image_hr),
        tf.string)
    return tf.reshape(map_fn, ())
  ds = ds.map(map_serialize_to_string)
  ds = ds.enumerate()
  ds = ds.apply(tf.data.experimental.group_by_window(
      lambda i, _: i % NUM_SHARDS,
      write_to_tfrecord,
      tf.int64.max))
  for data in ds:
    logging.info("Written to: %s" % data.numpy())

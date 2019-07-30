import os
from lib import dataset
from libs import settings
def generate_tf_record(
    dataset_name,
    data_dir,
    raw_data=False,
    tfrecord_path="serialized_dataset"):

  teacher_sett = settings.Settings(use_student_settings=False)
  student_sett = settings.Settings(use_student_settings=True)
  dataset_args = teacher_sett["dataset"]
  if raw_data:
    dataset = dataset.load_dataset_directory(
    		dataset_args["name"],
    		data_dir,
    		dataset.scale_down(
    				method=dataset_args["scale_method"],
    				size=student_sett["hr_size"]),
    		batch_size=teacher_sett["batch_size"])
  else:
    dataset = dataset.load_dataset(
        dataset_args["name"],
        dataset.scale_down(
            method=dataset_args["scale_method"],
            size=student_sett["hr_size"]),
        batch_size=teacher_sett["batch_size"],
        data_dir=data_dir)
  to_tfrecord(dataset, tfrecord_path)

def load_dataset(tfrecord_path):
  def _parse_tf_record(serialized_example):
    features = {
        "low_res_image": tf.io.VarLenFeature(dtype=tf.float32),
        "high_res_image": tf.io.VarLenFeature(dtype=tf.float32)}
    example = tf.io.parse_single_example(serialized_example, features)
    lr_image = example["low_res_image"]
    hr_image = example["high_res_image"]
    return lr_image, hr_image
  files = tf.convert_to_tensor(tf.io.gfile.glob("*.tfrecord"))
  dataset = tf.dataset.TFRecordDataset(files).map(_parse_tf_record)
  return dataset
  
def to_tfrecord(dataset, tfrecord_path, NUM_SHARDS=8):
  def _bytes_feature(value):
    return tf.train.Feature(byte_list=tf.train.BytesList([value]))
  def serialize_to_string(image_lr, image_hr):
    features = {
        "low_res_image": _bytes_feature(image_lr.numpy()),
        "high_res_image": _bytes_feature(image_hr.numpy())}
    example_proto = tf.train.Example(
        features=tf.train.Features(features))
    return example_proto.SerializeToString()
  def write_to_tfrecord(shard_id, dataset):
    filename = tf.convert_to_tensor(
        os.path.join(
            tfrecord_path,"dataset.%05d.tfrecord" % shard_id))
    with tf.dataset.experimental.TFRecordWriter(filename) as writer:
      writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)
  
  dataset = dataset.map(serialize_to_string).enumerate()
  dataset = dataset.apply(tf.data.experimental.group_by_window(
      lambda i, _: i % NUM_SHARDS,
      write_to_tfrecord,
      tf.int64.max))

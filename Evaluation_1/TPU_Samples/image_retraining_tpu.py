import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.python.tpu import tpu_estimator
from tensorflow.python.tpu import tpu_optimizer
from tensorflow.python.tpu import tpu_config
from functools import partial
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", None, "TPU Address")
flags.DEFINE_integer("iterations",2, "Number of Itertions")
flags.DEFINE_integer("batch_size", 10, "Size of eahc Batch")
flags.DEFINE_boolean("use_tpu", True, " Use TPU")
flags.DEFINE_string("model_dir", "model_dir/", "Directory to Save the Models and Checkpoint")
flags.DEFINE_string("dataset",
            "horses_or_humans",
            "TFDS Dataset Name. IMAGE Dimension should be >= 224, channel=3")
flags.DEFINE_string("data_dir", None, "Directory to Save Data to")
NUM_CLASSES = None
FEATURE_SHAPE = None
def input_(mode, batch_size, iterations, **kwargs):
    global NUM_CLASSES, FEATURE_SHAPE
    dataset, info = tfds.load(
        kwargs["dataset"],
        as_supervised=True,
        split="train" if mode == tf.estimator.ModeKeys.TRAIN else "test",
        with_info=True,
        data_dir=kwargs['data_dir']
        )
    NUM_CLASSES = info.features['label'].num_classes
    FEATURE_SHAPE = info.features['image'].shape
    def resize_and_scale(image, label):
        image = tf.image.resize(image, size=[224, 224])
        image = tf.cast(image, tf.float32)
        image = image / tf.reduce_max(tf.gather(image, 0))
#        label = tf.one_hot(label, info.features['label'].num_classes)
        return image, label

    dataset = dataset.map(resize_and_scale).shuffle(
        1000).repeat(iterations).batch(batch_size, drop_remainder=True)
    return dataset

def serving_reciever_input_fn():
    global FEATURE_SHAPE
    assert FEATURE_SHAPE is not None
    feature = tf.compat.v1.placeholder(tf.float32, shape=[None] + FEATURE_SHAPE, name="input")
    reciever_tensor = {"input": feature}
    return tf.estimator.export.ServingInputReciver(feature, reciever_tensor)
def model_fn(features, labels, mode, params):
    global NUM_CLASSES
    assert NUM_CLASSES is not None
    model = tf.keras.Sequential([
      hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
        output_shape=[2048],
        trainable=False
      ),
      tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    optimizer = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.optimizers.Adam(params.get("learing_rate", 1e-3))
        if params.get("use_tpu", True):
          optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    with tf.GradientTape() as tape:
        logits = model(features)
        if mode == tf.estimator.ModeKeys.PREDICT:
            preds = {
                "predictions": logits
            }
            return tpu_estimator.TPUEstimatorSpec(mode, predictions=preds)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)(labels, logits)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tpu_estimator.TPUEstimatorSpec(mode, loss=loss)

    def train_fn():
        assert optimizer is not None
        gradient = tape.gradient(loss, model.trainable_variables)
        global_step = tf.compat.v1.train.get_global_step()
        update_global_step = tf.compat.v1.assign(global_step, global_step + 1, name='update_global_step')
        with tf.control_dependencies([update_global_step]):
          apply_grads = optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        return apply_grads

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tpu_estimator.TPUEstimatorSpec(mode, loss=loss, train_op=train_fn())

def main(_):
  os.environ["TFHUB_CACHE_DIR"] = os.path.join(FLAGS.model_dir,"tfhub_modules")
  os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
  input_fn = partial(input_, iterations=FLAGS.iterations)
  cluster = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
  run_config =  tpu_config.RunConfig(
    model_dir = FLAGS.model_dir,
    cluster=cluster,
    tpu_config=tpu_config.TPUConfig(FLAGS.iterations))

  classifier = tpu_estimator.TPUEstimator(
    model_fn=model_fn,
    use_tpu=FLAGS.use_tpu,
    train_batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.batch_size,
    config=run_config,
    params={
      "use_tpu": FLAGS.use_tpu,
      "data_dir": FLAGS.data_dir,
      "dataset": FLAGS.dataset
    }
  )

  classifier.train(
          input_fn=lambda params:input_fn(
              mode=tf.estimator.ModeKeys.TRAIN,
              **params),
          max_steps=1000)#.evaluate(
                  #input_fn=lambda params:input_fn(
                  #    mode=tf.estimator.ModeKeys.EVAL,
                  #    **params),
                  #steps=1000)
  classifier.export_savedmodel(os.path.join(FLAGS.model_dir,"export"), serving_reciever_input_fn)
if __name__ == "__main__":
  app.run(main)

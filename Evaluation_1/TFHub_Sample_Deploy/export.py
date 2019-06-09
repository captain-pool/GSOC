import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
""" Create a Sample TF-Hub Module using SavedModel v2.0
The module has as a single signature which loads MNIST Dataset from TFDS and train a simple Neural Network for classifying the digits. The model is built and trained using Tewnsorlfow 
"""

class MNIST(tf.keras.models.Model):
    def __init__(self, output_activation="softmax"):
        super(MNIST, self).__init__()
        self.layer_1 = tf.keras.layers.Dense(64)
        self.layer_2 = tf.keras.layers.Dense(10, activation=output_activation)

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[
                    None,
                    28,
                    28,
                    1],
                dtype=tf.uint8)])
    def call(self, inputs):
        casted = tf.keras.layers.Lambda(
            lambda x: tf.cast(x, tf.float32))(inputs)
        flatten = tf.keras.layers.Flatten()(casted)
        normalize = tf.keras.layers.Lambda(
            lambda x: x / tf.reduce_max(tf.gather(x, 0)))(flatten)
        x = self.layer_1(normalize)
        output = self.layer_2(x)
        return output


model = MNIST()
train, test = tfds.load("mnist", split=["train", "test"])
optimizer_fn = tf.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.Mean()
model.compile(optimizer_fn, loss=loss_fn)
train = train.shuffle(1, reshuffle_each_iteration=True).batch(16)


@tf.function
def train_step(image, label):
    with tf.GradientTape() as tape:
        preds = model(image)
        label_onehot = tf.one_hot(label, 10)
        loss_ = loss_fn(label_onehot, preds)
    grads = tape.gradient(loss_, model.trainable_variables)
    optimizer_fn.apply_gradients(zip(grads, model.trainable_variables))
    metric(loss_)


@tf.function
def test(image, label):
    preds = model(image)
    label_onehot = tf.one_hot(label, 10)


for epoch in range(10):
    for step, data in enumerate(train):
        train_step(data['image'], data['label'])
        if step % 100 == 0:
            print("Epoch #{}::\tStep #{}:\tLoss: {}".format(
                epoch, step, metric.result().numpy()))
tf.saved_model.save(model, "/tmp/tfhub_modules/mnist/digits/1")

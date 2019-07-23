import tensorflow as tf
model = tf.keras.Sequential()
model.compile(optimizer="adam", loss="mean_squared_error")
tf.saved_model.save(model, "/tmp/model/2")

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import tensorflow_hub as hub
import os
import io
import requests
from tqdm import tqdm
import tarfile

DOWNLOAD_LINK = "https://s3.amazonaws.com/download.onnx/models/opset_8/shufflenet.tar.gz"


if not os.path.exists(DOWNLOAD_LINK.split("/")[-1]):
  response = requests.get(DOWNLOAD_LINK, stream=True)
  with open(DOWNLOAD_LINK.split("/")[-1], "wb") as handle:
    for data in tqdm(response.iter_content(chunk_size=io.DEFAULT_BUFFER_SIZE)):
      handle.write(data)
  tar = tarfile.open(DOWNLOAD_LINK.split("/")[-1])
  tar.extractall()
  tar.close()


if not os.path.exists("shufflenet.pb"):
  model = onnx.load("shufflenet/model.onnx")
  tf_rep = prepare(model)
  tf_rep.export_graph("shufflenet.pb")



def module_fn():
  input_name = "gpu_0/data_0:0"
  output_name = "Softmax:0"
  graph_def = tf.GraphDef()
  with tf.gfile.GFile("shufflenet.pb", 'rb') as f:
    graph_def.ParseFromString(f.read())
  input_tensor = tf.placeholder(tf.float32, shape=[1, 3, 224, 224])
  output_tensor, = tf.import_graph_def(graph_def, input_map={input_name: input_tensor}, return_elements=[output_name])
  hub.add_signature(inputs=input_tensor, outputs=output_tensor)


spec = hub.create_module_spec(module_fn)
module = hub.Module(spec)


with tf.Session() as sess:
  module.export("onnx/shufflenet/1", sess)


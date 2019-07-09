# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
SHUFFLENET_PB = "shufflenet.pb"


def load_shufflenet():
  # Download Shufflenet if it doesn't exist
  if not os.path.exists(DOWNLOAD_LINK.split("/")[-1]):
    response = requests.get(DOWNLOAD_LINK, stream=True)
    with open(DOWNLOAD_LINK.split("/")[-1], "wb") as handle:
      for data in tqdm(
              response.iter_content(
                  chunk_size=io.DEFAULT_BUFFER_SIZE),
              total=int(
                  response.headers['Content-length']) //
              io.DEFAULT_BUFFER_SIZE,
              desc="Downloading"):
        handle.write(data)
    tar = tarfile.open(DOWNLOAD_LINK.split("/")[-1])
    tar.extractall()
    tar.close()
  # Export Protobuf File if not present
  if not os.path.exists(SHUFFLENET_PB):
    model = onnx.load("shufflenet/model.onnx")
    tf_rep = prepare(model)
    tf_rep.export_graph(SHUFFLENET_PB)


def module_fn():
  input_name = "gpu_0/data_0:0"
  output_name = "Softmax:0"
  graph_def = tf.GraphDef()
  with tf.gfile.GFile(SHUFFLENET_PB, 'rb') as f:
    graph_def.ParseFromString(f.read())
  input_tensor = tf.placeholder(tf.float32, shape=[1, 3, 224, 224])
  output_tensor, = tf.import_graph_def(
      graph_def, input_map={
          input_name: input_tensor}, return_elements=[output_name])
  hub.add_signature(inputs=input_tensor, outputs=output_tensor)


def main():
  load_shufflenet()
  spec = hub.create_module_spec(module_fn)
  module = hub.Module(spec)
  with tf.Session() as sess:
    module.export("onnx/shufflenet/1", sess)
  print("Exported")


if __name__ == "__main__":
  main()

ShuffleNet ONNX to Tensorflow Hub Module Export
-------------------------------------------------
**Original Repsitory for ONNX Model**: https://github.com/onnx/models/tree/master/shufflenet

**Description**: Shufflenet is a Deep Convolutional Network for Classification

**Original Paper**: [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

## Module Properties
1. Load as **Channel First**
2. Input Shape: [1, 3, 224, 224]
2. Output Shape: [1, 1000]
3. Download Size: 5.3 MB

## Steps
- `pip install -r requirements.txt`
- `python3 export.py`

 The Tensorflow Hub Module is Exported as *onnx/shufflenet/1*

## Load and Use Model

 ```python3
 import tensorflow_hub as hub
 module = hub.Module("onnx/shufflenet/1")
 result = module(...)
 ```
 ## Important
 The module produced is **not** functional. Check issue, https://github.com/captain-pool/GSOC/issues/3

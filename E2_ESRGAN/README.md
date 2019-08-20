# Enhanced Super Resolution GAN
Tensorflow 2.0 Implementation of Enhanced Super Resolution Generative Adversarial Network (Xintao et. al.)
[https://arxiv.org/pdf/1809.00219.pdf](https://arxiv.org/pdf/1809.00219.pdf)

Enhanced Super Resolution GAN implemented as a part of Google Summer of Code 2019. [https://summerofcode.withgoogle.com/projects/#4662790671826944](https://summerofcode.withgoogle.com/projects/#4662790671826944)
The SavedModel is expected to be shipped as a Tensorflow Hub Module. [https://tfhub.dev/](https://tfhub.dev/)

## Overview
Enhanced Super Resolution GAN is an improved version of Super Resolution GAN (Ledig et.al.) [https://arxiv.org/abs/1609.04802](https://arxiv.org/abs/1609.04802).
The Model uses Residual-in-Residual Block, as the basic convolutional block instead of the basic Residual Network or simple Convolution trunk to provide a better flow of gradients at the microscopic level.
In addition to that the model lacks Batch Normalization layers, in the Generator to prevent smoothing out of the artifacts in the image. This allows
ESRGAN to produce images having better approximation of the sharp edges of the image artifacts.
ESRGAN uses a Relativistic Discriminator [https://arxiv.org/pdf/1807.00734.pdf](https://arxiv.org/pdf/1807.00734.pdf) to better approximate the probability of an
image being real or fake thus producing better result.
The generator uses a linear combination of Perceptual difference between real and fake image (using pretrained VGG19 Network), Pixelwise absolute difference between real and fake image
and Relativistic Average Loss between the real and fake image as loss function during adversarial training.
The generator is trained in a two phase training setup.
- First Phase focuses on reducing the Pixelwise L1 Distance of the input and target high resolution image to prevent local minimas
obtained while starting from complete randomness.
- Second Phase focuses on creating sharper and better reconstruction of minute artifacts.

The final trained model is then interpolated between the L1 loss model and adversarially trained model, to produce photo realistic
reconstruction.
## Example Usage
```python3
import tensorflow_hub as hub
import tensorflow as tf
model = hub.load("https://github.com/captain-pool/GSOC/releases/download/1.0.0/esrgan.tar.gz")
super_resolution = model.call(LOW_RESOLUTION_IMAGE_OF_SHAPE=[BATCH, HEIGHT, WIDTH, 3])
# Output Shape: [BATCH, 4 x HEIGHT, 4 x WIDTH, 3]
# Output DType: tf.float32.
# NOTE:
# The values are needed to be clipped between [0, 255]
# using tf.clip_by_value(...) and casted to tf.uint8 using tf.cast(...)
# before plotting or saving as image
```
## Results

The model trained on DIV2K dataset on reconstructing 128 x 128 image by a scaling factor 4, yielded the following images.

![ESRGAN_DIV2K](https://user-images.githubusercontent.com/13994201/63384084-c7ce5680-c3bb-11e9-96cc-99d9b8cb6804.jpg)

**The model is in State of the Art: 32.6 PSNR on 512 x 512 image patches.**

## SavedModel 2.0
Loadable SavedModel can be found at https://github.com/captain-pool/GSOC/releases/tag/1.0.0

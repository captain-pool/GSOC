# Enhanced Super Resolution GAN
Tensorflow 2.0 Implementation of Enhanced Super Resolution Generative Adversarial Network (Xintao et. al.)
[https://arxiv.org/pdf/1809.00219.pdf](https://arxiv.org/pdf/1809.00219.pdf)

Enhanced Super Resolution GAN implemented as a part of Google Summer of Code 2019. [https://summerofcode.withgoogle.com/projects/#4662790671826944](https://summerofcode.withgoogle.com/projects/#4662790671826944)
The SavedModel is expected to be shipped as a Tensorflow Hub Module. [https://tfhub.dev/](https://tfhub.dev/)

## Abstract
Enhanced Super Resolution GAN is an improved version of Super Resolution GAN (Ledig et.al.) [https://arxiv.org/abs/1609.04802](https://arxiv.org/abs/1609.04802).
The Model flaunts the use of Residual-in-Residual Block, as the basic convolutional block instead of the basic Residual Network used as the Generator of SRGAN.
Another improvement over SRGAN is, the lack of Batch Normalization layers, in the Generator to prevent smoothing out of the artifacts in the image. This allows
ESRGAN to produce images having better approximation of the sharp edges of the image artifacts.
ESRGAN uses a Relativistic Discriminator [https://arxiv.org/pdf/1807.00734.pdf](https://arxiv.org/pdf/1807.00734.pdf) to better approximate the probability of an
image being real or fake thus producing better result.
The generator uses a linear combination of Perceptual difference between real and fake image (using pretrained VGG19 Network), Pixelwise absolute difference between real and fake image
and Relativistic Average Loss between the real and fake image as a loss function.
The generator is trained in a two phase training setup.
- First Phase focuses on reducing the Pixelwise L1 Distance of the input and target high resolution image to prevent local minimas
obtained while starting from complete randomness.
- Second Phase focuses on creating sharper and better reconstruction of minute artifacts.

The final trained model is then interpolated between the L1 loss model and adversarially trained model, to produce photo realistic
reconstruction.

## Results

The model trained on COCO2014 dataset on reconstructing 64 x 64 image by a scaling factor 4, yielded the following images.
Original Image
![Origin Resolution](https://user-images.githubusercontent.com/13994201/61579693-23d85d80-ab26-11e9-8372-5cd3034742fc.jpg)
Low Resolution (Bicubic Interpolation)
![Low Resolution](https://user-images.githubusercontent.com/13994201/61579669-d0660f80-ab25-11e9-8932-5356b244b767.jpg)
Reconstructed Super Resolution Image
![Super Resolution](https://user-images.githubusercontent.com/13994201/61579668-d0660f80-ab25-11e9-80e2-2115e3f46f28.jpg)

*The model checkpoints and the SavedModel will be released soon.*

# GAN Distillation on Enhanced Super Resolution GAN
Link to ESRGAN Tranining Codes: [Click Here](E2_ESRGAN)

Knowledge Distillation on Enhanced Super Resolution GAN to perform Super Resolution on model with much smaller number of
variables.
The Training Algorithm is **inspired** from https://arxiv.org/abs/1902.00159, with a custom loss function specific to the 
problem of image super resolution.

Results
------------------

### ESRGAN
**Latency**: 17.117 Seconds

**Mean PSNR Achieved**: 28.2

**Sample**:

Input Image Shape: 180 x 320

Output image shape: 720 x 1280

![esrgan](https://user-images.githubusercontent.com/13994201/63640629-251a1e80-c6c0-11e9-98bc-04432c7064e2.jpg "ESRGAN")

**PSNR of the Image**: 30.462

### Compressed ESRGAN
**Latency**: _**0.4 Seconds**_

**Mean PSNR Achieved**: 25.3

**Sample**

Input Image Shape: 180 x 320

Output image shape: 720 x 1280

![compressed_esrgan](https://user-images.githubusercontent.com/13994201/63640526-1121ed00-c6bf-11e9-99f5-0b48069fe784.jpg "Compressed ESRGAN")
**PSNR of the Image**: 26.942

Student Model
----------------
The Residual in Residual Architecture of ESRGAN was followed. With much shallower trunk.
Specifically,

|Name of Node|Depth|
|:-:|:-:|
|Residual Dense Blocks(RDB)|2 Depthwise Convolutions|
|Residual in Residual Blocks(RRDB)|2 RDB units|
|Trunk|3 RRDB units|
|UpSample Layer|1 ConvTranspose unit with a stride length of 4|

**Size of Growth Channel (intermediate channel) used:** 32

Trained Saved Model and TF Lite File
-----------------------------------------

#### Specification of the Saved Model
Input Dimension: `[None, 180, 320, 3]`

Input Data Type: `Float32`

Output Dimension: `[None, 180, 320, 3]`

TensorFlow loadable link: https://github.com/captain-pool/GSOC/releases/download/2.0.0/compressed_esrgan.tar.gz

#### Specification of TF Lite
Input Dimension: `[1, 180, 320, 3]`

Input Data Type: `Float32`

Output Dimension: `[1, 720, 1280, 3]`

TensorFlow Lite: https://github.com/captain-pool/GSOC/releases/download/2.0.0/compressed_esrgan.tflite

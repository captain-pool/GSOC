## Super Resolution Video Player
With success from Distilling ESRGAN to produce high resolution images in milliseconds, (0.3 seconds to be specific). This
application tests the model on the extreme last level. This proof of concept video player takes in normal videos and downscales the frames
bicubically and then perform super resolution on the video frames on CPU and displays them in realtime.

Here's a sample.

Video used: Slash's Anastatia Live in Sydney. (https://www.youtube.com/watch?v=bC8EmPA6H6g)

![sr_player](https://user-images.githubusercontent.com/13994201/63652070-cdd88480-c779-11e9-8890-8dc379755926.png)

Usage
-----------

```bash
$ python3 player.py --file /path/to/video/file \
                    --saved_model /path/to/saved_model
```

Issues
-------
Check the issue tracker with "Super Resolution Player" as issue label or
[click here](https://github.com/captain-pool/GSOC/issues?q=is%3Aissue+is%3Aopen+label%3A%22Super+Resolution+Player%22)

Further Concepts
------------------
Building an online video streamer sending low resolution videos running on Super Resolved video on the client side.
Check the [experimental](experimental) folder for the codes.

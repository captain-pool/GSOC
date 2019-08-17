from absl import logging
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import os
import time
import pyaudio as pya
import threading
import queue
import numpy as np
from moviepy import editor
import pygame
pygame.init()
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"


class Player(object):
  def __init__(self, videofile, tflite="", saved_model=""):
    """
      Player Class for the Video
      Args:
        videofile: Path to the video file
        tflite: Path to the Super Resolution TFLite
        saved_model: path to Super Resolution SavedModel
    """
    self.video = editor.VideoFileClip(videofile)
    self.audio = self.video.audio
    self.tolerance = 2.25  # Higher Tolerance Faster Video
    self.running = False
    self.interpreter = None
    self.saved_model = None
    if saved_model:
      self.saved_model = hub.load(saved_model)
    if tflite:
      self.interpreter = tf.lite.Interpreter(model_path=tflite)
      self.interpreter.allocate_tensors()
      self.input_details = self.interpreter.get_input_details()
      self.output_details = self.interpreter.get_output_details()
    self.lock = threading.Lock()
    self.audio_thread = threading.Thread(target=self.write_audio_stream)
    self.video_thread = threading.Thread(target=self.write_video_stream)
    self.video_iterator = self.video.iter_frames()
    self.audio_iterator = self.audio.iter_chunks(int(self.audio.fps))
    self.video_queue = queue.Queue()
    self.audio_queue = queue.Queue()
    pyaudio = pya.PyAudio()
    issmallscreen = 1 if saved_model or tflite else 0.25
    self.screen = pygame.display.set_mode(
        (int(1080 * issmallscreen),
         int(720 * issmallscreen)), 0, 32)
    self.stream = pyaudio.open(
        format=pya.paFloat32,
        channels=2,
        rate=44100,
        output=True,
        frames_per_buffer=1024)

  def tflite_super_resolve(self, frame):
    """
      Super Resolve bicubically downsampled image frames
      using the TFLite of the model.
      Args:
        frame: Image frame to scale up.
    """
    frame = tf.expand_dims(tf.convert_to_tensor(frame), 0)
    frame = tf.image.resize(frame, size=[720 // 4, 1080 // 4])
    self.interpreter.set_tensor(self.input_details[0]['index'], frame)
    self.interpreter.invoke()
    frame = self.interpreter.get_tensor(self.output_details[0]['index'])
    frame = tf.squeeze(tf.cast(tf.clip_by_value(frame, 0, 255), tf.uint8))
    return frame.numpy()

  def saved_model_super_resolve(self, frames):
    """
      Super Resolve using exported SavedModel.
      Args:
        frames: Batch of Frames to Scale Up.
    """
    logging.debug("Stacking")
    frames = tf.stack(frames)
    logging.debug("Resizing")
    frames = tf.image.resize(frames, size=[720 // 4, 1080 // 4], method="bicubic")
    if self.saved_model:
      start = time.time()
      frames = self.saved_model.call(frames)
      logging.debug("Super Resolving Time: %f" % (time.time() - start))
    logging.debug("Casting and Clipping")
    frames = tf.cast(tf.clip_by_value(frames, 0, 255), tf.uint8)
    logging.debug("Returning Modified Frames")
    return frames.numpy()

  def video_second(self):
    """
      Fetch Video Frames for each second
      and super resolve them accordingly.
    """
    frames = []
    logging.debug("Fetching Frames")
    start = time.time()
    loop_time = time.time()
    for _ in range(int(self.video.fps)):
      logging.debug("Fetching Video Frame. %f" % (time.time() - loop_time))
      loop_time = time.time()
      if self.interpreter and not self.saved_model:
        frames.append(self.tflite_super_resolve(next(self.video_iterator)))
      if not self.interpreter:
        frames.append(tf.convert_to_tensor(next(self.video_iterator)))
    logging.debug("Frame Fetching Time: %f" % (time.time() - start))
    if not self.interpreter:
      frames = self.saved_model_super_resolve(frames)
    logging.debug("Fetched Frames")
    return frames

  def fetch_video(self):
    """
      Fetches audio and video frames from the file.
      And put them in player cache.
    """
    audio = next(self.audio_iterator)
    video = self.video_second()
    self.audio_queue.put(audio)
    self.video_queue.put(video)

  def write_audio_stream(self):
    """
      Write Audio Frames to default audio device.
    """
    try:
      while self.audio_queue.qsize() < 8:
        continue
      while self.running:
        audio = self.audio_queue.get(timeout=10)
        self.stream.write(audio.astype("float32").tostring())
    except BaseException:
      raise

  def write_video_stream(self):
    """
      Write Video frames to the player display.
    """
    try:
      while self.video_queue.qsize() < 8:
        continue
      while self.running:
        for video_frame in self.video_queue.get(timeout=10):
          video_frame = pygame.surfarray.make_surface(
              np.rot90(np.fliplr(video_frame)))
          self.screen.fill((0, 0, 2))
          self.screen.blit(video_frame, (0, 0))
          pygame.display.update()
          time.sleep((1000 / self.video.fps - self.tolerance) / 1000)
    except BaseException:
      raise

  def run(self):
  """
    Start the player threads and the frame streaming simulator.
  """
    with self.lock:
      if not self.running:
        self.running = True
    self.audio_thread.start()
    self.video_thread.start()
    for _ in range(int(self.video.duration)):
      logging.debug("Fetching Video")
      self.fetch_video()
      time.sleep(0.1)
    with self.lock:
      if not self.running:
        self.running = True
    self.audio_thread.join()
    self.video_thread.join()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-v", "--verbose",
      action="count",
      default=0,
      help="Increases Verbosity of Logging")
  parser.add_argument(
      "--file",
      default=None,
      help="File to play")
  parser.add_argument(
      "--tflite",
      default="",
      help="Path to TFLite File")
  parser.add_argument(
      "--saved_model",
      default="",
      help="Path to Saved Model File")
  FLAGS, unknown_args = parser.parse_known_args()
  log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
  current_log_level = log_levels[min(len(log_levels) - 1, FLAGS.verbose)]
  logging.set_verbosity(current_log_level)
  player = Player(
      videofile=FLAGS.file,
      saved_model=FLAGS.saved_model,
      tflite=FLAGS.tflite)
  player.run()

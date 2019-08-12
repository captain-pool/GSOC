from __future__ import print_function
import io
import socket
import threading
import time

from absl import logging
import queue
import numpy as np
import datapacket_pb2
import pyaudio as pya
import pygame
import tensorflow as tf
from pygame.locals import *  # pylint: disable=wildcard-import
pygame.init()


class Client(object):
  SYN = b'SYN'
  SYNACK = b'SYN/ACK'
  ACK = b'ACK'

  def __init__(self, ip, port):
    self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._metadata = datapacket_pb2.Metadata()
    self._audio_queue = queue.Queue()
    self._video_queue = queue.Queue()
    self._audio_thread = threading.Thread(
        target=self.write_to_stream, args=(True,))
    self._video_thread = threading.Thread(
        target=self.write_to_stream, args=(False,))
    self._running = False
    self.tolerance = 30  # Higher Tolerance Higher Frame Rate
    self._lock = threading.Lock()
    pyaudio = pya.PyAudio()
    self._audio_stream = pyaudio.open(
        format=pya.paFloat32,
        channels=2,
        rate=44100,
        output=True,
        frames_per_buffer=1024)
    self._socket.connect((ip, port))
    self.fetch_metadata()

  def readpacket(self, buffersize=2**30):
    buffer_ = io.BytesIO()
    done = False
    eof = False
    while not done:
      data = self._socket.recv(buffersize)
      if data:
        logging.debug("Reading Stream: Buffer Size: %d" % buffersize)
      if data[-5:] == b'<EOF>':
        logging.debug("Found EOF")
        data = data[:-5]
        eof = True
        done = True
      if data[-5:] == b'<END>':
        logging.debug("Find End of Message")
        data = data[:-5]
        done = True
      buffer_.write(data)
    buffer_.seek(0)
    return buffer_.read(), eof

  def fetch_metadata(self):
    logging.debug("Sending SYN...")
    self._socket.send(b'SYN')
    logging.debug("Sent Syn. Awating Metadata")
    data, eof = self.readpacket(8)
    self._metadata.ParseFromString(data)
    dimension = self._metadata.dimension
    self.screen = pygame.display.set_mode(dimension[:-1][::-1], 0, 32)

  def fetch_video(self):
    data, eof = self.readpacket()
    framedata = datapacket_pb2.FramePacket()
    framedata.ParseFromString(data)
    frames = []
    for frame in framedata.video_frames:
      frames.append(self.parse_frames(frame, False))
    self._audio_queue.put(framedata.audio_chunk)
    self._video_queue.put(frames)
    return eof

  def parse_frames(self, bytestring, superresolve=False):
    frame = tf.io.parse_tensor(bytestring, out_type=tf.float32)
    if superresolve:
      # Perform super resolution here
      pass
    frame = tf.cast(tf.clip_by_value(frame, 0, 255), tf.float32)
    return frame.numpy()

  def start(self):
    with self._lock:
      self._running = True
    if not self._audio_thread.isAlive():
      self._audio_thread.start()
    if not self._video_thread.isAlive():
      self._video_thread.start()
    self._socket.send(b'ACK')
    while not self.fetch_video():
      pass  # Wait till the end
    self.wait_to_end()

  def wait_to_end(self):
    self._audio_thread.join()
    self._video_thread.join()

  def stop(self):
    with self.lock:
      self._running = False

  def write_to_stream(self, isaudio=False):
    while self._running:
      try:
        if isaudio:
          if self._audio_queue.qsize() < 5:
            continue
          audio_chunk = self._audio_queue.get(timeout=10)
          self._audio_stream.write(audio_chunk)
        else:
          if self._video_queue.qsize() < 5:
            continue
          for video_frame in self._video_queue.get(timeout=10):
            video_frame = pygame.surfarray.make_surface(
                np.rot90(np.fliplr(video_frame)))
            self.screen.fill((0, 0, 2))
            self.screen.blit(video_frame, (0, 0))
            pygame.display.update()
            time.sleep(
                (1000 / self._metadata.video_fps - self.tolerance) / 1000)
      except StopIteration:
        pass


if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  client = Client("127.0.0.1", 8001)
  client.start()

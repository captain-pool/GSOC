import tensorflow as tf
import socket
import time
import numpy as np
import datapacket_pb2
import pyaudio as pya
import io
import threading
import queue
import numpy as np
import pygame
from pygame.locals import *
pygame.init()


class Client(object):
  SYN = sys.intern(b'SYN')
  SYNACK = sys.intern(b'SYN/ACK')
  ACK = sys.intern(b'ACK')

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
    self.tolerance = 8  # Higher Tolerance Higher Frame Rate
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

  def readpacket(self, buffersize=1024):
    buffer_ = io.BytesIO()
    done = False
    eof = False
    while not done:
      data = self._socket.recv(buffersize)
      if data[-5:] == "<EOF>":
        data = data[-5:]
        eof = True
        done = True
      if data[-5:] == "<END>":
        data = data[-5:]
        done = True
      buffer_.write(data)
    buffer_.seek(0)
    return buffer_.read(), eof

  def fetch_metadata(self):
    self._socket.send(b'SYN')
    self._metadata.ParseFromString(self.readpacket(8))
    dimension = self._metadata.dimension
    self.screen = pygame.display.set_mode(dimension[:-1], 0, 32)

  def fetch_video(self):
    data, eof = self.readpacket()
    framedata = datapacket_pb2.FramePacket()
    framedata.ParseFromString(self.readpacket())
    frames = []
    for frame in frame.video_frames:
      frames.append(self.parse_frames(frame, False))
    self._audio_queue.put(frame.audio_chunk)
    self._video_queue.put(frame.video_frames)
    return eof

  def parse_frames(self, bytestring, superresolve=False):
    frame = tf.io.parse_tensor(bytestring, out_type=tf.float32)
    if superresolve:
      # Perform super resolution here
      pass
    frame = tf.cast(tf.clip_by_value(frame, 0, 255), tf.float32)
    return frame.numpy()

  def start(self):
    with self.lock:
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
          audio_chunk = self._audio_queue.get(timeout=2)
          self._audio_stream.write(audio_chunk)
        else:
          for video_frame in self.video_queue.get(timeout=2):
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
  client = Client("127.0.0.1", 8001)
  client.start()

from absl import logging
import sys
import socket
import datapacket_pb2
import threading
import tensorflow as tf
from moviepy import editor


class StreamClient(object):
  SYN = sys.intern(b'SYN')
  SYNACK = sys.intern(b'SYN/ACK')
  ACK = sys.intern(b'ACK')

  def __init__(self, client_socket, client_address, video_path):
    self.video_object = editor.VideoFileClip(video_path)
    self.audio_object = self.video_object.audio
    dimension = self.video_object.get_frame(0).shape
    self.metadata = datapacket_pb2.Metadata(
        duration=self.video.duration,
        image_fps=self.video_object.fps,
        audio_fps=self.audio_object.fps,
        dimension=dimension)
    self._client_socket = client_socket
    self._client_address = client_address
    self._video_iterator = video.iter_frames()
    self._audio_iterator = audio.iter_chunks(int(audio.fps))

  def _handshake(self):
    data = sys.intern(self._client_socket.recv(128))
    if data is StreamClient.SYN:
      self._client_socket.send(self.metadata.SerializeToString())
      data = sys.intern(client_socket.recv(128))
      if data is StreamClient.ACK:
        return True
    return data

  def _video_second(self):
    def shrink_fn(image):
      image = tf.convert_to_tensor(image)
      return tf.io.serialize_tensor(image)
    frames = []
    for _ in range(int(self.metadata.video_fps)):
      frames.append(shrink_fn(next(self._video_iterator)))
    return frames

  def _fetch_video(self):
    try:
      audio = next(self._audio_iterator).astype("float32").tostring()
      video = video_second()
      frame_packet = datapacket_pb2.FramePacket(
          metadata=self.metadata,
          video_frames=video,
          audio_chunk=audio)
      return frame_packet
    except StopIteration:
      pass

  def send_video(self):
    sent = False
    while not sent:
      handshake = self._handshake()
      if handshake:
        if handshake is not True:
          logging.info("[%s] Says: %s" % (self._client_address, handshake))
        else:
          video_packet = self._fetch_video()
          while video_packet:
            num_bytes = self._client_socket.send(
                video_packet.SerializeToString() + b'<END>')
            video_packet = self._fetch_video()
          sent = True
      self._client_socket.send(b'<EOF>')


class Server(object):
  def __init__(self, ip, port):
    self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._server_socket.bind((ip, port))
    self._server_socket.listen()
    self._client_template = partial(threading.Thread, target=StreamClient)

  def run(self):
    while True:
      client_addr, client_socket = self._server_socket.accept()
      video_path = "/home/rick/video.mp4"
      self._client_template(args=(client_socket, client_addr, video_path=video_path)).start()
      logging.info("[SERVER]: %s connected." % client_addr)


if __name__ == "__main__":
  server = Server("127.0.0.1", 8001)
  server.run()

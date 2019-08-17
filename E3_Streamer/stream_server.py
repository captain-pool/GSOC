import socket
from absl import logging
from functools import partial
import threading
import datapacket_pb2
import tensorflow as tf
from moviepy import editor


class StreamClient(object):
  SYN = b'SYN'
  SYNACK = b'SYN/ACK'
  ACK = b'ACK'

  def __init__(self, client_socket, client_address, video_path):
    self.video_object = editor.VideoFileClip(video_path)
    self.audio_object = self.video_object.audio
    dimension = list(self.video_object.get_frame(0).shape)
    self.metadata = datapacket_pb2.Metadata(
        duration=int(self.video_object.duration),
        video_fps=int(self.video_object.fps),
        audio_fps=int(self.audio_object.fps),
        dimension=dimension)
    self._client_socket = client_socket
    self._client_address = client_address
    self._video_iterator = self.video_object.iter_frames()
    self._audio_iterator = self.audio_object.iter_chunks(self.metadata.audio_fps)
    self.send_video()
  def _handshake(self):
    logging.debug("Awaiting Handshake")
    data = self._client_socket.recv(128)
    if data == StreamClient.SYN:
      logging.debug("SYN Recieved. Sending Media Metadata")
      num_bytes = self._client_socket.send(self.metadata.SerializeToString()+b'<END>')
      logging.debug("Metadata sent: Num Bytes Written: %d" % num_bytes)
      logging.debug("Awaiting ACK")
      data = self._client_socket.recv(128)
      logging.debug("Data Recieved")
      if data == StreamClient.ACK:
        logging.debug("ACK Recieved")
        return True
    return data

  def _video_second(self):
    def shrink_fn(image):
      image = tf.convert_to_tensor(image)
      return image.numpy().tostring()
    frames = []
    for _ in range(int(self.metadata.video_fps)):
      frames.append(shrink_fn(next(self._video_iterator)))
    return frames

  def _fetch_video(self):
    try:
      audio = next(self._audio_iterator).astype("float32").tostring()
      video = self._video_second()
      frame_packet = datapacket_pb2.FramePacket(
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
            logging.debug("Sending: %d" % num_bytes)
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
      client_socket, client_addr = self._server_socket.accept()
      client_addr = list(map(str, client_addr))
      video_path = "/home/rick/video.mp4"
      self._client_template(
          args=(client_socket, client_addr, video_path)
      ).start()
      logging.info("[SERVER]: %s connected." % ":".join(client_addr))


if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  server = Server("127.0.0.1", 8001)
  server.run()

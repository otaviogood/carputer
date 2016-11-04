"""A threaded cv2 camera.

Largely from pyimagesearch.com
"""

import threading
import cv2


class CameraStream(object):
  def __init__(self, src=1):
    self.stream = cv2.VideoCapture(src)
    if not self.stream.isOpened():
      src = 1 - src
      self.stream = cv2.VideoCapture(src)
      if not self.stream.isOpened():
        sys.exit("Error: Camera didn't open for capture.")

    # Setup frame dims.
    self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 160)
    self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 120)

    self.grabbed, self.frame = self.stream.read()
    self.stopped = False

  def start(self):
    """Start the thread to read frames from the video stream."""
    t = threading.Thread(target=self.update, args=())
    t.daemon = True
    t.start()
    return self

  def update(self):
    """Grab frames until told to stop."""
    while not self.stopped:
      self.grabbed, self.frame = self.stream.read()

  def read(self):
    return self.frame

  def stop(self):
    self.stopped = True

"""Threaded keyboard input."""

import threading


class KeyWatcher(object):
  def __init__(self, last_key):
    """Have to pass in a mutable shared reference."""
    self.last_key = last_key
    self.stopped = False

  def start(self):
    t = threading.Thread(target=self.update, args=(self.last_key,))
    t.daemon = True
    t.start()
    return self

  def update(self, key):
    """Grab keys until told to stop.

    Not sure what will happen when other stuff prints to the screen..
    """
    while not self.stopped:
      key[0] = raw_input()

  def read(self):
    return self.last_key

  def stop(self):
    self.stopped = True

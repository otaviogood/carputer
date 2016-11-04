"""Manually testing the key_watcher module.

Remember to hit enter / return after you type stuff.
"""

import random
import sys
import threading
import time

import key_watcher


last_key = ['']
keys = key_watcher.KeyWatcher(last_key).start()

while True:
    print 'randomness: %0.2f' % random.random()
    print 'last key:', last_key
    time.sleep(2)

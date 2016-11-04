"""Fixing odo vals.

Usage:
  fix_odo.py <inpath> <outpath>
"""

import os
import shutil
import sys

from docopt import docopt


EXPECTED_MAX_ODO = 750


# Parse args.
args = docopt(__doc__)
inpath = os.path.expanduser(args['<inpath>'])
outpath = os.path.expanduser(args['<outpath>'])
file_count = len([f for f in os.listdir(inpath) if 'lidar' not in f])
print '%s files' % file_count

# Determine where to save.
if inpath[-1] == '/':
  inpath = inpath[:-1]
_, basepath = os.path.split(inpath)
savedir = os.path.join(outpath, '%s-fixed' % basepath)
if not os.path.exists(savedir):
  os.makedirs(savedir)

# Fix the odos..
last_odo = None
odo_fix = 0
for i, filename in enumerate(os.listdir(inpath)):
  # Filenames are like "frame_31464_thr_101_ste_91_mil_1070052_odo_00708.png"
  if 'lidar' in filename:
    continue
  if 'frame' not in filename:
    continue

  _, frame, _, throttle, _, steering, _, millis, _, odo = filename.split('_')
  odo = int(odo.split('.')[0])

  # Init last odo.
  if not last_odo:
    last_odo = odo
    continue

  if odo == 0 and last_odo < EXPECTED_MAX_ODO:
    odo_fix = last_odo
  elif odo == 0 and last_odo + odo_fix >= EXPECTED_MAX_ODO:
    odo_fix = 0

  new_odo = str(odo + odo_fix).zfill(5)
  new_filename = 'frame_%s_thr_%s_ste_%s_mil_%s_odo_%s.png' % (
    frame, throttle, steering, millis, new_odo)

  source_file = os.path.join(inpath, filename)
  dest_file = os.path.join(savedir, new_filename)
  shutil.copyfile(source_file, dest_file)

  last_odo = odo + odo_fix

  if i % 100 == 0:
    sys.stdout.write('\rprocessing.. %0.1f%%' % (100. * i / file_count))
    sys.stdout.flush()

print '\ncomplete.'

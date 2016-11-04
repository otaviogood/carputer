"""Plotting recorded data (steering, throttle, odometer and frame) vs time (millis).

Usage:
  plot_vs_time.py <indir> [--outdir=<path>]

Arguments:
  <indir>    path to recorded images

Options:
  --outdir=<path>  dir in which to save plots [default: /tmp]
"""

import os

from docopt import docopt
import matplotlib.pyplot as plt


# Get args.
args = docopt(__doc__)

# Build the outdir path.
in_path = os.path.expanduser(args['<indir>'])
in_path_last_dir = in_path.split('/')[-1]
out_path = os.path.join(args['--outdir'], '%s-analysis' % in_path_last_dir)
if not os.path.exists(out_path):
  os.makedirs(out_path)

# Get all filenames in specified dir and build outdir.
filenames = os.listdir(in_path)

# Parse all filenames (ala frame_02955_thr_99_ste_79_mil_100300_odo_2354.png).
parsed_data = []
for filename in filenames:
  if 'lidar' in filename:
    continue
  if 'frame' not in filename:
    continue
  if 'lidar' in filename:
    continue
  split_filename = filename.split('_')
  data = {
    'filename': filename,
    'frame': split_filename[1],
    'throttle': split_filename[3],
    'steering': split_filename[5],
    'millis': split_filename[7],
    'odometer': split_filename[9].split('.')[0],
  }
  parsed_data.append(data)

# Setup x-values (time in seconds).
x_values = [int(d['millis']) / 1000. for d in parsed_data]

# Plot each parameter.
labels = [
  ('steering', 'blue'),
  ('throttle', 'red'),
  ('odometer', 'green'),
  ('frame', 'purple'),
]
for param, color in labels:
  y_values = [d[param] for d in parsed_data]
  figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
  axes.plot(x_values, y_values, marker='.', color=color)
  plt.xlabel('time (s)')
  plt.ylabel(param)
  plt.title('%s vs time, %s' % (param, in_path))
  save_path = os.path.join(out_path, '%s.png' % param)
  print 'creating %s' % save_path
  figure.savefig(save_path)

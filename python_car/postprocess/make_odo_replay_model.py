"""Creates an odo replay 'model' from training images.

Usage:
  make_odo_replay_model.py <image-dirs>...
"""

import os
import json
import time

from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt


# Parse args and get all filenames.
args = docopt(__doc__)
image_dirs = args['<image-dirs>']
all_filenames = []
for image_dir in image_dirs:
  all_filenames.extend(os.listdir(image_dir))
dated_dir = time.strftime('%Y_%m_%d__%I_%M_%S_%p')

# Build the out path.
base_out_dir = '/tmp/odo-replay-models'
out_dir = os.path.join(base_out_dir, dated_dir)
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
print 'saving to %s' % out_dir


data = {}
print '%s total files' % len(all_filenames)
for filename in all_filenames:
  if '.DS_Store' in filename:
    continue
  if 'lidar' in filename:
    print 'error!'
  # Filenames look like: frame_04162_thr_107_ste_92_mil_141348_odo_3515.png
  _, frame, _, throttle, _, steering, _, mil, _, odo = filename.split('_')
  throttle = int(throttle)
  steering = int(steering)
  if '.' in odo:
    odo = int(odo.split('.')[0])
  # Store data, keyed by odo.
  if odo not in data:
    data[odo] = {
      'throttles': [throttle],
      'steerings': [steering],
    }
  else:
    data[odo]['throttles'].append(throttle)
    data[odo]['steerings'].append(steering)

# Post process to find averages and stdevs.
for key in data.keys():
  data[key]['samples'] = len(data[key]['throttles'])
  data[key]['average_throttle'] = np.mean(data[key]['throttles'])
  data[key]['average_steering'] = np.mean(data[key]['steerings'])
  data[key]['stdev_throttle'] = np.std(data[key]['throttles'])
  data[key]['stdev_steering'] = np.std(data[key]['steerings'])

# Save.
outpath = os.path.join(out_dir, 'odo-model.json')
with open(outpath, 'w') as outfile:
  outfile.write(json.dumps(data))

# Plot
odos = data.keys()
average_throttles = [data[k]['average_throttle'] for k in data.keys()]
average_steerings = [data[k]['average_steering'] for k in data.keys()]
stdev_throttles = [data[k]['stdev_throttle'] for k in data.keys()]
stdev_steerings = [data[k]['stdev_steering'] for k in data.keys()]
figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
axes.plot(odos, average_throttles, marker='.', color='red',
          label='average throttle')
axes.plot(odos, average_steerings, marker='.', color='blue',
          label='average steering')
axes.plot(odos, stdev_throttles, marker='.', color='green',
          label='stdev throttles')
axes.plot(odos, stdev_steerings, marker='.', color='purple',
          label='stdev steerings')
plt.xlabel('odo')
savepath = os.path.join(out_dir, 'plot.png')
plt.legend()
figure.savefig(savepath)

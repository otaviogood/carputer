"""Make a video from pngs with telemetry.

Usage:
  make_video.py <inpath> [--outdir=<path>]

Options:
  --outdir=<path>  dir in which to save video [default: /tmp]
"""

import os
import shutil
import sys

from docopt import docopt
import envoy
from PIL import Image
from PIL import ImageDraw


# Read args.
args = docopt(__doc__)

# Create a temp path for images with telemetry.
tmp_dir = '/tmp/video-source-images'
if os.path.exists(tmp_dir):
  shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)

# Build the outdir path.
in_path = os.path.expanduser(args['<inpath>'])
in_path_last_dir = in_path.split('/')[-1]
out_path = os.path.join(args['--outdir'], '%s-analysis' % in_path_last_dir)
if not os.path.exists(out_path):
  os.makedirs(out_path)

# Write telemetry on each image.
inpath = os.path.expanduser(args['<inpath>'])
file_count = len([f for f in os.listdir(inpath) if 'lidar' not in f])
green = (41, 153, 82)
red = (179, 46, 46)
i = 0
for filename in os.listdir(inpath):
  if 'frame' not in filename:
    continue
  if 'lidar' in filename:
    continue
  # Get telemetry.
  steering = int(float(filename.split('_')[5]))
  throttle = int(float(filename.split('_')[3]))
  odometer = filename.split('_')[9].split('.')[0]
  # Draw on the image.
  image = Image.open(os.path.join(inpath, filename))
  drawing = ImageDraw.Draw(image)
  drawing.rectangle([(0, 90), (30, 120)], (255, 255, 255))
  drawing.text((0, 90), 's %s' % steering, green)
  drawing.text((0, 100), 't %s' % throttle, green)
  # Change the odo color to watch for resets.
  if int(odometer) < 5:
    odometer_color = red
  else:
    odometer_color = green
  drawing.text((0, 110), odometer, odometer_color)
  image.save(os.path.join(tmp_dir, filename))
  i += 1
  if i % 100 == 0 or i == file_count:
    sys.stdout.write('\rwriting telemtry.. %0.1f%%' % (100. * i / file_count))
    sys.stdout.flush()

# Setup path to the images with telemetry.
full_path = os.path.join(tmp_dir, 'frame_*.png')

# Setup output path to video.
output_video_path = os.path.join(out_path, 'telemetry.mp4')

# Run ffmpeg.
command = ("""ffmpeg
           -framerate 30/1
           -pattern_type glob -i '%s'
           -c:v libx264
           -r 30
           -pix_fmt yuv420p
           -y
           %s""" % (full_path, output_video_path))
response = envoy.run(command)
print '\nsaving video to %s' % output_video_path

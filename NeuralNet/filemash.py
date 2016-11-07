"""Turns folders of training data into np arrays.

Usage:
  filemash.py <folders>... [--outdir=<path>]

Options:
  --outdir=<path>  where to save output npy files [default: ~/training-data]

Examples:
  python filemash.py /my/training/data ~/my/other/data
"""

import urllib2
import glob
import os.path
import os
import math
import random
from StringIO import StringIO
from docopt import docopt
import numpy as np
from PIL import Image
import cv2
import Warp
import numpy as np
import manual_throttle_map
import config

# Parse args.
args = docopt(__doc__)
all_folders = args['<folders>']

numTestImages = 384 * 2

lamecount =0
def ReadPNG(source, targetWidth, targetHeight, warp):
    global lamecount
    try:
        pngfile = Image.open(source)
        pngfile = pngfile.resize((targetWidth, targetHeight), Image.BILINEAR)
        pngfile = pngfile.crop((0, 0, targetWidth, targetHeight))
    except:
        print "failed to read file: " + source
        return None
    if warp:
        Warp.RandRects(pngfile)
        pngfile = Warp.WhiteUnbalance(pngfile)
    # pngfile.save("test" + str(lamecount) + ".png")
    lamecount+=1
    pixRGB = np.array(pngfile)
    ret = np.empty((targetHeight, targetWidth, 3), dtype=np.uint8)
    ret[:, :, 0] = pixRGB[:, :, 0]
    ret[:, :, 1] = pixRGB[:, :, 1]
    ret[:, :, 2] = pixRGB[:, :, 2]
    return ret


def read_lidar(filename):
    lidar_img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    lidar_img = np.squeeze(lidar_img).astype(np.float32)
    lidar_img /= 255.
    assert lidar_img.shape == (256,)
       
    return lidar_img

# http://www.iquilezles.org/apps/graphtoy/
# round((log(abs(x)+1))/(log(2))*(x)/(abs(x)))


def do_log_mapping_to_buckets(a):
    return int(round(math.copysign(math.log(abs(a) + 1, 2.0), a))) + 7

probability_drop = 0.3

if __name__ == '__main__':
    look_ahead = 3
    # Load all pngs and find filenames.
    allPNGs = []
    allLidars = []
    for folder in all_folders:
        filepaths = [os.path.join(folder, f) for f in os.listdir(folder)]
        filepaths = filter(lambda name: 'lidar' not in os.path.basename(name), filepaths)

        filepaths = filter(lambda name: ('ste_90' not in os.path.basename(name)) or np.random.binomial(1, probability_drop),
                            filepaths)
        # Cut off 2 seconds from the start and end and append to main list.
        allPNGs.extend(sorted(filepaths)[60:-60])

        print str(len(allPNGs)) +  "    " + str(len(allLidars))
        #assert len(allPNGs) == len(allLidars)

    allNames = [name[name.find("frame_"):] for name in allPNGs]

    all_odos = []
    all_vels = []
    all_formatted_pngs = []
    all_lidars = []
    all_groundtruth_steer = []
    all_groundtruth_throttle = []
    last_odo = 0
    last_millis = 0
    for i in xrange(len(allNames) - look_ahead):
        name = allNames[i]
        name_ahead = allNames[i + look_ahead]
        s = name.split("_")
        s_ahead = name_ahead.split("_")
        temp_odo = int(s[9].split(".")[0])
        last_reset = 0
        # for j in odo_resets:
        #     if j > temp_odo:
        #         break
        #     last_reset = j
        # if last_reset == 0: continue

        # load odometer millisecond marks and convert to speed.
        millis = float(s[7])
        vel = 0
        if temp_odo != last_odo:
            if millis - last_millis == 0:
                vel = 0
            else:
                vel = (temp_odo - last_odo) / (millis - last_millis)
            if (last_millis == 0) and (last_odo == 0):
                vel = 0
            if (temp_odo - last_odo > 50) or (last_odo >= temp_odo):
                vel = 0
            last_odo = temp_odo
            last_millis = millis

        throttle = int(float(s_ahead[3]))
        log_throttle = manual_throttle_map.to_throttle_buckets(throttle) #do_log_mapping_to_buckets(throttle - 90)
        steer = int(float(s_ahead[5]))
        log_steer = do_log_mapping_to_buckets(steer - 90)

        # generate x times more training data than we actually have. Each image
        # will be randomly mangled.
        iters = 1
        if i >= len(allNames) - numTestImages - look_ahead:
            iters = 1
        for j in xrange(iters):
            # only warp training data, not test.
            png = ReadPNG(allPNGs[i], 128, 128, i < len(
                allNames) - numTestImages - look_ahead)
            all_formatted_pngs.append(png.flatten())
            all_vels.append(vel * 10.0)  # arbitrary range units. hurray!
            all_odos.append((temp_odo - last_reset) / 1000.0)

            all_groundtruth_steer.append(log_steer)
            all_groundtruth_throttle.append(log_throttle)
        if ((int(s[1]) % 1024) == 1023):
            print s[1] + "    " + str(steer) + "    " + str(log_steer) + "    " + str(log_throttle)

    if config.use_median_filter_throttle:
        import numpy as np
        
        def medfilt (x, k):
            """Apply a length-k median filter to a 1D array x.
            Boundaries are extended by repeating endpoints.
            """
            assert k % 2 == 1, "Median filter length must be odd."
            assert x.ndim == 1, "Input must be one-dimensional."
            k2 = (k - 1) // 2
            y = np.zeros ((len (x), k), dtype=x.dtype)
            y[:,k2] = x
            for i in range (k2):
                j = k2 - i
                y[j:,i] = x[:-j]
                y[:j,i] = x[0]
                y[:-j,-(i+1)] = x[j:]
                y[-j:,-(i+1)] = x[-1]
            return np.median (y, axis=1)

        all_groundtruth_throttle = medfilt(all_groundtruth_throttle, 5)

    # Save data.
    outpath = os.path.expanduser(args['--outdir'])
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    data = (
        ("picArray", np.array(all_formatted_pngs)),
        ("gtArray", np.array(all_groundtruth_steer)),
        ("gtThrottlesArray", np.array(all_groundtruth_throttle)),
        ("odoArray", np.array(all_odos)),
        ("velArray", np.array(all_vels)),
    )
    for d in data:
        np.save(os.path.join(outpath, d[0]), d[1])
    print 'processed %s images' % len(allNames)
    print 'data saved to %s' % outpath

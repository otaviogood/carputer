"""Turns folders of training data into np arrays.

Usage:
  filemash.py [<folders>...] [--outdir=<path>] [--gen_test] [--gen_gan]

Options:
  --outdir=<path>  where to save output npy files [default: ~/training-data]
  --gen_test  generate test instead of training data
  --gen_gan  generate GAN data instead of training data

Examples:
  python filemash.py /my/training/data ~/my/other/data
"""

import os.path
import math
from docopt import docopt
from PIL import Image
import Warp
import numpy as np
import collections

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config

# Parse args.
args = docopt(__doc__)
all_folders = args['<folders>']

lamecount =0
def ReadPNG(source, targetWidth, targetHeight, train_or_test_or_gan):
    global lamecount
    try:
        pngfile = Image.open(source)
        pngfile = pngfile.resize((targetWidth, targetHeight), Image.BILINEAR)
        pngfile = pngfile.crop((0, 0, targetWidth, targetHeight))
        # if (random.random() < 0.5):
        #     pngfile = pngfile.filter(ImageFilter.GaussianBlur(radius=random.random()*5.0+1.5))
    except:
        print "failed to read file: " + source
        return None
    if train_or_test_or_gan != 1:
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

def is_finite(x):
    return not math.isnan(x) and not math.isinf(x)

def ParseGoodFloat(s):
    try:
        a = float(s)
        if is_finite(a):
            return float(a)
        else:
            return 0.0
    except ValueError:
        return 0.0


# Fix places where our crappy remote control is dropping signal. Hacky.
def FixBadSignal(arr, index):
    if index <= 1: return False
    if index >= len(arr) - 2: return False
    a = arr[index - 1]
    b = arr[index]
    c = arr[index + 1]
    da = abs(b-a)
    db = abs(c-b)
    if (da > 45) and (db > 45) and (b < a) and (b < c):
        arr[index] = arr[index - 1]
        print("fixed: " + str(index) + "  " + str(a) + " " + str(b) + " " + str(c))

def FixVeryBadSignal(arr, index):
    if index <= 2: return False
    if index >= len(arr) - 3: return False
    a = arr[index - 1]
    b = arr[index]
    c = arr[index + 1]
    d = arr[index + 2]
    da = abs(b-a)
    db = abs(d-c)
    if (da > 45) and (db > 45) and (b < a) and (c < d):
        arr[index] = arr[index - 1]
        arr[index + 1] = arr[index + 2]
        print("fixed bad: " + str(index) + "  " + str(a) + " " + str(b) + " " + str(c) + " " + str(d))


# True for training data generation, False for test data generation
def GenNumpyFiles(allPNGs, train_or_test_or_gan, slice=None, telemetry=None, do_medfilt=None):

    allNames = [name[name.find("frame_"):] for name in allPNGs]

    processed_pngs = []
    all_steering = []
    all_throttle = []
    all_odos = []
    all_vels = []
    last_odo = 0
    last_millis = 0
    c = collections.Counter()

    for i in xrange(len(allNames)):
        name = allNames[i]
        s = name.split("_")

        frame = ParseGoodFloat(s[1])

        if i  == 0:
            print name,s

        if ((i % 1024) == 1023):
            print s

        # only warp training data, not test.
        png = ReadPNG(allPNGs[i], 128, 128, train_or_test_or_gan)
        processed_pngs.append(png.flatten())

        all_steering.append(ParseGoodFloat(s[5]))
        all_throttle.append(ParseGoodFloat(s[3]))

        temp_odo = int(s[9].split(".")[0])

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

        # if config.use_throttle_manual_map:
        #     log_throttle = manual_throttle_map.to_throttle_buckets(throttle)
        # else:
        #     log_throttle = do_log_mapping_to_buckets(throttle - 90)

        # steer = int(float(s_ahead[5]))
        # log_steer = do_log_mapping_to_buckets(steer - 90)

        all_odos.append(temp_odo)
        all_vels.append(vel)

    # Fix places where our crappy remote control is dropping signal. Hacky.
    for i in xrange(len(allNames)):
        FixBadSignal(all_throttle, i)
        FixVeryBadSignal(all_throttle, i)
        FixBadSignal(all_steering, i)
        FixVeryBadSignal(all_steering, i)
    for i in xrange(len(allNames)):
        if i > 2:
            delta = abs(int(all_throttle[i-1]) - int(all_throttle[i]))
            c.update({delta : 1})
    print c

    # Save data.
    outpath = os.path.expanduser(args['--outdir'])
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    mode = ("train_", "test_", "gan_")[train_or_test_or_gan]

    data = [
        (mode + "pic_array", np.array(processed_pngs)),
        (mode + "steer_array", np.array(all_steering)),
        (mode + "throttle_array", np.array(all_throttle)),
        (mode + "odo_array", np.array(all_odos)),
        (mode + "vel_array", np.array(all_vels)),
    ]

    for d in data:
        np.save(os.path.join(outpath, d[0]), d[1])
    print 'processed %s images (%d outputs)' % (len(allNames), len(processed_pngs))
    print 'data saved to %s' % outpath

if __name__ == '__main__':
    np.random.seed(1)

    # Load all pngs and find filenames.
    allPNGs = []

    if len(all_folders) == 0:
        all_folders = [config.load('last_record_dir')]

    for folder in all_folders:
        filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if ('.png' in f.lower() or '.jpg' in f.lower()) and (f[0] != '.')]

        allPNGs.extend(sorted(filepaths))

        print str(len(filepaths))

    train_or_test_or_gan = 0
    if args['--gen_test']: train_or_test_or_gan = 1
    elif args['--gen_gan']: train_or_test_or_gan = 2

    print ("Generating TRAINING data.", "Generating TEST data", "Generating GAN data")[train_or_test_or_gan]
    GenNumpyFiles(allPNGs, train_or_test_or_gan)

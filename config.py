import math

# This file to contain all the config variables shared across modules

use_throttle_manual_map = True
use_median_filter_throttle = False
# floating point 0 or 1 to multiply against odo value
use_odometer = 1.0
# The laptop has very little GPU memory, so we have to scale down to run on it.
running_on_laptop = True
do_flip_augmentation = False

# Should tensorflow use the GPU
should_use_gpu = 0

camera_id = 0
# Uncomment the following line to specify the path of the trained mdodel, or put the uncommented line in local_config.py.
# If no tf_checkpoint_file variable is found, the latest generated model is loaded.
#tf_checkpoint_file = "/Users/otaviogood/convnet02-results/2016_11_06__04_48_13_PM/model.ckpt" 


# http://www.iquilezles.org/apps/graphtoy/
# round((log(abs(x)+1))/(log(2))*(x)/(abs(x)))

def do_log_mapping_to_buckets(a):
    return int(round(math.copysign(math.log(abs(a) + 1, 2.0), a))) + 7


# JSON cache of local settings (path to latest training data folder, model checkpoint, ...)

# Loads a setting
def load(what):
	import json
	try:
		js = json.load(open('dyn_config.json','r'))
	except:
		js = dict()

	if not js.has_key(what): return None
	return js[what]

# Stores a setting
def store(what, val):
	import json
	try:
		js = json.load(open('dyn_config.json','r'))
	except:
		js = dict()

	js[what] = val

	with open('dyn_config.json', 'w') as fp:
		json.dump(js, fp, indent=4)


# Create local_config.py to override local settings
try :
	from local_config import *
except:
	pass


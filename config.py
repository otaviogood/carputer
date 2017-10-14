import math

# This file to contain all the config variables shared across modules

# This is time time difference in frames used to calc the speed value. If it
# takes more than this many frames to get a tick, the speed will be 0.
odo_delta = 6
# The laptop has very little GPU memory, so we have to scale down to run on it.
running_on_laptop = True

# Should tensorflow use the GPU - only for the car, this var not used for training.
should_use_gpu = 0

camera_id = 1 
# Uncomment the following line to specify the path of the trained mdodel, or put the uncommented line in local_config.py.
# If no tf_checkpoint_file variable is found, the latest generated model is loaded.
#tf_checkpoint_file = "/Users/otaviogood/convnet02-results/2016_11_06__04_48_13_PM/model.ckpt" 

# Either alexnet or lstm. Use lower case.
neural_net_mode = 'alexnet'

# Neural net input params for image sizes
width = 128
height = 128
width_small = 16
height_small = 16
img_channels = 3


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


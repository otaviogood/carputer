
# This file to contain all the config variables shared across modules

use_throttle_manual_map = True
use_median_filter_throttle = False
split_softmax = False

camera_id = 0


# Create local_config.py to have local settings
try :
	from local_config import *
except:
	pass


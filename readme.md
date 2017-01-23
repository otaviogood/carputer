
# Carputer


## Recording pipline

0. activate the virtualenv: `source /path/to/venv/bin/activate`
0. run a script to drive and record training data: `python main_car.py record` --
this will let you have manual control over the car
and save out recordings when you flip the switch


## Run autonomously

0. activate the virtualenv: `source /path/to/venv/bin/activate`
0. run a script to let tensorflow drive: `python main_car.py tf` --
when you flip the switch, you will lose manual control
and the car will attempt to drive on its own
0. for autonomous kill switch: pull throttle and turn the steering wheel
0. to revive autonomous mode, hit the channel 3 button (near the trigger)


## Training pipline

0. convert images to np arrays: `python NeuralNet/filemash.py /path/to/data` (Can be multiple paths)
0. train a model: `python NeuralNet/convnet02.py`
0. use this model to drive the car (see above)


## Analysis

* for training info, see `debug.html` -- reminder: < 7 is right, > 7 is left
* run `analysis/make_video.py` for debug videos
* use `analysis/plot_vs_time.py` to view telemetry data


### Hardware TODOs

- [ ] Make the car drive slower. Smaller wheels? Smaller batteries?


### Software TODOs

- [ ] Debug crowded autonomous runs with lots of people standing around throwing off the neural net
- [ ] Fix the kill switch so it locks brakes instead of coasting. Just a logic bug I think.
- [ ] Look into remote SSH type software so we don't have to keep popping open the car.

### Hardware setup
Updates - We no longer use the IMUs and we're no longer trying to run the NVidia TX1 computer. Macbook works better.
![Wiring diagram](https://github.com/otaviogood/carputer/blob/master/CarDiagram.jpg "Wiring diagram")

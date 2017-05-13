import tensorflow as tf
import math
import numpy as np
import random
import os

import config
import convnetshared1 as convshared

def do_log_mapping_to_buckets(a):
    return int(round(math.copysign(math.log(abs(a) + 1, 2.0), a))) + 7

class TrainingData:
    def __init__(self):
        pass

    @classmethod
    def fromfilename(cls, prefix, indir):
        # return cls(open(name, 'rb'))
        obj = cls()
        # Load the filemashed data.
        inpath = os.path.expanduser(indir)
        obj.pic_array = np.load(inpath + "/" + prefix + "_pic_array.npy")
        obj.steer_array = np.load(inpath + "/" + prefix + "_steer_array.npy")
        obj.throttle_array = np.load(inpath + "/" + prefix + "_throttle_array.npy")
        odo_array = np.load(inpath + "/" + prefix + "_odo_array.npy")
        obj.vel_array = np.load(inpath + "/" + prefix + "_vel_array.npy")
        assert(len(obj.pic_array) == len(obj.steer_array) == len(obj.throttle_array) == len(odo_array) == len(obj.vel_array))
        size = len(obj.pic_array)

        obj.speed_array = np.zeros((size), dtype=np.float32)

        # Allocate one-hot arrays for labels
        obj.steer_onehot_array = np.zeros((size, convshared.NNModel.max_log_outs), dtype=np.float32)
        obj.throttle_onehot_array = np.zeros((size, convshared.NNModel.max_log_outs), dtype=np.float32)
        # obj.vel_onehot_array = np.zeros((size, config.latlon_buckets), dtype=np.float32)
        # obj.odo_onehot_array = np.zeros((size, config.latlon_buckets), dtype=np.float32)
        # obj.pulse_onehot_array = np.zeros((size, convshared.numPulses), dtype=np.float32)

        obj.steer_array -= 90
        obj.throttle_array -= 90
        # Convert to one-hots
        for i in xrange(size):
            # pos = int(min(0.9999, max(0.0, ((obj.steer_array[i] / 2.0) + 0.5))) * convshared.NNModel.max_log_outs)
            pos = do_log_mapping_to_buckets(obj.steer_array[i])
            obj.steer_onehot_array[i, pos] = 1.0

            # pos = int(throttle_array[i] * (convshared.NNModel.max_log_outs - 1))
            pos = do_log_mapping_to_buckets(obj.throttle_array[i])
            obj.throttle_onehot_array[i, pos] = 1.0

        return obj

    @classmethod
    def FromRealLife(cls, image, odo_ticks, vel):
        obj = cls()
        image = image.ravel()  # flatten the shape of the tensor.
        image = image[np.newaxis]  # make a batch of size 1.
        obj.pic_array = image
        size = len(obj.pic_array)
        assert size == 1

        obj.vel_array = np.zeros((size), dtype=np.float32)
        obj.vel_array[0] = vel

        # Allocate one-hot arrays for labels
        obj.steer_array = np.zeros((size), dtype=np.float32)
        obj.throttle_array = np.zeros((size), dtype=np.float32)
        obj.steer_onehot_array = np.zeros((size, convshared.NNModel.max_log_outs), dtype=np.float32)
        obj.throttle_onehot_array = np.zeros((size, convshared.NNModel.max_log_outs), dtype=np.float32)

        return obj

    def GenRandomBatch(self):
        randIndexes = random.sample(xrange(self.NumSamples()), min(64, self.NumSamples()))
        batch_xs = [self.pic_array[index] for index in randIndexes]
        batch_xs_vel = [self.vel_array[index] for index in randIndexes]
        batch_ys_regress = [self.steer_array[index] for index in randIndexes]
        batch_ys_regress_throttle = [self.throttle_array[index] for index in randIndexes]
        batch_ys = [self.steer_onehot_array[index] for index in randIndexes]
        batch_ys_t = [self.throttle_onehot_array[index] for index in randIndexes]
        result = TrainingData()
        result.pic_array = np.array(batch_xs)
        result.vel_array = np.array(batch_xs_vel)
        result.steer_array = np.array(batch_ys_regress)
        result.throttle_array = np.array(batch_ys_regress_throttle)
        result.steer_onehot_array = np.array(batch_ys)
        result.throttle_onehot_array = np.array(batch_ys_t)
        return result

    def NumSamples(self):
        return len(self.pic_array)

    def TrimArray(self, size, skip):
        self.pic_array = self.pic_array[:size:skip]
        self.vel_array = self.vel_array[:size:skip]
        self.steer_array = self.steer_array[:size:skip]
        self.throttle_array = self.throttle_array[:size:skip]
        self.steer_onehot_array = self.steer_onehot_array[:size:skip]
        self.throttle_onehot_array = self.throttle_onehot_array[:size:skip]

    def FeedDict(self, net_model, dropout_keep = 1.0, train_modep = 0.0):
        return {
            net_model.in_image: self.pic_array,
            net_model.in_speed: self.vel_array,
            net_model.steering_regress_: self.steer_array,
            net_model.throttle_regress_: self.throttle_array,
            net_model.steering_: self.steer_onehot_array,
            net_model.throttle_: self.throttle_onehot_array,
            net_model.keep_prob: dropout_keep,
            net_model.train_mode: train_modep,
        }


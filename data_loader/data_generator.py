# coding=utf-8
import numpy as np


class DataGenerator(object):
    def __init__(self):
        # load data here
        self.features = np.zeros((10, 64))
        self.labels = np.ones((10, 1))

    def next(self):
        for i in range(10):
            idx = np.random.choice(10)
            yield (self.features[idx], self.labels[idx])

# -*- coding: utf-8 -*-
import numpy as np


class Sampler:
    def __init__(self):
        pass

    def sample_ev(self, instance, n_scenarios):
        demand = self.sample_stoch(instance, n_scenarios)
        return np.around(np.average(demand, axis=2))

    def sample_stoch(self, instance, n_scenarios):
        return np.around(np.absolute(np.random.normal(
            1,
            0,
            size=(instance.n_stations, instance.n_stations, n_scenarios))
        ))

# -*- coding: utf-8 -*-
import numpy as np
import utility.OD_matrix_gen as gen

class Sampler:
    def __init__(self):
        pass

    def sample_ev(self, instance, n_scenarios):
        demand = self.sample_stoch(instance, n_scenarios)
        return np.around(np.average(demand, axis=2))

    def sample_stoch(self, instance, n_scenarios, distribution='norm'):
        # return np.around(np.absolute(np.random.normal(
        #     1,
        #     0,
        #     size=(instance.n_stations, instance.n_stations, n_scenarios))
        # ))
        generator = gen.Generator(n_scenarios, instance.n_stations, distribution)
        return generator.scenario_res
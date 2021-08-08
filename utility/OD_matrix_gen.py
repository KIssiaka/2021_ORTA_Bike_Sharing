import numpy as np
import random

random.seed(42)

class Generator():
    def __init__(self, n_scenarios = 1000, n_stations = 22, distr = 'norm'):
        self.min_od_matrix = np.around(np.absolute(np.random.uniform(
                    0,3,
                    size=(n_stations, n_stations))
                ))
        self.max_od_matrix = np.around(np.absolute(np.random.uniform(
                    10, 15,
                    size=(n_stations, n_stations))
                ))

        self.mean_od_matrix = np.around(np.mean( np.array([ self.min_od_matrix, self.max_od_matrix ]), axis=0 ))
        self.std_od_matrix = np.around(np.std( np.array([ self.min_od_matrix, self.max_od_matrix ]), axis=0 ))

        #self.func_list = [self.normal_matrix(), self.uniform_matrix(), self.exponential_matrix()]

        if distr  == 'norm':
            self.scenario_arrays = [self.normal_matrix() for _ in range(n_scenarios)]
        elif distr == 'uni':
            self.scenario_arrays = [self.uniform_matrix() for _ in range(n_scenarios)]
        elif distr == 'expo':
            self.scenario_arrays = [self.exponential_matrix() for _ in range(n_scenarios)]
        else:
            raise ValueError("distribution must be one between 'norm', 'uni' and 'expo'")

        #self.scenario_arrays = [self.demand_distr for _ in range(n_scenarios)]
        self.scenario_res = np.stack(self.scenario_arrays, axis=2)

    def normal_matrix(self):
        return np.around(np.absolute(np.random.normal(self.mean_od_matrix, self.std_od_matrix )))

    def uniform_matrix(self):
        return np.around(np.random.uniform(self.min_od_matrix, self.max_od_matrix))

    def exponential_matrix(self):
        return np.around(np.random.exponential(self.mean_od_matrix))



        
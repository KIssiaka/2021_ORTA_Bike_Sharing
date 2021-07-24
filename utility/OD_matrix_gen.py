import numpy as np
import random

random.seed(42)
min_od_matrix = np.around(np.absolute(np.random.uniform(
            0,1,
            size=(22, 22))
        ))

max_od_matrix = np.around(np.absolute(np.random.uniform(
            3,5,
            size=(22, 22))
        ))

mean_od_matrix = np.around(np.mean( np.array([ min_od_matrix, max_od_matrix ]), axis=0 ))

std_od_matrix = np.around(np.std( np.array([ min_od_matrix, max_od_matrix ]), axis=0 ))

def normal_matrix():
    return np.around(np.absolute(np.random.normal(mean_od_matrix, std_od_matrix )))


def uniform_matrix():
    return np.around(np.random.uniform(min_od_matrix, max_od_matrix ))

def exponential_matrix():
    return np.around(np.random.uniform(mean_od_matrix))

n_scenarios = 5
scenario_arrays = [exponential_matrix() for _ in range(n_scenarios)]
scenario_res = np.stack(scenario_arrays, axis=2)

print(scenario_res)
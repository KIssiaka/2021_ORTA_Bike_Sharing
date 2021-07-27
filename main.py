#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
from matplotlib import rc_params
import numpy as np
from simulator.instance import Instance
from simulator.tester import Tester
from solver.BikeSharing import BikeSharing
from heuristic.simpleHeu import SimpleHeu
from solver.sampler import Sampler
from utility.plot_results import plot_comparison_hist

np.random.seed(0)

if __name__ == '__main__':
    log_name = "./logs/main.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )

    fp = open("./etc/bike_share_settings.json", 'r')
    sim_setting = json.load(fp)
    fp.close()

    sam = Sampler()

    inst = Instance(sim_setting)
    dict_data = inst.get_data()

    # Reward generation
    n_scenarios = 5
    """
    We create the demand matrixes by using a monte carlo distribution that chooses between:
    - exponential
    - uniform 
    - normal 
    distributions.
    We also pass the inst variable which contains all the input settings.
    """
    demand_matrix = sam.sample_stoch(
        inst,
        n_scenarios=n_scenarios
    )

    # heu = SimpleHeu()
    # of_heu, sol_heu, comp_time_heu = heu.solve(
    #     dict_data,
    #     reward,
    #     n_scenarios,
    # )
    # print(of_heu, sol_heu, comp_time_heu)

    prb = BikeSharing()
    of_exact, sol_exact, comp_time_exact = prb.solve(
        dict_data,
        demand_matrix,
        n_scenarios,
        verbose=True
    )
    print(sol_exact)

    ## EXpected value problem
    mean_demand_matrix = sam.sample_ev(
        inst,
        n_scenarios=n_scenarios
    )

    EV_prob = BikeSharing()
    of_EV, sol_EV, comp_time_exact = EV_prob.solve_EV(
        dict_data,
        mean_demand_matrix,
        verbose=True
    )

    # COMPARISON:
    test = Tester()
    n_scenarios = 1000

    demand_RP = sam.sample_stoch(
        inst,
        n_scenarios=n_scenarios
    )
    ris_RP = test.solve_second_stages(
        inst,
        sol_exact,
        n_scenarios,
        demand_RP
    )

    RP = np.average(ris_RP)


    demand_1 = sam.sample_stoch(
        inst,
        n_scenarios=n_scenarios
    )
    ris_EV = test.solve_second_stages(
        inst,
        sol_EV,
        n_scenarios,
        demand_1
    )

    EVV = np.average(ris_EV)

    # Wait and See solution: sample n times and compute the whole process for each of them
    WS_demand = sam.sample_stoch(
        inst,
        n_scenarios=n_scenarios
    )
    ris2, WS_sol = test.solve_wait_and_see(
        inst,
        n_scenarios,
        WS_demand
    )
    print("Wait and see solution: ", WS_sol)
    print("Expected result of EV solution: ", EVV)

    print("EVPI: ", WS_sol-RP)

    print("VSS:", EVV-RP)

    # plot_comparison_hist(
    #     [ris1, ris2],
    #     ["run1", "run2"],
    #     ['red', 'blue'],
    #     "profit", "occurencies"
    # )

    '''
    heu = SimpleHeu(2)
    of_heu, sol_heu, comp_time_heu = heu.solve(
        dict_data
    )
    print(of_heu, sol_heu, comp_time_heu)

    # printing results of a file
    file_output = open(
        "./results/exp_general_table.csv",
        "w"
    )
    file_output.write("method, of, sol, time\n")
    file_output.write("{}, {}, {}, {}\n".format(
        "heu", of_heu, sol_heu, comp_time_heu
    ))
    file_output.write("{}, {}, {}, {}\n".format(
        "exact", of_exact, sol_exact, comp_time_exact
    ))
    file_output.close()
    '''

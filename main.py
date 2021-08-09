#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
from matplotlib import rc_params
import numpy as np
import matplotlib.pyplot as plt
from simulator.instance import Instance
from simulator.tester import Tester
from solver.BikeSharing import BikeSharing
from heuristic.simpleHeu import SimpleHeu
from solver.sampler import Sampler
from utility.plot_results import plot_comparison_hist
import csv
import itertools
import string

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
    n_scenarios = 100
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
        n_scenarios=n_scenarios,
        distribution='norm'
    )
    test = Tester()
    # printing results of a file
    # file_output = open(
    #     "./results/grid_serach_penalty.csv",
    #     "w"
    # )
    # file_output.write("method, of, time, rho, alpha, sol\n")


    """
    here we compute the exact solution using the model written in the paper
    """
    print("EXACT METHOD")
    prb = BikeSharing()
    # of_exact, sol_exact, comp_time_exact = prb.solve(
    #     inst,
    #     demand_matrix,
    #     n_scenarios,
    # )
    # file_output.write("{}, {}, {}, {}, {}, {}\n".format(
    #     "exact", of_exact, comp_time_exact, 0, 0 ' '.join(str(e) for e in sol_exact)
    # ))

    """
    heuristic solution using the progressive hedging algorithm
    """
    print("HEURISTIC METHOD")
    heu = SimpleHeu()
    # of_heu, sol_heu, comp_time_heu, iter = heu.solve(
    #     inst,
    #     demand_matrix,
    #     n_scenarios
    # )

    # ########################################################
    # SEARCH FOR GOOD VALUES OF PENALTY AND ALPHA
    # ########################################################

    # for r in np.arange(10, 110, 30):
    #     for alpha in [1.1, 10, 100]:
    #         print("TRYING WITH ALPHA=: ", alpha, "AND PENALTY=", r)
    #         of_heu, sol_heu, comp_time_heu = heu.solve(
    #             inst,
    #             demand_matrix,
    #             n_scenarios,
    #             round(r, 1),
    #             round(alpha, 1)
    #         )
    #         file_output.write("{}, {}, {}, {}, {}, {}\n".format(
    #             "heu", of_heu, comp_time_heu, r, alpha ' '.join(str(e) for e in sol_heu)
    #         ))

    # file_output.close()

    # #########################################################
    # RECOURSE PROBLEM
    # #########################################################
    """
    Here we make a first stage decision and then and then, we solve the second stage solutions and average those in orrder to understand the profit for each of the scenarios based on the first assumption and the demand. 
    """
    # demand_RP = sam.sample_stoch(
    #     inst,
    #     n_scenarios=n_scenarios
    # )
    # test = Tester()

    #prb = BikeSharing()
    # of_exact, sol_exact, comp_time_exact = prb.solve(
    #     dict_data,
    #     demand_matrix,
    #     n_scenarios,
    #     verbose=True
    # )

    # ris_RP = test.solve_second_stages(
    #     inst,
    #     sol_exact,
    #     n_scenarios,
    #     demand_RP
    # )

    # RP = np.average(ris_RP)
    # print("\nRecourse problem solution (RP)", RP)
    
    # #########################################################
    # EXPECTED VALUE PROBLEM and the VALUE OF THE STOCHASTIC SOLUTION
    # #########################################################
    """
    The Scenarios are all blend together and only the average of them is considered. The resulting solution is clearly suboptimal but allows us to understand what is the actual loss in not considering stochasticity at all.
    """
    # of_EV, sol_EV, comp_time_EV = prb.solve_EV(
    #     dict_data,
    #     demand_matrix,
    #     n_scenarios,
    #     verbose=True
    # )

    # demand_EV = sam.sample_stoch(
    #     inst,
    #     n_scenarios=n_scenarios
    # )

    # ris_EV = test.solve_second_stages(
    #     inst,
    #     sol_EV,
    #     n_scenarios,
    #     demand_EV
    # )

    # EVV = np.average(ris_EV)
    
    # print("\nExpected result of EV solution (EVV): ", EVV)
    # print("\nValue of the Stochastic Solution (VSS = EVV-RP):", EVV-RP)


    # ##########################################################
    # WAIT AND SEE and the EXPECTED VALUE OF PERFECT INFORMATION
    # ##########################################################
    """
    Considering each of the scenarios separately and solving the first stage problems with full knowledge of the scenario is going to unfold. This is useful to understand what is the actual value of "knowing the future" and being able to adapt the first stage variables to the possible demand. 
    """
    # WS_demand = sam.sample_stoch(
    #     inst,
    #     n_scenarios=n_scenarios
    # )
    # ris2, WS_sol = test.solve_wait_and_see(
    #     inst,
    #     n_scenarios,
    #     WS_demand
    # )
    # print("\nWait and see solution (WS): ", WS_sol)
    # print("\nExpected value of perfect information (EVPI = WS-RP): ", WS_sol-RP)

    # ##########################################################
    # IN SAMPLE STABILITY ANALYSIS
    # ##########################################################
    """
    Here we analyize the in sample stability for our scenario tree generation.
    This requirement guarantees that whichever scenario tree we choose, the optimal value of the objective function reported by the model itself is (approximately) the same.
    We evaluate different solutions on different generated trees and if the results are similar (distribution is gaussian), we have in sample stability
    """
    test = Tester()
    n_scenarios = 100
    n_rep = 100
    print("IN SAMPLE STABILITY ANALYSIS")
    
    print("EXACT MODEL START...")
    in_samp_exact = test.in_sample_stability(prb, sam, inst, n_rep, n_scenarios)

    print("HEUTISTIC MODEL START...")
    in_samp_heu = test.in_sample_stability(heu, sam, inst, n_rep, n_scenarios)

    plot_comparison_hist(
        [in_samp_exact, in_samp_heu],
        ["exact", "heuristic"],
        ['red', 'blue'], "In Sample Stability",
        "profit", "occurencies"
    )

    # ##########################################################
    # OUT OF SAMPLE STABILITY ANALYSIS
    # ##########################################################
    """
    The out-of-sample stability test investigates whether a scenario generation method, with the selected sample  size,  creates  scenario  trees  that  provide  optimal  solutions  that  give  approximately  the  same optimal value as when using the true probability distribution.
    """
    n_scenarios_first = 100
    n_scenarios_second = 100
    n_rep = 100
    print("OUT OF SAMPLE STABILITY ANALYSIS")
    
    print("EXACT MODEL START...")
    out_samp_exact = test.out_of_sample_stability(prb, sam, inst, n_rep, n_scenarios_first, n_scenarios_second)
    
    print("HEUTISTIC MODEL START...")
    out_samp_heu = test.out_of_sample_stability(heu, sam, inst, n_rep, n_scenarios_first, n_scenarios_second)

    plot_comparison_hist(
        [out_samp_exact, out_samp_heu],
        ["exact", "heuristic"],
        ['red', 'blue'], "Out of Sample Stability",
        "profit", "occurencies"
    )

    # ##########################################################################################
    # write results of in and out of saple analysis so you don't have to run it again
    # ##########################################################################################
    # rows = zip(in_samp_exact, in_samp_heu, out_samp_exact, out_samp_heu)

    # with open("./results/stability.csv", "w") as f:
    #     writer = csv.writer(f)
    #     f.write("in_samp_exact, in_samp_heu, out_samp_exact, out_samp_heu\n")
    #     for row in rows:
    #         writer.writerow(row)


    # #######################################################################
    # ALTRO
    # #######################################################################
    # heu = SimpleHeu(2)
    # of_heu, sol_heu, comp_time_heu = heu.solve(
    #     dict_data
    # )
    # print(of_heu, sol_heu, comp_time_heu)

    # printing results of a file
    # file_output = open(
    #     "./results/exp_general_table.csv",
    #     "w"
    # )
    # file_output.write("method, of, sol, time\n")
    # file_output.write("{}, {}, {}, {}\n".format(
    #     "heu", of_heu, sol_heu, comp_time_heu
    # ))
    # file_output.write("{}, {}, {}, {}\n".format(
    #     "exact", of_exact, sol_exact, comp_time_exact
    # ))
    # file_output.close()
    

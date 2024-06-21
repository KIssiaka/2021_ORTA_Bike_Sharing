#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from simulator.instance import Instance
from simulator.tester import Tester
from solver.BssMonModel import BikeSharing
from heuristic.heuristics import ProgressiveHedging
from solver.sampler import Sampler
from utility.plot_results import plot_comparison_hist
from utility.plot_optimum_number_of_scenarios import plot_opt_num_scenarios
import csv
import sys
import getopt

np.random.seed(5)  

if __name__ == '__main__':
    log_name = "./logs/main_mymodel.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )

    fp = open("./BssMonModel/bss_settings.json", 'r')
    sim_setting = json.load(fp)
    fp.close()

    sam = Sampler()

    inst = Instance(sim_setting)
    dict_data = inst.get_data()


    # Reward generation
    n_scenarios = 100
    distribution = "norm"

    help_str = 'BssMonModel/BSSmonmodel.py -n <n_scenarios> -d <distribution>'
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "hn:d:", ["n_scenarios=", "distribution="])
    except getopt.GetoptError:
        print(help_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_str)
            sys.exit()
        elif opt in ("-n", "--n_scenarios"):
            n_scenarios = int(arg)
        elif opt in ("-d", "--distribution"):
            if (arg in ("norm", "uni", "expo")):
                distribution = arg
            else:
                print("Choose a distribution among norm, uni and expo")
                sys.exit()

    demand_matrix = sam.sample_stoch(
        inst,
        n_scenarios=n_scenarios,
        distribution=distribution
    )    
    def solve_exact():
            """
            Here we compute the exact solution with GUROBI. We then save results in a csv file.
            It contains a table with columns: 
            1. Method (Exact),
            2. Objective function value in euro
            3. Computational Time
            4. First stage solution (number of bikes to put per stations at the beginning of each day)
            """
            print("EXACT METHOD")
            file_output = open(
                "./BssMonModel/results/exact_method.csv",
                "w"
            )
            file_output.write("method, of, time, sol, y_u, y_l\n")
            prb = BikeSharing()
            of_exact, sol_exact, comp_time_exact, y_u, y_l= prb.solve(
                inst,
                demand_matrix,
                n_scenarios,
            )

            file_output.write("{}, {}, {}, {}, {}, {}\n".format(
            "exact", of_exact, comp_time_exact, ' '.join(str(e) for e in sol_exact), y_u, y_l)
        )
            print("To see results, open the file: BssMonModel/results/exact_method.csv", n_scenarios)
            """
            print(prb.solve(
                inst,
                demand_matrix,
                n_scenarios,
            ))"""


    def solve_online():

        print("SOLVE ONLINE")
        s0 = np.random.randint(low=0, high=30, size=dict_data['n_stations']).tolist()
        prb = BikeSharing()
        solution = prb.solve_online(s0, sim_setting)
        if solution:
            print("Solution found: ", solution)
        else:
            print("No solution found")

    while True:
        try:
            option = input("What do you want to do? (Options: solve_exact, solve_online, solve_heu, search_penalty_alpha, vss_evpi, test_in_sample, test_out_sample, opt_scenarios or exit)\n")
            if option == "solve_exact":
                solve_exact()
            elif option == "solve_online":
                solve_online()
            else:
                print("Unsupported operation, please check the command")
        except KeyboardInterrupt:
            break

    """while True:
        try:
            option = input("What do you want to do? (Options: solve_exact, solve_heu, search_penalty_alpha, vss_evpi, test_in_sample, test_out_sample, opt_scenarios or exit)\n")
            if (option == "solve_exact"):
                solve_exact()
            else:
                print("Unsupported operation, please check the command")
        except KeyboardInterrupt:
            break"""
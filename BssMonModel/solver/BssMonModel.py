# -*- coding: utf-8 -*-
import time
import logging
import numpy as np
from docplex.mp.model import Model

class BikeSharing():
    """Class representing the Exact Solver using CPLEX/DOcplex.
    It has methods:
       1. solve() to solve the deterministic problem optimally
       2. solve_EV() to solve the Expected Value problem
    """
    def __init__(self):
        pass

    def solve(
        self, instance, demand_matrix, n_scenarios, time_limit=None,
        gap=None, verbose=False
    ):
        dict_data = instance.get_data()
        n_stations = dict_data['n_stations']
        stations = range(n_stations)
        scenarios = range(n_scenarios)

        problem_name = "Mon modèle : Bike_Rebalancing"
        logging.info("{}".format(problem_name))

        model = Model(name=problem_name)
        In = []
        for i in range(2):
            In.append(model.integer_var_list(n_stations, lb=0, name=f"Stocks_à_l'instant_{i}"))

        r = model.integer_var_cube(n_stations, n_stations, n_scenarios, lb=0, name='R')
        s = model.integer_var_matrix(n_stations, n_scenarios, lb=0, name='S')
        y_l = model.integer_var_matrix(n_stations, n_scenarios, lb=0, name='Y_L')
        y_u = model.integer_var_matrix(n_stations, n_scenarios, lb=0, name='Y_U')
        beta = model.integer_var_cube(n_stations, n_stations, n_scenarios, lb=0, name='beta')    

        obj_funct = 0

        obj_funct += model.sum(
            (
                dict_data['Ruptures'][i] * model.sum(r[i, j, s] for j in stations) + y_l[i, s] + y_u[i, s]
            )
            for i in stations for s in scenarios
        ) / (n_scenarios + 0.0)
        model.minimize(obj_funct)

        initial_stocks = np.random.randint(low=0, high=30, size=n_stations).tolist()
        for i in stations:
            model.add_constraint(
                In[0][i] == initial_stocks[i],
                ctname=f"Inventaires de départ en {i}"
            )
        
        for i in stations:
            model.add_constraint(
                In[1][i] <= dict_data['station_cap'][i],
                ctname=f"station_bike_limit_{i}"
            )
            
        for s in scenarios:
            model.add_constraint(
                model.sum(y_l[i, s] for i in stations) == model.sum(y_u[i, s] for i in stations),
                ctname=f"Rebalancement_des_flux_dans_le_scénario_{s}"
            )

        for i in stations:
            for s in scenarios:
                model.add_constraint(
                    y_u[i, s] <= dict_data['station_cap'][i] - In[0][i] 
                    #+ (model.sum(demand_matrix[i, j, s] for j in stations) - model.sum(demand_matrix[j, i, s] for j in stations)),
                    ,ctname=f"Réapprovisionnement_en_fonction_de_la_capacité_résiduelle_de_{i}_scénario_{s}"
                )
                
        for i in stations:
            for s in scenarios:
                model.add_constraint(
                    y_l[i, s] <= In[0][i]
                     # - (model.sum(demand_matrix[i, j, s] for j in stations) - model.sum(demand_matrix[j, i, s] for j in stations)),
                    ,ctname=f"Retrait_en_fonction_du_stock_résiduel_station_{i}_scenario_{s}"
                )
        
        for i in stations:
            for s in scenarios:
                model.add_constraint(
                    #model.sum(r[i, j, s] for j in stations) +
                    In[1][i] == In[0][i] + (y_u[i, s] - y_l[i, s])
                    - (model.sum(beta[i, j, s] for j in stations) - model.sum(beta[j, i, s] for j in stations)),
                    ctname=f"Bilan des stocks dans la station_{i} dans le scénario {s}"
                )

        for i in stations:
            for j in stations:
                for s in scenarios:
                    model.add_constraint(
                        beta[i, j, s] == demand_matrix[i, j, s] - r[i, j, s],
                        ctname=f"beta_{i}_{j}_{s}"
                    )

        if gap:
            model.parameters.mip.tolerances.mipgap = gap
        if time_limit:
            model.parameters.timelimit = time_limit
        model.parameters.threads = 1
        model.context.solver.log_output = verbose


        """for i in stations:
            for s in scenarios:
                print(f"Variable Y_U_{i}_{s}: lower bound = {y_u[i, s].lb}, upper bound = {y_u[i, s].ub}")"""
        

        start = time.time()
        solution = model.solve(log_output=True)
        end = time.time()
        comp_time = end - start

        sol_y_u = np.zeros((n_stations, n_scenarios))
        sol_y_l = np.zeros((n_stations, n_scenarios))
        sol = [0] * dict_data['n_stations']
        of = -1
        if solution:
            for i in stations:
                sol[i] = In[1][i].solution_value
            
            for i in stations:
                for s in scenarios:
                    sol_y_u[i, s] = y_u[i, s].solution_value
                    sol_y_l[i, s] = y_l[i, s].solution_value
            of = model.objective_value

        for var in model.iter_variables():
            print(var.name, "=", var.solution_value)
        return of, sol, comp_time, sol_y_u, sol_y_l
        #return solution

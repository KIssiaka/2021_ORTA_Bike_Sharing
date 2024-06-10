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

        problem_name = "Bike_Sharing"
        logging.info("{}".format(problem_name))

        model = Model(name=problem_name)
        X = model.integer_var_list(n_stations, lb=0, name='X')
        I_plus = model.integer_var_matrix(n_stations, n_scenarios, lb=0, name='I+')
        I_minus = model.integer_var_cube(n_stations, n_stations, n_scenarios, lb=0, name='I-')
        O_plus = model.integer_var_matrix(n_stations, n_scenarios, lb=0, name='O+')
        O_minus = model.integer_var_matrix(n_stations, n_scenarios, lb=0, name='O-')
        T_plus = model.integer_var_matrix(n_stations, n_scenarios, lb=0, name='T+')
        T_minus = model.integer_var_matrix(n_stations, n_scenarios, lb=0, name='T-')
        tau = model.integer_var_cube(n_stations, n_stations, n_scenarios, lb=0, name='tau')
        beta = model.integer_var_cube(n_stations, n_stations, n_scenarios, lb=0, name='beta')
        rho = model.integer_var_cube(n_stations, n_stations, n_scenarios, lb=0, name='rho')

        obj_funct = dict_data["procurement_cost"] * model.sum(X[i] for i in stations)

        obj_funct += model.sum(
            (
                dict_data['stock_out_cost'][i] * model.sum(I_minus[i, j, s] for j in stations) +
                dict_data["time_waste_cost"][i] * O_minus[i, s] +
                model.sum(dict_data['trans_ship_cost'][i][j] * tau[i, j, s] for j in stations)
            )
            for i in stations for s in scenarios
        ) / (n_scenarios + 0.0)
        model.minimize(obj_funct)

        for i in stations:
            model.add_constraint(
                X[i] <= dict_data['station_cap'][i],
                ctname=f"station_bike_limit_{i}"
            )

        for i in stations:
            for j in stations:
                for s in scenarios:
                    model.add_constraint(
                        beta[i, j, s] == demand_matrix[i, j, s] - I_minus[i, j, s],
                        ctname=f"rented_bikes_number_{i}_{j}_{s}"
                    )

        for i in stations:
            for s in scenarios:
                model.add_constraint(
                    I_plus[i, s] - model.sum(I_minus[i, j, s] for j in stations) == X[i] - model.sum(demand_matrix[i, j, s] for j in stations),
                    ctname=f"surplus_shortage_balance_{i}_{s}"
                )

        for i in stations:
            for s in scenarios:
                model.add_constraint(
                    O_plus[i, s] - O_minus[i, s] == dict_data['station_cap'][i] - X[i] + model.sum(beta[i, j, s] for j in stations) - model.sum(beta[j, i, s] for j in stations),
                    ctname=f"residual_overflow_balance_{i}_{s}"
                )

        for i in stations:
            for s in scenarios:
                model.add_constraint(
                    model.sum(rho[i, j, s] for j in stations) == O_minus[i, s],
                    ctname=f"redir_bikes_eq_overflow_{i}_{s}"
                )

        for i in stations:
            for s in scenarios:
                model.add_constraint(
                    model.sum(rho[j, i, s] for j in stations) <= O_plus[i, s],
                    ctname=f"redir_bikes_not_resid_cap_{i}_{s}"
                )

        for i in stations:
            for s in scenarios:
                model.add_constraint(
                    T_plus[i, s] - T_minus[i, s] == dict_data['station_cap'][i] - O_plus[i, s] + model.sum(rho[j, i, s] for j in stations) - X[i],
                    ctname=f"exceed_failure_balance_{i}_{s}"
                )

        for i in stations:
            for s in scenarios:
                model.add_constraint(
                    model.sum(tau[i, j, s] for j in stations) == T_plus[i, s],
                    ctname=f"tranship_equal_excess_{i}_{s}"
                )

        for i in stations:
            for s in scenarios:
                model.add_constraint(
                    model.sum(tau[j, i, s] for j in stations) <= T_minus[i, s],
                    ctname=f"tranship_equal_failure_{i}_{s}"
                )

        if gap:
            model.parameters.mip.tolerances.mipgap = gap
        if time_limit:
            model.parameters.timelimit = time_limit
        model.parameters.threads = 1
        model.context.solver.log_output= verbose

        start = time.time()
        solution = model.solve(log_output=verbose)
        end = time.time()
        comp_time = end - start

        sol = [0] * dict_data['n_stations']
        of = -1
        if solution:
            for i in stations:
                sol[i] = X[i].solution_value
            of = model.objective_value

        return of, sol, comp_time, None

    def solve_EV(
        self, instance, demand_matrix, time_limit=None,
        gap=None, verbose=False
    ):
        dict_data = instance.get_data()
        n_stations = dict_data['n_stations']
        stations = range(n_stations)

        problem_name = "Bike_Sharing_EV"
        logging.info("{}".format(problem_name))

        model = Model(name=problem_name)
        X = model.integer_var_list(n_stations, lb=0, name='X')
        I_plus = model.integer_var_list(n_stations, lb=0, name='I+')
        I_minus = model.integer_var_matrix(n_stations, n_stations, lb=0, name='I-')
        O_plus = model.integer_var_list(n_stations, lb=0, name='O+')
        O_minus = model.integer_var_list(n_stations, lb=0, name='O-')
        T_plus = model.integer_var_list(n_stations, lb=0, name='T+')
        T_minus = model.integer_var_list(n_stations, lb=0, name='T-')
        tau = model.integer_var_matrix(n_stations, n_stations, lb=0, name='tau')
        beta = model.integer_var_matrix(n_stations, n_stations, lb=0, name='beta')
        rho = model.integer_var_matrix(n_stations, n_stations, lb=0, name='rho')

        obj_funct = dict_data["procurement_cost"] * model.sum(X[i] for i in stations)

        obj_funct += model.sum(
            (
                dict_data['stock_out_cost'][i] * model.sum(I_minus[i, j] for j in stations) +
                dict_data["time_waste_cost"][i] * O_minus[i] +
                model.sum(dict_data['trans_ship_cost'][i][j] * tau[i, j] for j in stations)
            )
            for i in stations
        )
        model.minimize(obj_funct)

        for i in stations:
            model.add_constraint(
                X[i] <= dict_data['station_cap'][i],
                ctname=f"station_bike_limit_{i}"
            )

        for i in stations:
            for j in stations:
                model.add_constraint(
                    beta[i, j] == demand_matrix[i, j] - I_minus[i, j],
                    ctname=f"rented_bikes_number_{i}_{j}"
                )

        for i in stations:
            model.add_constraint(
                I_plus[i] - model.sum(I_minus[i, j] for j in stations) == X[i] - model.sum(demand_matrix[i, j] for j in stations),
                ctname=f"surplus_shortage_balance_{i}"
            )

        for i in stations:
            model.add_constraint(
                O_plus[i] - O_minus[i] == dict_data['station_cap'][i] - X[i] + model.sum(beta[i, j] for j in stations) - model.sum(beta[j, i] for j in stations),
                ctname=f"residual_overflow_balance_{i}"
            )

        for i in stations:
            model.add_constraint(
                model.sum(rho[i, j] for j in stations) == O_minus[i],
                ctname=f"redir_bikes_eq_overflow_{i}"
            )

        for i in stations:
            model.add_constraint(
                model.sum(rho[j, i] for j in stations) <= O_plus[i],
                ctname=f"redir_bikes_not_resid_cap_{i}"
            )

        for i in stations:
            model.add_constraint(
                T_plus[i] - T_minus[i] == dict_data['station_cap'][i] - O_plus[i] + model.sum(rho[j, i] for j in stations) - X[i],
                ctname=f"exceed_failure_balance_{i}"
            )

        for i in stations:
            model.add_constraint(
                model.sum(tau[i, j] for j in stations) == T_plus[i],
                ctname=f"tranship_equal_excess_{i}"
            )

        for i in stations:
            model.add_constraint(
                model.sum(tau[j, i] for j in stations) <= T_minus[i],
                ctname=f"tranship_equal_failure_{i}"
            )

        if gap:
            model.parameters.mip.tolerances.mipgap = gap
        if time_limit:
            model.parameters.timelimit = time_limit
        model.parameters.threads = 1
        model.context.solver.log_output= verbose

        start = time.time()
        solution = model.solve(log_output=verbose)
        end = time.time()
        comp_time = end - start

        sol = [0] * dict_data['n_stations']
        of = -1
        if solution:
            for i in stations:
                sol[i] = X[i].solution_value
            of = model.objective_value

        return of, sol, comp_time

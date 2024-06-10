# -*- coding: utf-8 -*-
import time
import numpy as np
from docplex.mp.model import Model

class ProgressiveHedging():
    """Class representing the PH heuristics method.
    It has two methods:
        1. DEP_solver() to solve the deterministic problem with Augmented Relaxation
        2. solve() is the actual Progressive Hedging algorithm
    """
    def __init__(self):
        pass

    def DEP_solver(self, dict_data, scenario_input, TGS, lambd=0, pen_rho=0, iteration=0):
        problem_name = "DEP"
        model = Model(problem_name)
        n_stations = dict_data['n_stations']
        stations = range(n_stations)
                
        ### Variables
        X = model.integer_var_list(
            n_stations,
            lb=0,
            name='X'
        )
        
        I_plus = model.integer_var_list(
            n_stations,
            lb=0,
            name='I+'
        )

        I_minus = model.integer_var_matrix(
            n_stations, n_stations,
            lb=0,
            name='I-'
        )

        O_plus = model.integer_var_list(
            n_stations,
            lb=0,
            name='O+'
        )

        O_minus = model.integer_var_list(
            n_stations,
            lb=0,
            name='O-'
        )

        T_plus = model.integer_var_list(
            n_stations,
            lb=0,
            name='T+'
        )

        T_minus = model.integer_var_list(
            n_stations,
            lb=0,
            name='T-'
        )


        tau = model.integer_var_matrix(
            n_stations, n_stations,
            lb=0,
            name='tau'
        )

        beta = model.integer_var_matrix(
            n_stations, n_stations,
            lb=0,
            name='beta'
        )

        rho = model.integer_var_matrix(
            n_stations, n_stations,
            lb=0,
            name='rho'
        )
        
        sol = [X[i] for i in stations]


        ## Objective Function
        obj_funct = dict_data["procurement_cost"] * model.sum(X[i] for i in stations)
        
        obj_funct += model.sum(
            (
                dict_data['stock_out_cost'][i]*model.sum(I_minus[i, j] for j in stations) +
                dict_data["time_waste_cost"][i]*O_minus[i] +
                model.sum(dict_data['trans_ship_cost'][i][j]*tau[i, j]  for j in stations)
            ) for i in stations
        )
        if iteration != 0:
            relax = np.dot(lambd.T, (np.array(sol) - TGS))
            penalty = (pen_rho / 2) * (np.dot((np.array(sol) - TGS), (np.array(sol) - TGS).T))
            
            obj_funct += relax
            obj_funct += penalty

        model.minimize(obj_funct)


        ### Constraints

        for i in stations:
            model.add_constraint(
                X[i] <= dict_data['station_cap'][i],
                f"station_bike_limit_{i}"
            )
            
        for i in stations:
            for j in stations:
                model.add_constraint(
                    beta[i, j] == scenario_input[i, j] - I_minus[i, j],
                    f"rented_bikes_number_{i}_{j}"
                )
            
        for i in stations:
            model.add_constraint(
                I_plus[i] - model.sum(I_minus[i, j] for j in stations) == X[i] - model.sum(scenario_input[i, j] for j in stations),
                f"surplus_shortage_balance_{i}"
            )

        for i in stations:
            model.add_constraint(
                O_plus[i] - O_minus[i] == dict_data['station_cap'][i] - X[i] + model.sum(beta[i, j] for j in stations) - model.sum(beta[j, i] for j in stations),
                f"residual_overflow_balance_{i}"
            )

        for i in stations:
            model.add_constraint(
                model.sum(rho[i, j] for j in stations) == O_minus[i],
                f"redir_bikes_eq_overflow_{i}"
            )

        for i in stations:
            model.add_constraint(
                model.sum(rho[j, i] for j in stations) <= O_plus[i],
                f"redir_bikes_not_resid_cap_{i}"
            )

        for i in stations:
            model.add_constraint(
                T_plus[i] - T_minus[i] == dict_data['station_cap'][i] - O_plus[i] + model.sum(rho[j, i] for j in stations) - X[i],
                f"exceed_failure_balance_{i}"
            )

        for i in stations:
            model.add_constraint(
                model.sum(tau[i, j] for j in stations) == T_plus[i],
                f"tranship_equal_excess_{i}"
            )

        for i in stations:
            model.add_constraint(
                model.sum(tau[j, i] for j in stations) <= T_minus[i],
                f"tranship_equal_failure_{i}"
            )

        model.set_log_output(None)
        solution = model.solve()

        sol = [0] * dict_data['n_stations']
        of = -1
        if solution:
            for i in stations:
                sol[i] = X[i].solution_value
            of = model.objective_value
        return of, np.array(sol)
    

    def solve(
        self, instance, scenarios, n_scenarios, rho=70, alpha=100
    ):
        ans = []
        of_array = []
        dict_data = instance.get_data()

        # temporary global solution (initialize the TGS for the first iteration)
        TGS = 0

        # max iterations
        maxiter = 100

        # iteration
        k = 0

        # lagrangian multiplier
        lam = np.zeros((n_scenarios, dict_data['n_stations']))
        start = time.time()
        # solve the base problem for each of the solutions 
        # (solve the problem for the first time in order to initialize the first stage solution at the zero-th iteration)
        # For each scenario, solve the mono-scenario problem
        for i, s in enumerate(np.rollaxis(scenarios, 2)):
            of, sol = self.DEP_solver(dict_data, s, TGS, lam[i], rho)
            of_array.append(of)
            ans.append(np.array(sol))
        x_s_arrays = ans
        
        # compute temporary global solution for first iteration
        TGS = np.average(x_s_arrays, axis=0).astype(int)
        lam = rho * (x_s_arrays - TGS)

        for k in range(1, maxiter + 1):

            if np.all(abs(x_s_arrays - TGS) == 0):
                break

            x_s_arrays = []
            of_array = []
            ans = []

            # solve monoscenario problems
            for i, s in enumerate(np.rollaxis(scenarios, 2)):
                of, sol = self.DEP_solver(dict_data, s, TGS, lam[i], rho, k)
                of_array.append(of)
                ans.append(np.array(sol))
            x_s_arrays = ans

            # compute temporary global solution for first iteration
            TGS = np.average(x_s_arrays, axis=0).astype(int)

            # update the multipliers
            lam = lam + rho * (x_s_arrays - TGS)
            rho = alpha * rho

        end = time.time()
        comp_time = end - start

        sol_x = TGS
        of = np.average(of_array, axis=0)
        return of, sol_x, comp_time, k

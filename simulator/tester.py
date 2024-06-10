# -*- coding: utf-8 -*-
import time
import numpy as np
from docplex.mp.model import Model

class Tester():
    """Class containing different useful methods:
        1. solve_second_stage() to solve the second stage problem
        2. solve_wait_and_see() to solve the WS problem used to compute the EVPI
        3. in_sample_stability() to analyze the in-sample stability of the scenario generation method
        4. out_of_sample_stability() to analyze the out-of-sample stability of the scenario generation method
    """
    def __init__(self):
        pass

    def solve_second_stages(self, 
        inst, sol, n_scenarios, demand_matrix
    ):
        ans = []
        dict_data = inst.get_data()
        obj_fs = 0
        n_stations = inst.n_stations
        stations = range(n_stations)
        for i in stations:
            obj_fs += dict_data["procurement_cost"] * sol[i]
        
        
        for s in range(n_scenarios):
            problem_name = "SecondStagePrb"
            model = Model(name=problem_name)
            
            ### Variables
            
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

            
            ### Objective Function
        
            obj_funct = model.sum(
                (
                    dict_data['stock_out_cost'][i] * model.sum(I_minus[i, j] for j in stations) +
                    dict_data["time_waste_cost"][i] * O_minus[i] +
                    model.sum(dict_data['trans_ship_cost'][i][j] * tau[i, j]  for j in stations)
                ) for i in stations
            )

            model.minimize(obj_funct)
            
            ### Constraints
            
            for i in stations:
                for j in stations:
                    model.add_constraint(
                        beta[i, j] == demand_matrix[i, j, s] - I_minus[i, j],
                        ctname=f"rented_bikes_number_{i}_{j}"
                    )
                
            for i in stations:
                model.add_constraint(
                    I_plus[i] - model.sum(I_minus[i, j] for j in stations) == sol[i] - model.sum(demand_matrix[i, j, s] for j in stations),
                    ctname=f"surplus_shortage_balance_{i}"
                )

            for i in stations:
                model.add_constraint(
                    O_plus[i] - O_minus[i] == dict_data['station_cap'][i] - sol[i] + model.sum(beta[i, j] for j in stations) - model.sum(beta[j, i] for j in stations),
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
                    T_plus[i] - T_minus[i] == dict_data['station_cap'][i] - O_plus[i] + model.sum(rho[j, i] for j in stations) - sol[i],
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

            model.parameters.threads = 1
            model.context.solver.log_output= False

            model.solve()
            ans.append(obj_fs + model.objective_value)

        return ans


    def solve_wait_and_see(self, 
        inst, n_scenarios, demand_matrix
    ):
        ans = []
        dict_data = inst.get_data()
        n_stations = inst.n_stations
        stations = range(n_stations)
        
        
        for s in range(n_scenarios):
            problem_name = "Wait_and_See_BikeShare"
            model = Model(name=problem_name)
            
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
            ### Objective Function
            obj_funct = dict_data["procurement_cost"] * model.sum(X[i] for i in stations)
            
            obj_funct += model.sum(
                (
                    dict_data['stock_out_cost'][i] * model.sum(I_minus[i, j] for j in stations) +
                    dict_data["time_waste_cost"][i] * O_minus[i] +
                    model.sum(dict_data['trans_ship_cost'][i][j] * tau[i, j]  for j in stations)
                ) for i in stations
            )

            model.minimize(obj_funct)
            
            ### Constraints
            
            for i in stations:
                model.add_constraint(
                    X[i] <= dict_data['station_cap'][i],
                    ctname=f"station_bike_limit_{i}"
                )

            for i in stations:
                for j in stations:
                    model.add_constraint(
                        beta[i, j] == demand_matrix[i, j, s] - I_minus[i, j],
                        ctname=f"rented_bikes_number_{i}_{j}"
                    )
                
            for i in stations:
                model.add_constraint(
                    I_plus[i] - model.sum(I_minus[i, j] for j in stations) == X[i] - model.sum(demand_matrix[i, j, s] for j in stations),
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

            model.parameters.threads = 1
            model.context.solver.log_output= False

            model.solve()
            ans.append(model.objective_value)

        WS = np.average(ans)
        return ans, WS


    def in_sample_stability(self, problem, sampler, instance, n_repetitions, n_scenarios_sol, distribution):
        ans = [0] * n_repetitions
        for i in range(n_repetitions):
            print("Scenario Tree: ", i)
            a = time.time()
            reward = sampler.sample_stoch(
                instance,
                n_scenarios=n_scenarios_sol,
                distribution=distribution
            )
            of, sol, comp_time, ite = problem.solve(
                instance,
                reward,
                n_scenarios_sol
            )
            b = time.time()
            print("Time spent:", b-a)
            ans[i] = of
        return ans
    
    def out_of_sample_stability(self, problem, sampler, instance, n_repetitions, n_scenarios_sol, n_scenarios_out):
        ans = [0] * n_repetitions
        for i in range(n_repetitions):
            print("Scenario Tree: ", i)
            a = time.time()
            reward = sampler.sample_stoch(
                instance,
                n_scenarios=n_scenarios_sol
            )
            of, sol, comp_time, ite = problem.solve(
                instance,
                reward,
                n_scenarios_sol
            )
            reward_out = sampler.sample_stoch(
                instance,
                n_scenarios=n_scenarios_out
            )
            profits = self.solve_second_stages(
                instance, sol,
                n_scenarios_out, reward_out
            )
            b = time.time()
            print("Time spent:", b-a)
            ans[i] = np.mean(profits)
            
        return ans

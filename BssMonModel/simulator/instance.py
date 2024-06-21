# -*- coding: utf-8 -*-
import logging
import numpy as np

np.random.seed(42)
class Instance():
    def __init__(self, sim_setting):

        logging.info("starting simulation...")
        
        self.n_stations = sim_setting['n_stations']

        #v_i
        self.I_0 = np.random.randint(low=0, high=30, size=self.n_stations).tolist()
        
        #v_i
        self.Ruptures = [sim_setting["Ruptures"]] * self.n_stations
        
        #w_i
        self.Surplus = [sim_setting['Surplus']] * self.n_stations
        
        #t_ij
        self.trans_ship_cost = [[sim_setting['trans_ship_cost'] for x in range(self.n_stations)] for y in range(self.n_stations)] 
        
        # k_i
        self.stations_cap = np.around(np.random.uniform(
                                sim_setting['station_max_cap'],
                                sim_setting['station_min_cap'],
                                size=self.n_stations
                                )
                            )

        logging.info(f"stations_number: {self.n_stations}")
        logging.info(f"inventaires_avant_rebalancement: {self.I_0}")
        logging.info(f"Ruptures: {self.Ruptures}")
        logging.info(f"Surplus: {self.Surplus}")
        logging.info(f"Transshipment_cost: {self.trans_ship_cost}")
        logging.info(f"stations_capacity: {self.stations_cap}")
        logging.info("simulation end")

    def get_data(self):
        logging.info("getting data from instance...")
        return {
            "inventaire_avant_rebalancement": self.I_0,
            "Ruptures": self.Ruptures,
            "Surplus": self.Surplus,
            "trans_ship_cost": self.trans_ship_cost,
            "n_stations": self.n_stations,
            "station_cap": self.stations_cap
        }

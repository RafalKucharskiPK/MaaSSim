################################################################################
# Module: main.py
# Description: Simulator object
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################
from MaaSSim.utils import get_config, load_G, save_config, prep_supply_and_demand, generate_demand, generate_vehicles, initialize_df  # simulator
from MaaSSim.simulators import simulate, simulate_parallel
from MaaSSim.decisions import f_platform_choice

import os
from dotmap import DotMap
import pandas as pd
import logging


def supply_demand():
    # explore various supply and demand setting with parallel simulations replicatrd 10 times,
    # each stored to separate .zip file (to be further merged with collect_results from utils)
    from MaaSSim.data_structures import structures as inData

    params = get_config('../../data/config/delft.json')  # load configuration

    params.paths.dumps = '../docs/experiments/sd'
    params.times.patience = 1200
    params.simTime = 4
    params.parallel.nThread = 4
    params.parallel.nReplications = 10

    params.paths.G = '../../data/graphs/delft.graphml'
    params.paths.skim = '../../data/graphs/delft.csv'

    inData = load_G(inData, params)  # load network graph

    space = DotMap()
    space.nP = list(range(50,1001,50))
    space.nV = list(range(20,201,20))
    print(dict(space))

    simulate_parallel(params = params, search_space = space)


def platforms():
    # play the market competition scenarios. You compete with platform of 20 vehs in fleet and charging 1/km.
    # You explore how to compete varying fleet from 5 to 40 vehicles and price ranges from .6 to 1.4
    # experiments are run sequentially and collect to ine csv
    # (error prone and using single thread, only for small experiments)
    print(os.getcwd())
    def experiment(inData, params, fare=0.8, fleet=20):
        params.nV = 20 + fleet  # vehicles
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0] = [1, 'Platform1', 30]
        inData.platforms.loc[1] = [fare, 'Platform2', 30]
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.vehicles.platform = [0] * 20 + [1] * fleet

        # inData.vehicles.platform = inData.vehicles.apply(lambda x: 1 if random.random()<0.5 else 0, axis = 1) # randomly 30% vehicles for first platform, 70% for 2nd
        inData.passengers.platforms = inData.passengers.apply(lambda x: [0, 1], axis=1)
        sim = simulate(params=params, inData=inData, print=False, f_platform_choice=f_platform_choice,
                       logger_level=logging.CRITICAL)
        ret = sim.res[0].veh_exp
        ret['platform'] = inData.vehicles.platform
        return ret[ret.platform == 1].ARRIVES_AT_DROPOFF.sum()
    from MaaSSim.data_structures import structures as inData

    params = get_config('../../data/config/platforms.json')  # load configuration

    params.paths.G = '../../data/graphs/delft.graphml'
    params.paths.skim = '../../data/graphs/delft.csv'

    inData = load_G(inData, params)  # load network graph

    params.nP = 200  # reuqests (and passengers)
    params.simTime = 4
    params.nD = 1
    ret = []
    for repl in range(10):
        for fleet in [5, 10, 15, 20, 25, 30, 40]:
            for fare in [0.6, 0.8, 1, 1.2, 1.4]:
                revenue = experiment(inData, params, fare=fare, fleet=fleet)
                print(fleet, fare, revenue, repl)
                ret.append({'fleet': fleet, 'fare': fare, 'revenue': revenue, 'repl': repl})

    pd.DataFrame(ret).to_csv('pricing.csv')

if __name__ == "__main__":
    platforms()

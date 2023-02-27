
import os, sys # add MaaSSim to path (not needed if MaaSSim is already in path)
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from MaaSSim.utils import get_config, load_G, prep_supply_and_demand, generate_demand, generate_vehicles, initialize_df  # simulator
from MaaSSim.data_structures import structures as inData
from MaaSSim.simulators import simulate
from MaaSSim.visualizations import plot_veh
from MaaSSim.shared import prep_shared_rides
import logging
import matplotlib.pyplot as plt

import pandas as pd
import ExMAS


# Delft

params = get_config('D:/Development/MaaSSim/data/config/delft.json')  # load configuration

params.times.pickup_patience = 3600 # 1 hour of simulation
params.simTime = 4 # 6 minutes hour of simulation
params.nP = 40 # reuqests (and passengers)
params.nV = 20 # vehicles

params.t0 = pd.Timestamp.now()
params.shareability.avg_speed = params.speeds.ride
params.shareability.shared_discount = 0.25
params.shareability.delay_value = 1
params.shareability.WtS = 1.3
params.shareability.price = 1.5 #eur/km
params.shareability.VoT = 0.0035 #eur/s
params.shareability.matching_obj = 'u_veh' #minimize VHT for vehicles
params.shareability.pax_delay = 0
params.shareability.horizon = 600
params.shareability.max_degree = 4
params.shareability.nP = params.nP
params.shareability.share = 1
params.shareability.without_matching = True
params.shareability.operating_cost = 0.5
params.shareability.comm_rate = 0.2

inData = load_G(inData, params)  # load network graph 

inData = generate_demand(inData, params, avg_speed = False)
inData.vehicles = generate_vehicles(inData,params.nV)
inData.vehicles.platform = inData.vehicles.apply(lambda x: 0, axis = 1)
inData.passengers.platforms = inData.passengers.apply(lambda x: [0], axis = 1)
inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0], axis = 1) 
inData.platforms = initialize_df(inData.platforms)
inData.platforms.loc[0]=[1,'Uber',30]
params.shareability.share = 1
params.shareability.without_matching = True




inData = ExMAS.main(inData, params.shareability, plot=False) # create shareability graph (ExMAS) 

inData = prep_shared_rides(inData, params.shareability) # prepare schedules

inData.sblts.rides

print("MaaSSIm Simulation Begins")  

# Profit Maximization 

params.kpi = 1

sim = simulate(params = params, inData = inData, logger_level = logging.WARNING) # simulate

sim.res[0].veh_exp['REVENUE'].to_list()

print(sim.res[0].veh_exp['REVENUE'].to_list())

sim.res[0].all_kpi # All driver revenue 

print(sim.res[0])



# Solo ride-hailing

params.kpi = 2

sim = simulate(params = params, inData = inData, logger_level = logging.WARNING) # simulate

sim.res[0].veh_exp['REVENUE'].to_list()

print(sim.res[0].veh_exp['REVENUE'].to_list())

sim.res[0].all_kpi # All driver revenue 

print(sim.res[0].all_kpi)


# Nearest pickup ride-pooling

params.kpi = 3

sim = simulate(params = params, inData = inData, logger_level = logging.WARNING) # simulate

sim.res[0].veh_exp['REVENUE'].to_list()

print(sim.res[0].veh_exp['REVENUE'].to_list())

total = sim.res[0].all_kpi # All driver revenue 

print(total)





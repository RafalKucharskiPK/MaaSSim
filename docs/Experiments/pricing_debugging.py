
import os, sys # add MaaSSim to path (not needed if MaaSSim is already in path)
#from envs.MaaSSim.Lib.mailcap import show
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
# Prepare 

from MaaSSim.utils import get_config, load_G, prep_supply_and_demand, generate_demand, generate_vehicles, initialize_df  # simulator
from MaaSSim.data_structures import structures as inData
from MaaSSim.simulators import simulate
from MaaSSim.visualizations import plot_veh
from MaaSSim.shared import prep_shared_rides
import logging

import pandas as pd
import ExMAS

params = get_config('D:/Development/MaaSSim/data/config/Nootdorp.json')  # load configuration

params.times.pickup_patience = 3600 # 1 hour of simulation
params.simTime = 0.1 # 6 minutes hour of simulation
params.nP = 10 # reuqests (and passengers)
params.nV = 10 # vehicles

params.t0 = pd.Timestamp.now()
params.shareability.avg_speed = params.speeds.ride
params.shareability.shared_discount = 0.3
params.shareability.delay_value = 1
params.shareability.WtS = 1.3
params.shareability.price = 1.5 #eur/km
params.shareability.VoT = 0.0035 #eur/s
params.shareability.matching_obj = 'u_pax' #minimize VHT for vehicles
params.shareability.pax_delay = 0
params.shareability.horizon = 600
params.shareability.max_degree = 4
params.shareability.nP = params.nP
params.shareability.share = 1
params.shareability.without_matching = True

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

prep_shared_rides = inData = prep_shared_rides(inData, params)  # prepare schedules
print(prep_shared_rides) 

print(inData.sblts.rides)

#Indexes = inData.sblts.schedule[['indexes']]
#print(Indexes)  

# Simulate Result 

print("MaaSSIm Simulation Begins")  

sim = simulate(params = params, inData = inData, logger_level = logging.WARNING) # simulate
print(sim)  

pax = sim.inData.requests.position
print(pax)


df = sim.runs[0].rides
df[df.veh==1]
print(df)

trips = sim.runs[0].trips
trips[trips.pax==8]
trips[trips.pax==0]
print(trips)

for i in range(params.nV):
    paxes = df[df.veh==i].paxes
    if paxes.apply(lambda x: len(x)).max()>1:
        break
plot_veh(inData.G, df[df.veh ==2], lw = 1)

print(plot_veh)

self = sim.pax[0]
from MaaSSim.visualizations import plot_trip
plot_trip(sim, self.id)
# try: # this is how it actually imports
#     from core.main import Simulator
#     from core.sim_utils import *
#     from sblt import sblt
#     from core.shared import prep_shared_rides
#
#     from core.sim_utils import structures as inData
#     pd.set_option('mode.chained_assignment', None)
#
# except:  # and this is just how it sees modules in PyCharm... to be fixed
#     from .core.main import Simulator
#     from .core.sim_utils import *
#     from .sblt import sblt
#     from .core.shared import prep_shared_rides
#
#     from .core.sim_utils import structures as inData
#     pd.set_option('mode.chained_assignment', None)


from MaaSSim.maassim import Simulator
from MaaSSim.shared import prep_shared_rides
from MaaSSim.data_structures import structures as inData
from MaaSSim.utils import get_config, load_G, generate_demand, generate_vehicles, initialize_df  # simulator


import random
import ExMAS
import numpy as np
import pandas as pd
import logging


def f_platform_opt_out(*args, **kwargs):
    # dummy function to handle opting out from platform
    pax = kwargs.get('pax', None)
    return pax.request.platform == -1


def get_wait_travel_times(_sim):
    # calculates KPIs from simulation results
    df = _sim.runs[0].trips
    ret = dict()
    for i, req in sim.inData.requests.iterrows():
        trip = df[df.pax == req.name]
        trip = trip.set_index('event').t
        if ('ACCEPTS_OFFER' in trip.index) and ('MEETS_DRIVER_AT_PICKUP' in trip.index):
            wait = trip.loc['MEETS_DRIVER_AT_PICKUP'] - trip.loc['ACCEPTS_OFFER']
        else:
            wait = np.nan
        if ('ARRIVES_AT_DROPOFF' in trip.index) and ('DEPARTS_FROM_PICKUP' in trip.index):
            travel = trip.loc['ARRIVES_AT_DROPOFF'] - trip.loc['DEPARTS_FROM_PICKUP']
        else:
            travel = np.nan

        ret[req.name] = {'wait_time': wait, 'travel_time': travel, 'platform':req.platform}
    return pd.DataFrame(ret).T


params = get_config('mdp.json')  # load parameters
params.nP = 1000
params.shareability.nP = params.nP
params.t0 = pd.Timestamp.now()

inData = load_G(inData, params, stats=True)  # load the graph of Delft

inData = generate_demand(inData, params, avg_speed=False)  # generate requests (o,d,t) - random

inData.platforms = initialize_df(inData.platforms)  # create platforms
inData.platforms.loc[1] = [1, 'Platform1', 30]  # those platform attributes are not used now
inData.platforms.loc[2] = [1, 'Platform2', 30]

inData.vehicles = generate_vehicles(inData, params.nV)  # create vehicles
inData.vehicles.platform = inData.vehicles.apply(lambda x: 1 if x.name <= params.nV / 2 else 2,
                                                 axis=1)  # 50-50 split of vehicles to platforms


params.p = 0.5  # split between platforms

inData.passengers.platforms = inData.passengers.apply(lambda x: [1] if random.random() < params.p else [2], axis=1)
inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0], axis=1)

inData = ExMAS.main(inData, params.shareability, plot=False)  # create shareability graph (ExMAS)
#inData = sblt.calc_sblt(inData, params.shareability, plot=False, _print=True)  # calculate the shareability graph


ret = list()
for p in [0.5]: #np.arange(0, 1.0001, 0.05):
    params.p = p
    for repl in range(1):
        # assing travellers to platforms
        inData.passengers.platforms = inData.passengers.apply(lambda x: [1] if random.random() < params.p else [2], axis=1)
        inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0], axis=1)
        inData.sblts.requests.platform = inData.requests.platform

        # here we do the matching of trips per platform
        inData = prep_shared_rides(inData, params.shareability, _print=True)
        params.assert_me = False
        # initialize Simulator object
        sim = Simulator(inData, params=params, print=True, f_trav_out=f_platform_opt_out, logger_level=logging.WARNING)
        
        #try:
        sim.make_and_run()  # run simulations
        sim.output()  # process results
        res = get_wait_travel_times(sim)  # calculate our KPIs (wait and travel time)
        res['p'] = p  # store column with platform split
        res['repl'] = repl  # store column with replication id
        ret.append(res)
        #except:
        #    pass


df = pd.concat(ret)  # store the precomputed results to csv
df['pax'] = df.index.copy()
df.to_csv('offline.csv', index=False)

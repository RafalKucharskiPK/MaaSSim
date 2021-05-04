from MaaSSim.evolution.supply import stop_crit_supply
from MaaSSim.evolution.demand import stop_crit_demand
import logging


from MaaSSim.utils import get_config, load_G, prep_supply_and_demand, generate_demand, generate_vehicles, initialize_df  # simulator

from MaaSSim.simulators import simulate
from MaaSSim.maassim import Simulator
from MaaSSim.shared import prep_shared_rides
import random
import ExMAS
from MaaSSim.evolution.demand import *
from MaaSSim.evolution.supply import *

import pandas as pd
from MaaSSim.data_structures import structures as inData


def stop_crit_coevolution(**kwargs):
    supply_converged = stop_crit_supply(**kwargs)
    demand_converged = stop_crit_demand(**kwargs)
    return supply_converged and demand_converged


def pipeline(params = None, **kwargs):
    def f_platform_opt_out(*args, **kwargs):
        # dummy function to handle opting out from platform
        pax = kwargs.get('pax', None)
        return pax.request.platform == -2


    from MaaSSim.data_structures import structures as inData


    # load config
    if params is None:
        params = get_config('../../data/config/coevolution.json', root_path=kwargs.get('root_path'))  # load from .json file

    inData = load_G(inData, params)  # load network graph

    # generate simulation data
    inData.vehicles = generate_vehicles_coevolution(inData, params)
    fixed_poisitons = inData.vehicles.pos.values.copy()
    inData.vehicles.platform = 1  # all vehicles serving the same platform (pooled in private ride-hailing)
    inData = generate_demand_coevolution(inData, params)
    inData.platforms = pd.read_csv(params.paths.platforms, index_col=0)

    inData = ExMAS.main(inData, params.shareability, plot=False)  # create shareability graph (ExMAS)

    inData.requests['platform'] = inData.requests.apply(lambda x: random.choice([-2, -1, 0]),
                                                        axis=1)  # this will be substituded with choices
    inData.requests['shareable'] = inData.requests.platform.apply(lambda x: x >= 0)

    inData.sblts.requests['platform'] = inData.requests['platform']    # bookkeeping
    inData.sblts.requests['shareable'] = inData.requests['shareable']    # bookkeeping

    # replace the private/pool choices with the platform to which vehicles are assigned
    inData.passengers.platforms = inData.passengers.apply(lambda x: inData.requests.loc[x.name].platform,
                                                          axis=1)
    inData.passengers.platforms = inData.passengers.platforms.apply(lambda
                                                                        x: -2 if x == -2 else 1)

    sim = Simulator(inData, params=params,
                    kpi_veh=supply_kpi_coevolution,
                    kpi_pax=demand_kpi_coevolution,
                    f_trav_out=f_platform_opt_out,
                    f_driver_out = driver_out_d2d,
                    logger_level=logging.WARNING)  # initialize
    inData = prep_shared_rides(inData, params.shareability)  # prepare schedules
    inData = set_fixed_utilities(inData, params)
    sim.generate()
    for run_id in range(params.nD):
        sim = update_utils(sim)
        inData = prep_shared_rides(inData, params.shareability)  # prepare schedules


        inData.passengers.platforms = inData.passengers.apply(lambda x: [-2] if x.platforms == -2 else [1], axis=1)


        sim.make_and_run(run_id=run_id)
        sim.output()  # calc results
        sim = travellers_learning(sim=sim)
        sim = drivers_learning(sim=sim)
        if stop_crit_coevolution(sim=sim):
            break


if __name__ == "__main__":
    pipeline()




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

    def f_traveller_opt_out(*args, **kwargs):
        # dummy function to handle opting out from platform
        pax = kwargs.get('pax', None)
        return pax.request.platform == -2

    def f_driver_opt_out(*args, **kwargs):
        # dummy function to handle opting out from platform
        veh = kwargs.get('veh',None)
        return sim.inData.vehicles.loc[veh.id].drive_decision == 'out'


    from MaaSSim.data_structures import structures as inData

    # load config
    if params is None:
        params = get_config('../../data/config/coevolution.json', root_path=kwargs.get('root_path'))  # load from .json file

    inData = load_G(inData, params)  # load network graph

    # generate simulation data
    inData.vehicles = generate_vehicles_coevolution(inData, params)
    fixed_positions = inData.vehicles.pos.values.copy()
    inData.vehicles.platform = 1  # all vehicles serving the same platform (pooled in private ride-hailing)
    inData = generate_demand_coevolution(inData, params) # travel times a
    inData.platforms = pd.read_csv(params.paths.platforms, index_col=0)
    params.shareability.logger_level = 'WARNING'

    inData = ExMAS.main(inData, params.shareability, plot=False)  # create shareability graph (ExMAS)

    inData.requests.ttrav = pd.to_timedelta(inData.sblts.requests.ttrav, unit = 's') # update travel times consistently with ExMAS

    # inData.requests['platform'] = inData.requests.apply(lambda x: random.choice([-2, -1, 0]),
    #                                                     axis=1)  # this will be substituded with choices
    # inData.requests['shareable'] = inData.requests.platform.apply(lambda x: x >= 0)
    #
    # inData.sblts.requests['platform'] = inData.requests['platform']    # bookkeeping
    # inData.sblts.requests['shareable'] = inData.requests['shareable']    # bookkeeping
    #
    # # replace the private/pool choices with the platform to which vehicles are assigned
    # inData.passengers.platforms = inData.passengers.apply(lambda x: inData.requests.loc[x.name].platform,
    #                                                       axis=1)
    # inData.passengers.platforms = inData.passengers.platforms.apply(lambda
    #                                                                     x: -2 if x == -2 else 1)

    sim = Simulator(inData, params=params,
                    kpi_veh=supply_kpi_coevolution,
                    kpi_pax=demand_kpi_coevolution,
                    f_trav_out=f_traveller_opt_out,
                    f_driver_out = f_driver_opt_out,
                    logger_level=logging.WARNING)  # initialize
    sim = set_fixed_utilities(sim)

    sim.report = dict()

    for run_id in range(params.nD):
        sim = update_utils(sim)

        sim.logger.warn(sim.inData.passengers.groupby('travel_decision').size())
        sim.inData.passengers.platforms = sim.inData.passengers.apply(
            lambda x: sim.inData.platforms[sim.inData.platforms.name == x.travel_decision].index[0], axis=1)

        sim.inData.requests['platform'] = sim.inData.passengers.platforms
        sim.inData.requests['shareable'] = sim.inData.requests.platform.apply(lambda x: x >= 0)
        sim.inData.sblts.requests['platform'] = sim.inData.requests['platform']  # bookkeeping
        sim.inData.sblts.requests['shareable'] = sim.inData.requests['shareable']  # bookkeeping


        sim.inData = prep_shared_rides(sim.inData, sim.params.shareability)  # prepare schedules
        sim.logger.warn("shared:{} \t degree:{:.2f}".format(sim.inData.requests.shareable.sum(),
                                                            sim.inData.sblts.schedule.degree.mean()))

        sim.inData.passengers.platforms = sim.inData.passengers.apply(lambda x: [-2] if x.platforms == -2 else [1], axis=1)

        sim.make_and_run(run_id=run_id)
        sim.output()  # calc results
        sim = travellers_learning(sim=sim)
        sim = drivers_learning(sim=sim)
        sim.report[run_id] = {'day': run_id,
                       'n_trav': sim.res[run_id].pax_exp[~sim.res[run_id].pax_exp.NO_REQUEST].shape[0],
                       'loses_patience': sim.res[run_id].pax_exp[sim.res[run_id].pax_exp.LOSES_PATIENCE>0].shape[0],
                       'mean_wait': sim.res[run_id].pax_exp[sim.res[run_id].pax_exp.TRAVEL>0].WAIT.mean(),
                       'mean_wait_rh': sim.res[run_id].pax_exp[sim.res[run_id].pax_exp.TRAVEL > 0].WAIT_rh.mean(),
                       'mean_wait_rp': sim.res[run_id].pax_exp[
                                  sim.res[run_id].pax_exp.TRAVEL > 0].WAIT_rp.mean(),
                       'mean_exp': sim.res[run_id].pax_exp.expected_wait_rh.mean(),
                       'n_exp': sim.inData.passengers[sim.inData.passengers.experienced].shape[0],
                       'total exp': sim.inData.passengers.days_with_exp.sum(),
                       'mean_prob_rh': sim.res[run_id-1].pax_exp.prob_rh.mean() if run_id>0 else -1,
                       'mean_prob_rs': sim.res[run_id-1].pax_exp.prob_rp.mean() if run_id>0 else -1,
                      'n_drivers': sim.res[run_id].veh_exp[~sim.res[run_id].veh_exp.OUT].shape[0],
                       'mean_income':sim.res[run_id].veh_exp[~sim.res[run_id].veh_exp.OUT].NET_INCOME.mean(),
                       'perc_income':sim.res[run_id].veh_exp[~sim.res[run_id].veh_exp.OUT].perc_income.mean()}
        sim.logger.warn(sim.report[run_id])

        if stop_crit_coevolution(sim=sim):
            break
    return sim


if __name__ == "__main__":
    pipeline()




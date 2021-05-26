import logging
import re

from scipy.optimize import brute

from MaaSSim.utils import get_config, load_G, slice_space, generate_demand, generate_vehicles, \
    initialize_df  # simulator

from MaaSSim.maassim import Simulator
from MaaSSim.shared import prep_shared_rides
import random
import ExMAS
from MaaSSim.evolution.demand import *
from MaaSSim.evolution.supply import *

import pandas as pd

EXPERIMENT_NAME = 'DEFAULT'


def stop_crit_coevolution(**kwargs):
    supply_converged = stop_crit_supply(**kwargs)
    demand_converged = stop_crit_demand(**kwargs)
    return supply_converged and demand_converged


def pipeline(params=None, filename=None, **kwargs):
    def f_traveller_opt_out(*args, **kwargs):
        # dummy function to handle opting out from platform
        pax = kwargs.get('pax', None)
        return pax.request.platform == -2

    def f_driver_opt_out(*args, **kwargs):
        # dummy function to handle opting out from platform
        veh = kwargs.get('veh', None)
        return sim.inData.vehicles.loc[veh.id].drive_decision == 'out'

    from MaaSSim.data_structures import structures as inData

    # load config
    if params is None:
        params = get_config('../../data/config/coevolution.json',
                            root_path=kwargs.get('root_path'))  # load from .json file

    inData = load_G(inData, params)  # load network graph

    # generate simulation data

    inData.vehicles = generate_vehicles_coevolution(inData, params)
    fixed_positions = pd.Series(inData.vehicles.pos.values.copy(), index=inData.vehicles.index.copy())
    inData.vehicles.platform = 1  # all vehicles serving the same platform (pooled in private ride-hailing)
    inData = generate_demand_coevolution(inData, params)  # travel times a
    inData.platforms = pd.read_csv(params.paths.platforms, index_col=0)
    params.shareability.logger_level = 'INFO'

    if params.shareability.shared_discount == 0:
        params.shareability.max_degree = 1 # to  make sure shared-rides are not computed
        # when they are not offered by the platform

    inData = ExMAS.main(inData, params.shareability, plot=False)  # create shareability graph (ExMAS)

    inData.requests.ttrav = pd.to_timedelta(inData.sblts.requests.ttrav,
                                            unit='s')  # update travel times consistently with ExMAS



    sim = Simulator(inData, params=params,
                    kpi_veh=supply_kpi_coevolution,
                    kpi_pax=demand_kpi_coevolution,
                    f_trav_out=f_traveller_opt_out,
                    f_driver_out=f_driver_opt_out,
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
        if sim.params.shareability.shared_discount >0:
            sim.logger.warn("shared:{} \t degree:{:.2f}".format(sim.inData.requests.shareable.sum(),
                                                                sim.inData.sblts.schedule.degree.mean()))

        sim.inData.passengers.platforms = sim.inData.passengers.apply(lambda x: [-2] if x.platforms == -2 else [1],
                                                                      axis=1)

        sim.make_and_run(run_id=run_id)
        sim.output()  # calc results
        sim = travellers_learning(sim=sim)
        sim = drivers_learning(sim=sim)
        sim.report[run_id] = day_report(sim, run_id)

        sim.logger.warn(sim.report[run_id])

        if stop_crit_coevolution(sim=sim):
            break

    # dump results
    for i in sim.run_ids:
        sim.res[i].veh_exp['day'] = i
        sim.res[i].pax_exp['day'] = i
    if filename is None:
        filename = EXPERIMENT_NAME
    df = pd.concat([sim.res[i].veh_exp for i in range(params.nD)])
    df.to_csv('{}_veh_exp.csv'.format(filename))
    df = pd.concat([sim.res[i].pax_exp for i in range(params.nD)])
    df.to_csv('{}_pax_exp.csv'.format(filename))
    sim.report = pd.DataFrame(sim.report).T
    sim.report.to_csv('{}_report.csv'.format(filename))

    return sim


def day_report(sim,run_id):
    if run_id>1:
        conv_rp = abs(sim.res[run_id-1].pax_exp.prob_rp.mean() - sim.res[run_id-2].pax_exp.prob_rp.mean()) / \
               sim.res[run_id-2].pax_exp.prob_rp.mean()
        conv_rh = abs(sim.res[run_id-1].pax_exp.prob_rh.mean() - sim.res[run_id - 2].pax_exp.prob_rh.mean()) / \
                  sim.res[run_id - 2].pax_exp.prob_rp.mean()
        conv_supply = abs(sim.res[run_id-1].veh_exp.prob_d.mean() - sim.res[run_id - 2].veh_exp.prob_d.mean()) / \
                  sim.res[run_id-2].veh_exp.prob_d.mean()
    else:
        conv_rp, conv_rh, conv_supply = 0,0,0



    day_report = {'day': run_id, # index
                  # experimental input
                  'nP': sim.params.nP,
                  'nV': sim.params.nV,
                  'comm_rate': sim.params.evol.comm_rate,
                  'discount': sim.params.shareability.shared_discount,

                  # travel decision results
                  'travel_decisions': sim.res[run_id].pax_exp.groupby('travel_decision').size(),
                  'n_trav': sim.res[run_id].pax_exp[~sim.res[run_id].pax_exp.NO_REQUEST].shape[0],
                  'n_drivers': sim.res[run_id].veh_exp[~sim.res[run_id].veh_exp.OUT].shape[0],
                  #profits
                  'fare':  sim.res[run_id-1].pax_exp.fare.sum() if run_id > 1 else 0,
                  'commision': sim.res[run_id - 1].veh_exp.COMMISSION.sum() if run_id > 1 else 0,
                  'revenue': sim.res[run_id - 1].veh_exp.REVENUE.sum() if run_id > 1 else 0,
                  'income':sim.res[run_id - 1].veh_exp.NET_INCOME.sum() if run_id > 1 else 0,
                  #pivot variables - experience
                  'mean_wait_rh': sim.res[run_id].pax_exp[
                      sim.res[run_id].pax_exp.TRAVEL > 0].WAIT_rh.mean(),
                  'mean_wait_rp': sim.res[run_id].pax_exp[
                      sim.res[run_id].pax_exp.TRAVEL > 0].WAIT_rp.mean(),
                  'mean_travel_rp': sim.res[run_id].pax_exp[
                      sim.res[run_id].pax_exp.TRAVEL > 0].TRAVEL_rp.mean(),
                  'mean_income': sim.res[run_id].veh_exp[~sim.res[run_id].veh_exp.OUT].NET_INCOME.mean(),
                  #pivot variables - expected
                  'mean_expected_wait_rh': sim.res[run_id].pax_exp.expected_wait_rh.mean(),
                  'mean_expected_wait_rp': sim.res[run_id].pax_exp.expected_wait_rp.mean(),
                  'mean_expected_travel_rp': sim.res[
                      run_id - 1].pax_exp.expected_travel_rp.mean() if run_id > 0 else -1,
                  'mean_expected_income': sim.res[run_id].veh_exp[~sim.res[run_id].veh_exp.OUT].perc_income.mean(),
                  # probabilities
                  'mean_prob_rh': sim.res[run_id - 1].pax_exp.prob_rh.mean() if run_id > 0 else -1,
                  'mean_prob_rs': sim.res[run_id - 1].pax_exp.prob_rp.mean() if run_id > 0 else -1,
                  # convergence
                  'n_exp': sim.res[run_id].pax_exp[sim.res[run_id].pax_exp.experienced].shape[0],
                  'learned_rh': sim.res[run_id - 1].pax_exp.learned_rh.sum() if run_id > 0 else 0,
                  'learned_rp': sim.res[run_id - 1].pax_exp.learned_rp.sum() if run_id > 0 else 0,
                  'learned': sim.res[run_id - 1].pax_exp.learned.sum() if run_id > 0 else 0,
                  'learned_drivers': sim.res[run_id - 1].veh_exp.learned.sum() if run_id > 0 else 0,
                  'conv_rp': conv_rp,
                  'conv_rh': conv_rh,
                  'conv_supply': conv_supply,
                  # others
                  'shareability': sim.inData.sblts.schedule.degree.mean() if sim.params.shareability.shared_discount>0 else 0,
                  'unserved':
                      sim.res[run_id].pax_exp[sim.res[run_id].pax_exp.LOSES_PATIENCE > 0].shape[0],
                  }
    return day_report

def evolution_search_space():
    # to see if code works
    test_space = DotMap()
    test_space.nP = [100, 300, 500, 700, 900]  # number of requests per sim time
    test_space.nV = [10, 30, 50, 70, 90]  # number of requests per sim time
    test_space.comm_rate = [0.1, 0.4, 0.7, 1, 1.3]
    test_space.shared_discount = [0, 0.15, 0.3, 0.45]
    return test_space


def simulate_parallel(config="../data/config/parallel.json", inData=None, params=None, search_space=None, **kwargs):
    if params is None:
        params = get_config(config)  # load from .json file

    brute(func=parallel_runs,
          ranges=slice_space(search_space, replications=params.parallel.get("nReplications", 1)),
          args=(params, search_space),
          full_output=True,
          finish=None,
          workers=params.parallel.get('nThread', 1))


def parallel_runs(one_slice, *args):
    # function to be used with optimize brute
    params, search_space = args  # read static input

    # _inData = inData.copy()
    _params = params.copy()
    stamp = dict()
    # parameterize
    for i, key in enumerate(search_space.keys()):
        val = search_space[key][int(one_slice[int(i)])]
        stamp[key] = val
        if key == 'comm_rate':
            _params.evol.comm_rate = val
        elif key == 'shared_discount':
            _params.shareability.shared_discount = val
        else:
            _params[key] = val

    stamp['dt'] = str(pd.Timestamp.now()).replace('-', '').replace('.', '').replace(' ', '')

    filename = EXPERIMENT_NAME

    for key, value in stamp.items():
        filename += '-{}_{}'.format(key, value)
    filename = re.sub('[^-a-zA-Z0-9_.() ]+', '', filename)


    pipeline(params=_params, filename=filename)
    # sim.dump(dump_id=filename, path = _params.paths.get('dumps', None))  # store results

    print(filename, pd.Timestamp.now(), 'end')
    return 0


if __name__ == "__main__":
    simulate_parallel(config='../../data/config/coevolution.json', search_space=evolution_search_space(), params=None)
    #pipeline()

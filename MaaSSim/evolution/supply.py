import numpy as np
from MaaSSim.utils import generate_vehicles
from MaaSSim.driver import driverEvent
import pandas as pd
import math
import random

def generate_vehicles_coevolution(_inData, _params=None):
    """
    generates vehicle database
    index is consecutive number if dataframe
    registered whether drivers have already made sign up decision
    position is random graph node
    event is set to STARTS_DAY
    """

    vehs = generate_vehicles(_inData, _params.nV)

    vehs.expected_income = np.nan
    vehs['res_wage'] = np.random.normal(_params.evol.drivers.res_wage.mean, _params.evol.drivers.res_wage.std,
                                        _params.nV)
    vehs['informed'] = (np.random.rand(_params.nV) < _params.evol.drivers.inform.prob_start)
    vehs['registered'] = (np.random.rand(_params.nV) < _params.evol.drivers.regist.prob_start) & vehs.informed
    vehs.loc[
        vehs.registered, "expected_income"] = _params.evol.drivers.init_inc_ratio * _params.evol.drivers.res_wage.mean

    return vehs


# Evaluation
def supply_kpi_coevolution(*args, **kwargs):
    # calculate vehicle KPIs (global and individual)
    sim = kwargs.get('sim', None)
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    vehindex = sim.inData.vehicles.index
    params = sim.params
    df = simrun['rides'].copy()  # results of previous simulation
    DECIDES_NOT_TO_DRIVE = df[df.event == driverEvent.DECIDES_NOT_TO_DRIVE.name].veh  # track drivers out
    dfs = df.shift(-1)  # to map time periods between events
    dfs.columns = [_ + "_s" for _ in df.columns]  # columns with _s are shifted
    df = pd.concat([df, dfs], axis=1)  # now we have time periods
    df = df[df.veh == df.veh_s]  # filter for the same vehicles only
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['veh', 'event'])['dt'].sum().unstack()  # aggregated by vehicle and event
    ret.columns.name = None
    ret = ret.reindex(vehindex)  # update for vehicles with no record
    ret['nRIDES'] = df[df.event == driverEvent.ARRIVES_AT_DROPOFF.name].groupby(
        ['veh']).size().reindex(ret.index)
    ret['nREJECTED'] = df[df.event == driverEvent.IS_REJECTED_BY_TRAVELLER.name].groupby(
        ['veh']).size().reindex(ret.index)

    dfd = df.loc[df.event == 'DEPARTS_FROM_PICKUP']
    dfd['fare'] = (dfd.dt * (params.speeds.ride / 1000) * params.platforms.fare).add(params.platforms.base_fare)
    dfd['min_fare'] = params.platforms.min_fare
    dfd['revenue'] = (dfd[['fare', 'min_fare']].max(axis=1) * (1 - params.platforms.comm_rate))

    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0  # cover all statuss
    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret = ret[['nRIDES', 'nREJECTED', 'OUT'] + [_.name for _ in driverEvent]].fillna(0)  # nans become 0

    ret['DRIVING_TIME'] = ret.REJECTS_REQUEST + ret.IS_ACCEPTED_BY_TRAVELLER + ret.DEPARTS_FROM_PICKUP
    ret['DRIVING_DIST'] = ret['DRIVING_TIME'] * (params.speeds.ride / 1000)
    ret['REVENUE'] = dfd.groupby(['veh'])['revenue'].sum().reindex(ret.index).fillna(0)
    ret['COST'] = ret['DRIVING_DIST'] * (params.drivers.fuel_costs)
    ret['NET_INCOME'] = ret['REVENUE'] - ret['COST']
    ret = ret[
        ['nRIDES', 'nREJECTED', 'DRIVING_TIME', 'DRIVING_DIST', 'REVENUE', 'COST', 'NET_INCOME', 'OUT'] + [_.name for _
                                                                                                           in
                                                                                                           driverEvent]]
    ret.index.name = 'veh'

    # KPIs
    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nV'] = ret.shape[0]
    return {'veh_exp': ret, 'veh_kpi': kpi}


def driver_out_d2d(*args, **kwargs):
    """ returns True if driver decides not to drive, and False if he drives"""
    veh = kwargs.get('veh',None)
    sim = veh.sim
    if len(sim.run_ids)==0:
        return False
    else:
        run_id = sim.run_ids[-1]

        perc_income = sim.res[run_id].veh_exp.loc[veh.id].perc_income
        util_d = veh.sim.params.evol.drivers.particip.beta * perc_income
        util_nd = veh.sim.params.evol.drivers.particip.beta * veh.veh.res_wage
        prob_d_reg = math.exp(util_d) / (math.exp(util_d) + math.exp(util_nd))
        prob_d_all = prob_d_reg
        return bool(prob_d_all < random.random())


def drivers_learning(*args, **kwargs):
    "updating drivers' day experience and determination of new perceived income"

    def update_expected_income(row):


        if run_id > 0 and sim.res[run_id - 1].veh_exp.loc[row.name].learned:  # learning is over expectations are fixed
            return sim.res[run_id - 1].veh_exp.loc[row.name].perc_income  # do not update
        elif sim.res[run_id].veh_exp.loc[row.name].OUT:  # no experience from yesterday
            if run_id > 0:
                return sim.res[run_id - 1].veh_exp.loc[row.name].perc_income  # do not update
            else:
                return vehs.loc[row.name].expected_income
        else:
            if row.worked_days == 0:
                kappa = 0.5
                old = vehs.loc[row.name].res_wage
            else:
                old = sim.res[run_id - 1].veh_exp.loc[row.name].perc_income
                kappa = 1 / (min(params.evol.drivers.window, row.worked_days) + 1)
            new_experience = sim.res[run_id].veh_exp.loc[row.name].NET_INCOME
            return old * (1 - kappa) + kappa * new_experience


    def learned(row):
        if run_id == 0:
            return False  # first day
        else:
            if sim.res[run_id - 1].veh_exp.loc[row.name].learned:  # you already learned
                return True
            else:
                if ~row.experienced:  # if not enough experiences
                    if run_id > 2 * params.evol.drivers.window:  # but many days have passed
                        # if consistently out (only after two windows)
                        last_days_out = [1 if sim.res[run_id - i].veh_exp.loc[row.name].OUT else 0
                                         for i in range(params.evol.drivers.window)]
                        # see how many times you were out in the last window
                        return sum(
                            last_days_out) > 0.9 * params.evol.drivers.window  # if more than 70% of the window - you learned.
                    else:
                        return False
                else:
                    # see how much we update the expectations
                    update = abs(
                        sim.res[run_id - 1].veh_exp.loc[row.name].perc_income - row.perc_income) / row.perc_income
                    return update < sim.params.evol.drivers.stopping_criteria  # are they stable

    sim = kwargs.get('sim', None)

    vehs = sim.inData.vehicles
    params = sim.params
    run_id = sim.run_ids[-1]

    veh_exp = sim.res[run_id].veh_exp

    if run_id > 0:
        hist = pd.concat([~sim.res[_]['veh_exp'].OUT for _ in range(0, run_id)], axis=1, ignore_index=True)
        veh_exp['worked_days'] = hist.sum(axis=1)
    else:
        veh_exp['worked_days'] = 0

    veh_exp['experienced'] = veh_exp['worked_days'] > params.evol.travellers.window

    #veh_exp['informed'] = sim.vehicles.informed
    #veh_exp['registered'] = sim.vehicles.registered.to_numpy()
    veh_exp['perc_income'] = veh_exp.apply(update_expected_income, axis=1)


    #veh_exp.loc[veh_exp.OUT, 'exp_inc'] = np.nan
    #veh_exp['kappa'] = (veh_exp.experienced/ params.evol.drivers.omega + (1 - veh_exp.experienced) / (veh_exp.worked_days + 1)) * (
    #            1 - veh_exp.OUT)
    #
    #new_perc_inc = (1 - veh_exp.kappa) * veh_exp.init_perc_inc + veh_exp.kappa * veh_exp.exp_inc

    #veh_exp['perc_inc'] = new_perc_inc.to_numpy()
    #veh_exp.loc[(veh_exp.registered) & (veh_exp.OUT), 'perc_inc'] = veh_exp.loc[
    #    veh_exp.registered & veh_exp.OUT, 'init_perc_inc']
    veh_exp['learned'] = veh_exp.apply(learned, axis=1)


    sim.res[run_id].veh_exp = veh_exp

    return sim

def stop_crit_supply(**kwargs):
    # stops simulations if at least 80% travellers have ended their learing
    sim = kwargs.get('sim', None)
    run_id = sim.run_ids[-1]
    if run_id < 5:
        return False
    else:
        conv = abs(sim.res[run_id].veh_exp.perc_income.mean().round(2) -
                   sim.res[run_id - 1].veh_exp.perc_income.mean().round(2)) / \
               sim.res[run_id].veh_exp.perc_income.mean().round(2)
        sim.logger.critical("driver's learning \tday: {}\tlearned: {:.2f}\tperc_inc: {:.2f}\tconv: {:.2f}".format(run_id,
                                                                                              sim.res[
                                                                                                  run_id].veh_exp.learned.sum() / sim.params.nV,
                                                                                              sim.res[
                                                                                                  run_id].veh_exp.perc_income.mean().round(
                                                                                                  2),
                                                                                              conv))
        return conv < sim.params.evol.drivers.stopping_criteria \
               and sim.res[run_id].veh_exp.learned.sum() > sim.params.evol.drivers.share_learned * sim.params.nV
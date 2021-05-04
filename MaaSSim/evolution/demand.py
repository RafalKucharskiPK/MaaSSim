import numpy as np
import random
import pandas as pd
import statistics
from MaaSSim.utils import generate_demand
from MaaSSim.traveller import travellerEvent


def generate_demand_coevolution(_inData, _params=None):
    """
    generates vehicle database
    index is consecutive number if dataframe
    registered whether drivers have already made sign up decision
    position is random graph node
    event is set to STARTS_DAY
    """

    inData = generate_demand(_inData, _params)

    inData.passengers['informed'] = np.random.rand(_params.nP) < _params.evol.travellers.inform.prob_start
    inData.passengers['learned'] = False
    inData.passengers['expected_wait_rs'] = _params.evol.travellers.inform.start_wait
    inData.passengers['expected_wait_rp'] = _params.evol.travellers.inform.start_wait



    return inData


def demand_kpi_coevolution(*args ,**kwargs):
    # calculate passenger indicators (global and individual)

    sim = kwargs.get('sim', None)
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    paxindex = sim.inData.passengers.index

    df = simrun['trips'].copy()  # results of previous simulation
    dfs = df.shift(-1)  # to map time periods between events
    dfs.columns = [_ + "_s" for _ in df.columns]  # columns with _s are shifted
    df = pd.concat([df, dfs], axis=1)  # now we have time periods
    df = df[df.pax == df.pax_s]  # filter for the same vehicles only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['pax', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event

    ret.columns.name = None
    ret = ret.reindex(paxindex)  # update for vehicles with no record

    if 'PREFERS_OTHER_SERVICE' in ret.columns:
        ret['NO_REQUEST'] = ~ret.PREFERS_OTHER_SERVICE.isna()
    else:
        ret['NO_REQUEST'] = False

    if 'REJECTS_OFFER' in ret.columns:
        ret['OTHER_MODE'] = ~ret.REJECTS_OFFER.isna()
    else:
        ret['OTHER_MODE'] = False

    ret.index.name = 'pax'
    ret = ret.fillna(0)

    for status in travellerEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0  # cover all statuses

    # meaningful names
    ret['TRAVEL'] = ret['ARRIVES_AT_DROPOFF']  # time with traveller (paid time)
    ret['WAIT'] = ret['RECEIVES_OFFER'] + ret[
        'MEETS_DRIVER_AT_PICKUP']  # time waiting for traveller (by default zero)
    ret['OPERATIONS'] = ret['ACCEPTS_OFFER'] + ret['DEPARTS_FROM_PICKUP'] + ret['SETS_OFF_FOR_DEST']

    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nP'] = ret.shape[0]
    return {'pax_exp': ret, 'pax_kpi': kpi}



def set_fixed_utilities(inData, params):
    mcp = params.mode_choice
    mset = params.alt_modes

    inData.passengers['ivt_seconds'] = inData.passengers.apply(lambda x: inData.requests.loc[x.name].ttrav.seconds,
                                                               axis=1)

    inData.passengers['expected_travel_time_rp'] = inData.passengers['ivt_seconds']

    def set_U_car(row):
        car_ivt = row.ivt_seconds  # assumed same as RS
        car_cost = mset.car.km_cost * car_ivt * (params.speeds.ride / 1000) + mset.car.park_cost
        U = mcp.beta_access * mset.car.access_time + \
            mcp.beta_time_moto * car_ivt + \
            mcp.beta_cost * car_cost + \
            mcp.ASC_car
        return U

    inData.passengers['U_car'] = inData.passengers.apply(set_U_car, axis=1)

    def set_U_pt(row):
        pt_ivt = row.ivt_seconds * (params.speeds.ride / params.speeds.pt)
        pt_fare = mset.pt.base_fare + mset.pt.km_fare * pt_ivt * (params.speeds.ride / 1000)
        U = mcp.beta_access * mset.pt.access_time + \
            mcp.beta_wait_pt * mset.pt.wait_time + \
            mcp.beta_time_moto * pt_ivt + mcp.beta_cost * pt_fare + \
            mcp.ASC_pt
        return U

    inData.passengers['U_pt'] = inData.passengers.apply(set_U_pt, axis=1)

    def set_U_bike(row):
        bike_tt = row.ivt_seconds * \
                  (params.speeds.ride / params.speeds.bike)
        U = mcp.beta_time_bike * bike_tt
        return U

    inData.passengers['U_bike'] = inData.passengers.apply(set_U_bike, axis=1)

    def set_rs_fare(row):
        rs_fare = max(
            params.platforms.base_fare + params.platforms.fare * row.ivt_seconds * (params.speeds.ride / 1000),
            params.platforms.min_fare)
        return rs_fare

    def set_rp_fare(row):
        return row.rs_fare * (1- params.shareability.shared_discount)

    inData.passengers['rs_fare'] = inData.passengers.apply(set_rs_fare, axis=1)
    inData.passengers['rp_fare'] = inData.passengers.apply(set_rp_fare, axis=1)
    inData.passengers['fixed_U_rs'] = inData.passengers.apply(lambda row: mcp.beta_time_moto * row.ivt_seconds +
                                                                          mcp.beta_cost * row.rs_fare + mcp.ASC_rs,
                                                              axis=1)

    inData.passengers['fixed_U_rp'] = inData.passengers.apply(lambda row: mcp.beta_cost * row.rp_fare + mcp.ASC_rs,
                                                              axis=1)

    inData.passengers['exp_sum_fixed'] = inData.passengers.apply(
        lambda x: np.exp(x.U_car) + np.exp(x.U_pt) + np.exp(x.U_bike),
        axis=1)

    return inData


def travellers_learning(sim):
    # collects history and updates expected waiting time of travellers
    # determines whether learning is over for some travellers
    params = sim.params
    pax = sim.inData.passengers
    run_id = sim.run_ids[-1]

    hist = pd.concat([~sim.res[_]['pax_exp'].NO_REQUEST for _ in range(0, run_id + 1)], axis=1, ignore_index=True)

    pax_exp = sim.res[run_id].pax_exp
    pax['days_with_exp'] = hist.sum(axis=1)
    pax['experienced'] = pax['days_with_exp'] > params.evol.travellers.window

    def experience_window(row):
        pax_id = row.name
        experience_window = list()
        for i in range(1, run_id + 1):  # browse history from now backwards
            if ~sim.res[run_id - i].pax_exp.loc[pax_id].NO_REQUEST:  # did you request
                experience_window.append(sim.res[run_id - i].pax_exp.loc[pax_id].expected_wait_rs)  # experienced wait
            if len(experience_window) >= params.evol.travellers.window:  # collect only up to window #days
                break
        return experience_window

    def learned(row):
        if run_id == 0:
            return False  # first day
        else:
            if sim.res[run_id - 1].pax_exp.loc[row.name].learned:  # you already learned
                return True
            else:
                if len(row.experiences) < params.evol.travellers.window:  # if not enough experiences
                    if run_id > 2 * params.evol.travellers.window:  # but many days have passed
                        # if consistently out (only after two windows)
                        last_days_out = [1 if sim.res[run_id - i].pax_exp.loc[row.name].NO_REQUEST else 0
                                         for i in range(params.evol.travellers.window)]
                        # see how many times you were out in the last window
                        return sum(
                            last_days_out) > 0.9 * params.evol.travellers.window  # if more than 70% of the window - you learned.
                    else:
                        return False
                else:
                    # see how much we update the expectations
                    update = abs(
                        sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait_rs - row.expected_wait_rs) / row.expected_wait_rs
                    return update < sim.params.evol.travellers.stopping_criteria  # are they stable

    def update_expected_wait(row):

        # elif len(row.experiences) == 0: # still no memories
        #    return sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait
        if run_id > 0 and sim.res[run_id - 1].pax_exp.loc[row.name].learned:  # learning is over expectations are fixed
            return sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait_rs  # do not update
        elif sim.res[run_id].pax_exp.loc[row.name].NO_REQUEST:  # no experience from yesterday
            if run_id > 0:
                return sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait_rs  # do not update
            else:
                return pax.loc[row.name].expected_wait_rs
        else:  # we update
            old = row.experiences
            if len(old) == 0:
                kappa = 0.5
                old = pax.loc[row.name].expected_wait_rs
            else:
                kappa = 1 / (len(old) + 1)
                old = pd.Series(old).mean()
            if sim.res[run_id].pax_exp.loc[row.name].LOSES_PATIENCE > 0:  # were you served
                new_experience = params.evol.travellers.reject_penalty  # bad experience
            else:
                new_experience = sim.res[run_id].pax_exp.loc[row.name].WAIT
            return old * (1 - kappa) + kappa * new_experience

    pax_exp['experiences'] = pax.apply(experience_window, axis=1)  # previous experiences
    pax_exp['expected_wait_rs'] = pax_exp.apply(update_expected_wait, axis=1)
    pax_exp['learned'] = pax_exp.apply(learned, axis=1)

    sim.res[run_id].pax_exp = pax_exp
    sim.inData.passengers = pax

    return sim


def update_utils(sim):
    # updates probabilities with new attributes
    params = sim.params
    mcp = params.mode_choice

    if len(sim.res) < 1:
        sim.inData.passengers['U_rs'] = sim.inData.passengers.apply(
            lambda row: row.fixed_U_rs + mcp.beta_wait_rs * row.expected_wait_rs,
            axis=1)
    else:
        run_id = sim.run_ids[-1]
        sim.inData.passengers['U_rs'] = sim.inData.passengers.apply(
            lambda row: row.fixed_U_rs + mcp.beta_wait_rs * sim.res[run_id].pax_exp.loc[row.name].expected_wait_rs,
            axis=1)

    sim.inData.passengers['exp_sum'] = sim.inData.passengers.apply(lambda row: np.exp(row.U_rs) + row.exp_sum_fixed,
                                                                   axis=1)
    sim.inData.passengers['prob_rs'] = sim.inData.passengers.apply(lambda row: np.exp(row.U_rs) / row.exp_sum,
                                                                   axis=1)
    return sim


def trav_out_d2d(**kwargs):
    # decision if traveller participates in the system today (prbability is precomputed in update_utils
    traveller = kwargs.get('pax', None)
    sim = traveller.sim
    return sim.inData.passengers.loc[traveller.id].prob_rs < random.random()


def stop_crit_demand(**kwargs):
    # stops simulations if at least 80% travellers have ended their learing
    sim = kwargs.get('sim', None)
    run_id = sim.run_ids[-1]
    if run_id < 5:
        return False
    else:
        conv = abs(sim.res[run_id].pax_exp.expected_wait_rs.mean().round(2) -
                   sim.res[run_id - 1].pax_exp.expected_wait_rs.mean().round(2)) / \
               sim.res[run_id].pax_exp.expected_wait_rs.mean().round(2)
        sim.logger.critical("travellers learning \t day: {}\tlearned: {:.2f}\texp_wait: {:.2f}\tconv: {:.2f}".format(run_id,
                                                                                              sim.res[
                                                                                                  run_id].pax_exp.learned.sum() / sim.params.nP,
                                                                                              sim.res[
                                                                                                  run_id].pax_exp.expected_wait_rs.mean().round(
                                                                                                  2),
                                                                                              conv))
        return conv < sim.params.evol.travellers.stopping_criteria \
               and sim.res[run_id].pax_exp.learned.sum() > sim.params.evol.travellers.share_learned * sim.params.nP


# deprecated
def collect_experience(sim):
    params = sim.params
    run_id = sim.run_ids[-1]

    "updating travellers' experience and updating new expected waiting time"
    df = sim.res[run_id].pax_exp

    hist = pd.concat([~sim.res[_]['pax_exp'].NO_REQUEST for _ in range(0, run_id + 1)], axis=1, ignore_index=True)

    df['days_with_exp'] = hist.sum(axis=1)
    df['experienced'] = df['days_with_exp'] > params.evol.travellers.omega

    df['previous'] = sim.inData.passengers.expected_wait.values

    df['penalty'] = df.apply(
        lambda row: (~(row.NO_REQUEST) & (row.LOSES_PATIENCE > 0)) * params.evol.travellers.reject_penalty, axis=1)

    df['WAIT'] = df.apply(lambda row: max(row.penalty, row.WAIT), axis=1)

    # df['kappa'] = df.apply(
    #    lambda row: params.evol.travellers.early_kappa if ~row.experienced else 1 / (row.days_with_exp+1), axis=1)
    df['kappa'] = df.apply(
        lambda row: 1 / (row.days_with_exp + 1) if row.experienced else params.evol.travellers.early_kappa, axis=1)
    df['expected_wait'] = df.apply(
        lambda row: row.previous if row.NO_REQUEST else (1 - row.kappa) * row.previous + row.kappa * row.WAIT, axis=1)

    sim.inData.passengers.expected_wait = df['expected_wait']
    sim.inData.passengers['experienced'] = df['experienced']
    sim.inData.passengers['days_with_exp'] = df['days_with_exp']
    sim.inData.passengers['kappa'] = df['kappa']
    sim.res[run_id].pax_exp = df

    return sim

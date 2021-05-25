import numpy as np
import random
import pandas as pd
import statistics
from MaaSSim.utils import generate_demand
from MaaSSim.traveller import travellerEvent
from dotmap import DotMap


def generate_demand_coevolution(_inData, _params=None):
    """
    generates passengers DataFrame
    index is consecutive number in dataframe
    registered whether traveller have already made sign up decision
    """

    inData = generate_demand(_inData, _params, avg_speed=False)  # basic demand generation
    inData.passengers['informed'] = np.random.rand(_params.nP) < _params.evol.travellers.inform.prob_start
    inData.passengers['learned'] = False

    # two pivot variables for learning
    inData.passengers['expected_wait_rh'] = _params.evol.travellers.inform.start_wait
    inData.passengers['expected_wait_rp'] = _params.evol.travellers.inform.start_wait

    return inData


def demand_kpi_coevolution(*args, **kwargs):
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
    ret['travel_decision'] = sim.inData.passengers.travel_decision.values.copy()

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

    ret['TRAVEL_rh'] = ret.apply(lambda x: x['TRAVEL'] if x.travel_decision == 'rh' else np.nan, axis = 1)
    ret['TRAVEL_rp'] = ret.apply(lambda x: x['TRAVEL'] if x.travel_decision == 'rp' else np.nan, axis = 1)
    ret['WAIT_rh'] = ret.apply(lambda x: x['WAIT'] if x.travel_decision == 'rh' else np.nan,axis=1)


    def gwt_rp_wait(row):
        if row.travel_decision == 'rp':
            pax = sim.pax[row.name]
            if pax.request.shareable:
                if pax.request.position == 0:
                    return row.WAIT
                else:
                    leader = pax.request.sim_schedule.req_id.dropna().values[0]
                    return ret.loc[leader].WAIT
            else:
                cos_nie_tak
        else:
            return np.nan

    ret['WAIT_rp'] = ret.apply(gwt_rp_wait, axis=1)





    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nP'] = ret.shape[0]
    return {'pax_exp': ret, 'pax_kpi': kpi}


def set_fixed_utilities(sim):
    inData = sim.inData
    params = sim.params
    mcp = params.mode_choice
    mset = params.alt_modes

    inData.passengers['ivt_seconds'] = inData.passengers.apply(lambda x: inData.requests.loc[x.name].ttrav.seconds,
                                                               axis=1)


    inData.passengers['expected_travel_rp'] = inData.passengers['ivt_seconds']

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

    def set_rh_fare(row):
        rs_fare = max(
            params.platforms.base_fare + params.platforms.fare * row.ivt_seconds * (params.speeds.ride / 1000),
            params.platforms.min_fare)
        return rs_fare

    def set_rp_fare(row):
        return row.rh_fare * (1 - params.shareability.shared_discount)

    inData.passengers['rh_fare'] = inData.passengers.apply(set_rh_fare, axis=1)
    inData.passengers['rp_fare'] = inData.passengers.apply(set_rp_fare, axis=1)
    inData.passengers['fixed_U_rh'] = inData.passengers.apply(lambda row: mcp.beta_time_moto * row.ivt_seconds +
                                                                          mcp.beta_cost * row.rh_fare + mcp.ASC_rh,
                                                              axis=1)

    inData.passengers['fixed_U_rp'] = inData.passengers.apply(lambda row: mcp.beta_cost * row.rp_fare + mcp.ASC_rp -
                                                                          100000 if params.shareability.shared_discount ==0 else 0,
                                                              axis=1)


    inData.passengers['exp_sum_fixed'] = inData.passengers.apply(
        lambda x: np.exp(x.U_car) + np.exp(x.U_pt) + np.exp(x.U_bike),
        axis=1)


    return sim


def update_utils(sim):

    def decision(row):
        return np.random.choice(['out','rh','rp'], 1, p=[row.prob_out, row.prob_rh, row.prob_rp])[0]

    # updates probabilities with new attributes
    params = sim.params
    mcp = params.mode_choice


    if len(sim.runs)==0:
        pax_exp = sim.inData.passengers.copy() # we use defaults as expectations
    else:
        run_id = sim.run_ids[-1] # we use history as expectations
        pax_exp = sim.res[run_id].pax_exp

    # sim.inData.passengers['U_rh'] = sim.inData.passengers.apply(
    #     lambda row: row.fixed_U_rh + mcp.beta_wait_rh * row.expected_wait_rh,
    #     axis=1)
    # sim.inData.passengers['U_rp'] = sim.inData.passengers.apply(
    #     lambda row: row.fixed_U_rp +
    #                 mcp.beta_wait_rh * row.expected_wait_rh +
    #                 mcp.beta_time_moto * row.expected_travel_rp,
    #     axis=1)
    pax_exp['U_rh'] = pax_exp.apply(lambda row:
                                    sim.inData.passengers.loc[row.name].fixed_U_rh +
                                    mcp.beta_wait_rh * row.expected_wait_rh,
                                    axis=1)
    pax_exp['U_rp'] = pax_exp.apply(lambda row:
                                    sim.inData.passengers.loc[row.name].fixed_U_rp +
                                    mcp.beta_wait_rp * row.expected_wait_rp +
                                    mcp.beta_time_moto * row.expected_travel_rp,
                                    axis=1)

    pax_exp['exp_sum'] = pax_exp.apply(lambda row:
                                       sim.inData.passengers.loc[row.name].exp_sum_fixed +
                                       np.exp(row.U_rh) +
                                       np.exp(row.U_rp),
                                       axis=1)

    pax_exp['prob_rh'] = pax_exp.apply(lambda row: np.exp(row.U_rh) / row.exp_sum,axis=1)
    pax_exp['prob_rp'] = pax_exp.apply(lambda row: np.exp(row.U_rp) / row.exp_sum, axis=1)
    pax_exp['prob_out'] = 1 - pax_exp.prob_rh - pax_exp.prob_rp
    pax_exp['travel_decision'] = pax_exp.apply(decision, axis = 1)

    pax_exp['fare'] = pax_exp.apply(lambda x:
                                    sim.inData.passengers.loc[x.name].rh_fare if x.travel_decision == 'rh' else
                                    sim.inData.passengers.loc[x.name].rp_fare if x.travel_decision == 'rp' else
                                    0, axis = 1)
    if len(sim.runs)>0:
        sim.res[run_id].pax_exp = pax_exp

    sim.inData.passengers['travel_decision'] = pax_exp['travel_decision']


    if len(sim.runs)==0:
        sim.res[-1] =DotMap({'pax_exp': pax_exp})
    return sim


def travellers_learning(sim):
    # collects history and updates expected waiting time of travellers
    # determines whether learning is over for some travellers
    params = sim.params
    pax = sim.inData.passengers
    run_id = sim.run_ids[-1]
    pax_exp = sim.res[run_id].pax_exp

    pax_exp['history'] = pax_exp.apply(lambda x: [sim.res[_].pax_exp.travel_decision[x.name] for _ in sim.run_ids[:-1]],
                                       axis=1)
    pax_exp['days_with_exp'] = pax_exp.apply(lambda x: len(x.history) - x.history.count('out'), axis=1)
    pax_exp['days_with_rh'] = pax_exp.apply(lambda x: x.history.count('rh'), axis=1)
    pax_exp['days_with_rp'] = pax_exp.apply(lambda x: x.history.count('rp'), axis=1)
    pax_exp['days_out'] = pax_exp.apply(lambda x: x.history.count('out'), axis=1)

    pax_exp['experienced'] = pax_exp['days_with_exp'] > params.evol.travellers.window

    def experience_window_rh(row):
        pax_id = row.name
        experience_window_rh = list()
        for i in range(run_id ):  # browse history from now backwards
            if sim.res[run_id - i - 1].pax_exp.loc[pax_id].travel_decision == 'rh':  # did you request
                experience_window_rh.append(sim.res[run_id - i].pax_exp.loc[pax_id].WAIT_rh)  # experienced wait
            if len(experience_window_rh) >= params.evol.travellers.window:  # collect only up to window #days
                break
        return experience_window_rh

    def experience_window_rp(row):
        pax_id = row.name
        experience_window_rp = list()
        for i in range(run_id):  # browse history from now backwards
            if sim.res[run_id - i -1].pax_exp.loc[pax_id].travel_decision == 'rp':  # did you request
                experience_window_rp.append(sim.res[run_id - i].pax_exp.loc[pax_id].WAIT_rp)  # experienced wait
            if len(experience_window_rp) >= params.evol.travellers.window:  # collect only up to window #days
                break
        return experience_window_rp

    def experience_window_travel_rp(row):
        pax_id = row.name
        experience_window_travel_rp = list()
        for i in range(run_id):  # browse history from now backwards
            if sim.res[run_id - i - 1].pax_exp.loc[pax_id].travel_decision == 'rp':  # did you request
                experience_window_travel_rp.append(sim.res[run_id - i].pax_exp.loc[pax_id].TRAVEL_rp)  # experienced wait
            if len(experience_window_travel_rp) >= params.evol.travellers.window:  # collect only up to window #days
                break
        return experience_window_travel_rp

    def learned_rh(row):
        if run_id == 0:
            return False  # first day
        else:
            if sim.res[run_id - 1].pax_exp.loc[row.name].learned_rh:  # you already learned
                return True
            elif row.travel_decision == 'rh':
                update = abs(sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait_rh - row.expected_wait_rh) / row.expected_wait_rh
                return update < sim.params.evol.travellers.stopping_criteria  # are they stable
            else:
                return False

    def learned_rp(row):
        if run_id == 0:
            return False  # first day
        else:
            if sim.res[run_id - 1].pax_exp.loc[row.name].learned_rp:  # you already learned
                return True
            elif row.travel_decision == 'rp':
                update_wait = abs(sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait_rp - row.expected_wait_rp) / row.expected_wait_rp
                update_travel_time = abs(sim.res[run_id - 1].pax_exp.loc[
                                      row.name].expected_travel_rp - row.expected_travel_rp) / row.expected_travel_rp

                update = max(update_wait, update_travel_time)

                return update < sim.params.evol.travellers.stopping_criteria  # are they stable
            else:
                return False

    def learned(row):
        if run_id == 0:
            return False  # first day
        else:
            if sim.res[run_id - 1].pax_exp.loc[row.name].learned:  # you already learned
                return True
            else:
                if row.days_with_exp < params.evol.travellers.window:  # if not enough experiences
                    if run_id > 2 * params.evol.travellers.window:  # but many days have passed
                        return row.days_out > 0.9 * params.evol.travellers.window  # if more than 70% of the window - you learned.
                    else:
                        return False
                else:
                    return row.learned_rh & row.learned_rp


    def update_expected_wait_rh(row):


        # elif len(row.experiences) == 0: # still no memories
        #    return sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait
        if run_id > 0 and sim.res[run_id - 1].pax_exp.loc[row.name].learned_rh:  # learning is over expectations are fixed
            return sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait_rh  # do not update
        elif sim.res[run_id].pax_exp.loc[row.name].travel_decision != 'rh':  # no experience from yesterday
            if run_id > 0:
                return sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait_rh  # do not update
            else:
                return pax.loc[row.name].expected_wait_rh
        else:  # we update
            old = row.experiences_rh
            if len(old) == 0:
                kappa = 0.8
                old = pax.loc[row.name].expected_wait_rh
            else:
                kappa = 1 / (len(old) + 1)
                old = pd.Series(old).mean()
            if sim.res[run_id].pax_exp.loc[row.name].LOSES_PATIENCE > 0:  # were you served
                new_experience = params.evol.travellers.reject_penalty  # bad experience
            else:
                new_experience = sim.res[run_id].pax_exp.loc[row.name].WAIT_rh
            return old * (1 - kappa) + kappa * new_experience


    def update_expected_wait_rp(row):

        # elif len(row.experiences) == 0: # still no memories
        #    return sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait
        if run_id > 0 and sim.res[run_id - 1].pax_exp.loc[row.name].learned_rp:  # learning is over expectations are fixed
            return sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait_rp  # do not update
        elif sim.res[run_id].pax_exp.loc[row.name].travel_decision != 'rp':  # no experience from yesterday
            if run_id > 0:
                return sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait_rp  # do not update
            else:
                return pax.loc[row.name].expected_wait_rp
        else:  # we update
            old = row.experiences_rp
            if len(old) == 0:
                kappa = 0.8
                old = pax.loc[row.name].expected_wait_rp
            else:
                kappa = 1 / (len(old) + 1)
                old = pd.Series(old).mean()
            if sim.res[run_id].pax_exp.loc[row.name].LOSES_PATIENCE > 0:  # were you served
                new_experience = params.evol.travellers.reject_penalty  # bad experience
            else:
                new_experience = sim.res[run_id].pax_exp.loc[row.name].WAIT_rp
            return old * (1 - kappa) + kappa * new_experience

    def update_expected_travel_rp(row):

        # elif len(row.experiences) == 0: # still no memories
        #    return sim.res[run_id - 1].pax_exp.loc[row.name].expected_wait
        if run_id > 0 and sim.res[run_id - 1].pax_exp.loc[row.name].learned_rp:  # learning is over expectations are fixed
            return sim.res[run_id - 1].pax_exp.loc[row.name].expected_travel_rp  # do not update
        elif sim.res[run_id].pax_exp.loc[row.name].travel_decision != 'rp':  # no experience from yesterday
            if run_id > 0:
                return sim.res[run_id - 1].pax_exp.loc[row.name].expected_travel_rp  # do not update
            else:
                return pax.loc[row.name].expected_travel_rp
        else:  # we update
            old = row.experiences_travel_rp
            if len(old) == 0:
                kappa = 0.5
                old = pax.loc[row.name].expected_travel_rp
            else:
                kappa = 1 / (len(old) + 1)
                old = pd.Series(old).mean()
            if sim.res[run_id].pax_exp.loc[row.name].LOSES_PATIENCE > 0:  # were you served
                new_experience = params.evol.travellers.reject_penalty  # bad experience
            else:
                new_experience = sim.res[run_id].pax_exp.loc[row.name].TRAVEL_rp
            return old * (1 - kappa) + kappa * new_experience

    pax_exp['experiences_rh'] = pax_exp.apply(experience_window_rh, axis=1)  # previous experiences
    pax_exp['experiences_rp'] = pax_exp.apply(experience_window_rp, axis=1)  # previous experiences
    pax_exp['experiences_travel_rp'] = pax_exp.apply(experience_window_travel_rp, axis=1)  # previous experiences
    pax_exp['expected_wait_rh'] = pax_exp.apply(update_expected_wait_rh, axis=1)
    pax_exp['expected_wait_rp'] = pax_exp.apply(update_expected_wait_rp, axis=1)

    pax_exp['expected_travel_rp'] = pax_exp.apply(update_expected_travel_rp, axis=1)

    pax_exp['learned_rh'] = pax_exp.apply(learned_rh, axis=1)
    pax_exp['learned_rp'] = pax_exp.apply(learned_rp, axis=1)
    pax_exp['learned'] = pax_exp.apply(learned, axis=1)

    sim.res[run_id].pax_exp = pax_exp
    sim.inData.passengers = pax

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
        conv_rh = abs(sim.res[run_id].pax_exp.expected_wait_rh.mean().round(2) -
                   sim.res[run_id - 1].pax_exp.expected_wait_rh.mean().round(2)) / \
               sim.res[run_id].pax_exp.expected_wait_rh.mean().round(2)
        conv_rp = abs(sim.res[run_id].pax_exp.expected_wait_rp.mean().round(2) -
                   sim.res[run_id - 1].pax_exp.expected_wait_rp.mean().round(2)) / \
               sim.res[run_id].pax_exp.expected_wait_rp.mean().round(2)
        sim.logger.critical(
            "travellers learning \t day: {}\tlearned: {:.2f}\texp_wait: {:.2f}\tconv_rh: {:.2f}\tconv_rp: {:.2f}".format(run_id,
                                                                                                     sim.res[
                                                                                                         run_id].pax_exp.learned.sum() / sim.params.nP,
                                                                                                     sim.res[
                                                                                                         run_id].pax_exp.expected_wait_rp.mean().round(
                                                                                                         2),
                                                                                                     conv_rh, conv_rp))
        return max(conv_rp,conv_rh) < sim.params.evol.travellers.stopping_criteria \
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

from MaaSSim.traveller import travellerEvent
import pandas as pd
import numpy as np
import random

def D2D_kpi_pax(*args ,**kwargs):
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

def update_d2d_travellers(*args, **kwargs):
    "updating travellers' experience"
    sim = kwargs.get('sim', None)
    params = kwargs.get('params', None)
    run_id = len(sim.res) - 1

    ret = pd.DataFrame()
    ret['pax'] = np.arange(1, params.nP + 1)
    ret['orig'] = sim.inData.requests.origin.to_numpy()
    ret['dest'] = sim.inData.requests.destination.to_numpy()
    ret['t_req'] = sim.inData.requests.treq.to_numpy()
    ret['tt_min'] = sim.inData.requests.ttrav.to_numpy()
    ret['dist'] = sim.inData.requests.dist.to_numpy()
    ret['informed'] = sim.passengers.informed.to_numpy()
    ret['requests'] = ~sim.res[run_id].pax_exp['NO_REQUEST']
    ret['gets_offer'] = (sim.res[run_id].pax_exp['LOSES_PATIENCE'].apply(lambda x: True if x == 0 else False)
                         & ret['requests']).to_numpy()
    ret['accepts_offer'] = ~sim.res[run_id].pax_exp['OTHER_MODE'] & ret['gets_offer']
    ret['xp_wait'] = sim.res[run_id].pax_exp.WAIT.to_numpy()
    ret['xp_ivt'] = sim.res[run_id].pax_exp.TRAVEL.to_numpy()
    ret['xp_ops'] = sim.res[run_id].pax_exp.OPERATIONS.to_numpy()
    ret.loc[(ret.requests == False) | (ret.gets_offer == False) | (ret.accepts_offer == False), ['xp_wait', 'xp_ivt',
                                                                                                 'xp_ops']] = np.nan
    ret['xp_tt_total'] = ret.xp_wait + ret.xp_ivt + ret.xp_ops
    ret['exp_tt_wait_prev'] = 0
    ret = ret.set_index('pax')

    return ret

def d2d_no_request(*args, **kwargs):
    " returns True if traveller does not check ridesourcing offer, False if he does"
    traveller = kwargs.get('pax',None)
    sim = traveller.sim
    params = sim.params
    mcp = params.mode_choice
    mset = params.alt_modes

    # Ridesourcing attributes
    rs_wait = traveller.pax.expected_wait
    rs_ivt = traveller.request.ttrav.seconds
    rs_fare = max(params.platforms.base_fare + params.platforms.fare * rs_ivt * (params.speeds.ride / 1000),
                  params.platforms.min_fare)

    # Attributes of alternative modes
    car_ivt = traveller.request.ttrav.seconds  # assumed same as RS
    car_cost = mset.car.km_cost * car_ivt * (params.speeds.ride / 1000)
    pt_ivt = traveller.request.ttrav.seconds  # assumed same as RS
    pt_fare = mset.pt.base_fare + mset.pt.km_fare * pt_ivt * (params.speeds.ride / 1000)

    bike_tt = sim.skims.ride.T[traveller.request.origin][traveller.request.destination] * (
                params.speeds.ride / params.speeds.bike)

    # Utilities
    U_rs = mcp.beta_wait_rs * rs_wait + mcp.beta_time_moto * rs_ivt + mcp.beta_cost * rs_fare + mcp.ASC_rs
    U_car = mcp.beta_access * mset.car.access_time + mcp.beta_time_moto * car_ivt + mcp.beta_cost * car_cost + mcp.ASC_car
    U_pt = mcp.beta_access * mset.pt.access_time + mcp.beta_wait_pt * mset.pt.wait_time + mcp.beta_time_moto * pt_ivt + mcp.beta_cost * pt_fare + mcp.ASC_pt
    U_bike = mcp.beta_time_bike * bike_tt
    U_list = [U_rs, U_car, U_pt, U_bike]
    exp_sum = np.exp(U_rs) + np.exp(U_car) + np.exp(U_pt) + np.exp(U_bike)

    # Decision
    P_list = [np.exp(U_list[mode_id]) / exp_sum for mode_id in range(len(U_list))]
    draw = np.cumsum(P_list) > random.random()
    decis = np.full((len(U_list)), False, dtype=bool)
    decis[np.argmax(draw)] = True
    trav_out = not decis[0] or not traveller.pax.informed

    return trav_out

def wom_trav(inData, **kwargs):
    "determine which travellers are informed before the start of the new day"
    params = kwargs.get('params', None)
    nP_inf = inData.passengers.informed.sum()
    nP_uninf = params.nP - nP_inf
    if nP_uninf > 0:
        exp_inf_day = (params.evol.travellers.inform.beta * nP_inf * nP_uninf) / params.nP
        prob_inf = exp_inf_day / nP_uninf
    else:
        prob_inf = 0

    new_inf = np.random.rand(params.nP) < prob_inf
    prev_inf = inData.passengers.informed.to_numpy()
    informed = (np.concatenate(([prev_inf],[new_inf]),axis=0).transpose()).any(axis=1)
    res_inf = pd.DataFrame(data = {'informed': informed}, index=np.arange(0,params.nP))

    return res_inf

def pax_mode_choice(*args, **kwargs):
    # returns boolean True if passenger decides not to use (private) ridesourcing for a given day (i.e. low quality offer)
    traveller = kwargs.get('traveller', None)
    sim = traveller.sim
    params = sim.params
    mcp = params.mode_choice
    mset = params.alt_modes

    platform_id, offer = list(traveller.offers.items())[0]

    rs_wait = sim.skims.ride.T[sim.vehs[offer['veh_id']].veh.pos][traveller.request.origin]
    rs_ivt = traveller.request.ttrav.seconds
    rs_fare = max(params.platforms.base_fare + params.platforms.fare * rs_ivt * (params.speeds.ride / 1000),
                  params.platforms.min_fare)

    car_ivt = traveller.request.ttrav.seconds  # assumed same as RS
    car_cost = mset.car.km_cost * car_ivt * (params.speeds.ride / 1000)
    pt_ivt = traveller.request.ttrav.seconds  # assumed same as RS
    pt_fare = mset.pt.base_fare + mset.pt.km_fare * pt_ivt * (params.speeds.ride / 1000)

    bike_tt = sim.skims.ride.T[traveller.request.origin][traveller.request.destination] * (
                params.speeds.ride / params.speeds.bike)

    U_rs = mcp.beta_wait_rs * rs_wait + mcp.beta_time_moto * rs_ivt + mcp.beta_cost * rs_fare + mcp.ASC_rs
    U_car = mcp.beta_access * mset.car.access_time + mcp.beta_time_moto * car_ivt + mcp.beta_cost * car_cost + mcp.ASC_car
    U_pt = mcp.beta_access * mset.pt.access_time + mcp.beta_wait_pt * mset.pt.wait_time + mcp.beta_time_moto * pt_ivt + mcp.beta_cost * pt_fare + mcp.ASC_pt
    U_bike = mcp.beta_time_bike * bike_tt
    U_list = [U_rs, U_car, U_pt, U_bike]
    exp_sum = np.exp(U_rs) + np.exp(U_car) + np.exp(U_pt) + np.exp(U_bike)

    P_list = [np.exp(U_list[mode_id]) / exp_sum for mode_id in range(len(U_list))]
    draw = np.cumsum(P_list) > random.random()
    decis = np.full((len(U_list)), False, dtype=bool)
    decis[np.argmax(draw)] = True

    other_mode = not decis[0]

    return other_mode
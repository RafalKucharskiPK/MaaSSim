from MaaSSim.driver import driverEvent
from MaaSSim.utils import generate_vehicles
import pandas as pd
import numpy as np
from dotmap import DotMap
import math
import random

def D2D_veh_exp(*args,**kwargs):
    #calculate vehicle KPIs (global and individual)
    sim =  kwargs.get('sim', None)
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    vehindex = sim.inData.vehicles.index
    params = sim.params
    df = simrun['rides'].copy() #results of previous simulation
    DECIDES_NOT_TO_DRIVE = df[df.event == driverEvent.DECIDES_NOT_TO_DRIVE.name].veh # track drivers out
    dfs = df.shift(-1) # to map time periods between events
    dfs.columns = [_+"_s" for _ in df.columns] #columns with _s are shifted
    df = pd.concat([df,dfs],axis=1) # now we have time periods
    df = df[df.veh == df.veh_s] #filter for the same vehicles only
    df=df[~(df.t == df.t_s)] # filter for positive time periods only
    df['dt'] = df.t_s - df.t # make time intervals
    ret = df.groupby(['veh','event'])['dt'].sum().unstack() #aggregated by vehicle and event
    ret.columns.name = None
    ret = ret.reindex(vehindex) #update for vehicles with no record
    ret['nRIDES'] = df[df.event == driverEvent.ARRIVES_AT_DROPOFF.name].groupby(
        ['veh']).size().reindex(ret.index)
    ret['nREJECTED'] = df[df.event == driverEvent.IS_REJECTED_BY_TRAVELLER.name].groupby(
        ['veh']).size().reindex(ret.index)
    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name]=0 #cover all statuss
    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret['DRIVING_TIME'] = ret.REJECTS_REQUEST + ret.IS_REJECTED_BY_TRAVELLER + ret.IS_ACCEPTED_BY_TRAVELLER + ret.DEPARTS_FROM_PICKUP + ret.CONTINUES_SHIFT
    ret['DRIVING_DIST'] = ret['DRIVING_TIME'] * (params.speeds.ride/1000)
    ret['REVENUE'] = (ret.DEPARTS_FROM_PICKUP * (params.speeds.ride/1000) * params.platforms.fare).add(params.platforms.base_fare * ret.nRIDES) * (1-params.platforms.comm_rate)
    ret['COST'] = ret['DRIVING_DIST'] * (params.drivers.fuel_costs)
    ret['NET_INCOME'] = ret['REVENUE'] - ret['COST']
    ret = ret[['nRIDES','nREJECTED', 'DRIVING_TIME', 'DRIVING_DIST', 'REVENUE', 'COST', 'NET_INCOME', 'OUT']+[_.name for _ in driverEvent]].fillna(0) #nans become 0
    ret.index.name = 'veh'

    #KPIs
    kpi = ret.agg(['sum','mean','std'])
    kpi['nV']=ret.shape[0]
    return {'veh_exp':ret,'veh_kpi':kpi}

def D2D_driver_out(*args, **kwargs):
    """ returns True if driver decides not to drive, and False if he drives"""
    veh = kwargs.get('veh',None)

    perc_income = veh.veh.expected_income
    if ~veh.veh.registered:
        return True
    if veh.sim.params.evol.drivers.particip.probabilistic:
        util_d = veh.sim.params.evol.drivers.particip.beta * perc_income
        util_nd = veh.sim.params.evol.drivers.particip.beta * veh.veh.res_wage
        prob_d_reg = math.exp(util_d) / (math.exp(util_d) + math.exp(util_nd))
        prob_d_all = prob_d_reg
        return bool(prob_d_all < random.random())
    return bool(perc_income < veh.veh.res_wage)

def update_d2d_drivers(*args, **kwargs):
    "updating drivers' day experience and determination of new perceived income"
    sim = kwargs.get('sim',None)
    params = kwargs.get('params',None)
    run_id = len(sim.res)-1
    hist = pd.concat([~sim.res[_]['veh_exp'].OUT for _ in range(0,run_id+1)],axis=1,ignore_index=True)
    worked_days = hist.sum(axis=1)

    ret = pd.DataFrame()
    ret['veh'] = np.arange(1,params.nV+1)
    ret['pos'] = sim.vehicles.pos.to_numpy()
    ret['informed'] = sim.vehicles.informed.to_numpy()
    ret['registered'] = sim.vehicles.registered.to_numpy()
    ret['out'] = sim.res[run_id].veh_exp.OUT.to_numpy()
    ret['init_perc_inc'] = sim.vehicles.expected_income.to_numpy()
    ret['exp_inc'] = sim.res[run_id].veh_exp.NET_INCOME.to_numpy()
    ret.loc[ret.out, 'exp_inc'] = np.nan
    ret['worked_days'] = worked_days.to_numpy()
    experienced_driver = (ret.worked_days >= params.evol.drivers.omega).astype(int)
    kappa = (experienced_driver / params.evol.drivers.omega + (1 - experienced_driver) / (ret.worked_days + 1)) * (1 - ret.out)
    new_perc_inc = (1 - kappa) * ret.init_perc_inc + kappa * ret.exp_inc
    
    ret['new_perc_inc'] = new_perc_inc.to_numpy()
    ret.loc[(ret.registered) & (ret.out), 'new_perc_inc'] = ret.loc[(ret.registered) & (ret.out), 'init_perc_inc']
    cols = list(ret.columns)
    a, b = cols.index('new_perc_inc'), cols.index('worked_days')
    cols[b], cols[a] = cols[a], cols[b]
    ret = ret[cols]
    ret = ret.set_index('veh')

    return  ret


def update_d2d_travellers(*args, **kwargs):
    "updating travellers' experience"
    sim = kwargs.get('sim',None)
    params = kwargs.get('params',None)
    run_id = len(sim.res)-1
    
    ret = pd.DataFrame()
    ret['pax'] = np.arange(1,params.nP+1)
    ret['orig'] = sim.requests.origin.to_numpy()
    ret['dest'] = sim.requests.destination.to_numpy()
    ret['t_req'] = sim.requests.treq.to_numpy()
    ret['tt_min'] = sim.requests.ttrav.to_numpy()
    ret['dist'] = sim.requests.dist.to_numpy()
    ret['informed'] = True
    ret['requests'] = sim.res[run_id].pax_exp['PREFERS_OTHER_SERVICE'].apply(lambda x: True if x == 0 else False).to_numpy()
    ret['gets_offer'] = (sim.res[run_id].pax_exp['LOSES_PATIENCE'].apply(lambda x: True if x == 0 else False) & ret['requests']).to_numpy()
    ret['accepts_offer'] = (sim.res[run_id].pax_exp['REJECTS_OFFER'].apply(lambda x: True if x == 0 else False) & ret['gets_offer']).to_numpy()
    ret['xp_wait'] = sim.res[run_id].pax_exp.WAIT.to_numpy()
    ret['xp_ivt'] = sim.res[run_id].pax_exp.TRAVEL.to_numpy()
    ret['xp_ops'] = sim.res[run_id].pax_exp.OPERATIONS.to_numpy()
    ret.loc[(ret.requests == False)|(ret.gets_offer == False)|(ret.accepts_offer == False), ['xp_wait','xp_ivt','xp_ops']] = np.nan
    ret['xp_tt_total'] = ret.xp_wait + ret.xp_ivt + ret.xp_ops
    ret['exp_tt_wait_prev'] = 0
    ret = ret.set_index('pax')
    
#     hist = pd.concat([~sim.res[_]['veh_exp'].OUT for _ in range(0,run_id+1)],axis=1,ignore_index=True)
#     worked_days = hist.sum(axis=1)

#     ret = pd.DataFrame()
#     ret['veh'] = np.arange(1,params.nV+1)
#     ret['informed'] = sim.vehicles.informed.to_numpy()
#     ret['registered'] = sim.vehicles.registered.to_numpy()
#     ret['out'] = sim.res[run_id].veh_exp.OUT.to_numpy()
#     ret['init_perc_inc'] = sim.vehicles.expected_income.to_numpy()
#     ret['exp_inc'] = sim.res[run_id].veh_exp.NET_INCOME.to_numpy()
#     ret.loc[ret.out, 'exp_inc'] = np.nan
#     ret['worked_days'] = worked_days.to_numpy()
#     experienced_driver = (ret.worked_days >= params.evol.drivers.omega).astype(int)
#     kappa = (experienced_driver / params.evol.drivers.omega + (1 - experienced_driver) / (ret.worked_days + 1)) * (1 - ret.out)
#     new_perc_inc = (1 - kappa) * ret.init_perc_inc + kappa * ret.exp_inc
#     ret['new_perc_inc'] = new_perc_inc.to_numpy()
#     ret.loc[(ret.registered) & (ret.out), 'new_perc_inc'] = ret.loc[(ret.registered) & (ret.out), 'init_perc_inc']
#     ret = ret.set_index('veh')

#     return  ret

    return ret


def D2D_stop_crit(*args, **kwargs):
    "returns True if simulation will be stopped, False otherwise"
    res = kwargs.get('d2d_res', None)
    params = kwargs.get('params', None)

    if len(res) < params.evol.min_it:
        return False
    ret = (res[len(res)-1].new_perc_inc - res[len(res)-1].init_perc_inc) / res[len(res)-1].init_perc_inc
    return bool(ret.abs().max() <= params.evol.conv)

def platform_regist(inData, end_day, **kwargs):
    "determine probability of registering at platform overnight for all unregistered drivers"
    params = kwargs.get('params', None)
    exp_reg_drivers = end_day.loc[end_day['registered']]
    average_perc_income = exp_reg_drivers.new_perc_inc.mean()

    samp = np.random.rand(params.nV) <= params.evol.drivers.regist.samp   # Sample of drivers making registration choice

    util_reg = params.evol.drivers.regist.beta * average_perc_income
    util_not_reg = params.evol.drivers.regist.beta * (inData.vehicles.res_wage + params.evol.drivers.regist.cost_comp)
    prob_regist = np.exp(util_reg) / (np.exp(util_reg) + np.exp(util_not_reg))
    regist_decision = (np.random.rand(params.nV) < prob_regist) & inData.vehicles.informed & samp

    prev_regist = inData.vehicles.registered.to_numpy()
    registered = (np.concatenate(([prev_regist],[regist_decision]),axis=0).transpose()).any(axis=1)
    regist_res = pd.DataFrame(data = {'registered': registered, 'expected_income': end_day.new_perc_inc})
    regist_res.loc[((inData.vehicles.registered is False) & (regist_res.registered)), "expected_income"] = average_perc_income

    return regist_res

def word_of_mouth(inData, **kwargs):
    "determine which drivers are informed before the start of the new day"
    params = kwargs.get('params', None)
    nV_inf = inData.vehicles.informed.sum()
    nV_uninf = params.nV - nV_inf
    if nV_uninf > 0:
        exp_inf_day = (params.evol.drivers.inform.beta * nV_inf * nV_uninf) / params.nV
        prob_inf = exp_inf_day / nV_uninf
    else:
        prob_inf = 0

    new_inf = np.random.rand(params.nV) < prob_inf
    prev_inf = inData.vehicles.informed.to_numpy()
    informed = (np.concatenate(([prev_inf],[new_inf]),axis=0).transpose()).any(axis=1)
    res_inf = pd.DataFrame(data = {'informed': informed}, index=np.arange(1,params.nV+1))

    return res_inf

def D2D_summary(**kwargs):
    "create day-to-day stats"
    d2d = kwargs.get('d2d', None)
    evol_micro = DotMap()
    evol_agg = DotMap()
    
    # Supply
    drivers = d2d.drivers
    inform = pd.concat([drivers[i].informed for i in range(len(drivers))],axis=1)
    inform.columns = list(range(len(drivers)))
    regist = pd.concat([drivers[i].registered for i in range(len(drivers))],axis=1)
    regist.columns = list(range(len(drivers)))
    ptcp = pd.concat([~drivers[i].out for i in range(0,len(drivers))],axis=1)
    ptcp.columns = list(range(len(drivers)))
    init_perc_inc = pd.concat([drivers[i].init_perc_inc for i in range(len(drivers))],axis=1)
    init_perc_inc.columns = list(range(len(drivers)))
    exp_inc = pd.concat([drivers[i].exp_inc for i in range(len(drivers))],axis=1)
    exp_inc.columns = list(range(len(drivers)))
    evol_micro.supply = DotMap()
    evol_micro.supply.inform = inform
    evol_micro.supply.regist = regist
    evol_micro.supply.ptcp = ptcp
    evol_micro.supply.perc_inc = init_perc_inc
    evol_micro.supply.exp_inc = exp_inc
    evol_agg.supply = pd.DataFrame({'inform': evol_micro.supply.inform.sum(), 'regist': evol_micro.supply.regist.sum(), 'particip': evol_micro.supply.ptcp.sum(), 'mean_perc_inc': evol_micro.supply.perc_inc.mean(), 'mean_exp_inc': evol_micro.supply.exp_inc.mean()})
    evol_agg.supply.index.name = 'day'

    # Demand
    travs = d2d.travs
    inform = pd.concat([travs[i].informed for i in range(len(travs))],axis=1)
    inform.columns = list(range(len(travs)))
    requests = pd.concat([travs[i].requests for i in range(len(travs))],axis=1)
    requests.columns = list(range(len(travs)))
    gets_offer = pd.concat([travs[i].gets_offer for i in range(len(travs))],axis=1)
    gets_offer.columns = list(range(len(travs)))
    accepts_offer = pd.concat([travs[i].accepts_offer for i in range(len(travs))],axis=1)
    accepts_offer.columns = list(range(len(travs)))
    wait_time = pd.concat([travs[i].xp_wait for i in range(len(travs))],axis=1)
    wait_time.columns = list(range(len(travs)))
    evol_micro.demand = DotMap()
    evol_micro.demand.inform = inform
    evol_micro.demand.requests = requests
    evol_micro.demand.gets_offer = gets_offer
    evol_micro.demand.accepts_offer = accepts_offer
    evol_micro.demand.wait_time = wait_time
    evol_agg.demand = pd.DataFrame({'inform': evol_micro.demand.inform.sum(), 'requests': evol_micro.demand.requests.sum(), 'gets_offer': evol_micro.demand.gets_offer.sum(), 'accepts_offer': evol_micro.demand.accepts_offer.sum(), 'mean_wait': evol_micro.demand.wait_time.mean()})
    evol_agg.demand.index.name = 'day'
    
    return evol_micro, evol_agg

def generate_vehicles_d2d(_inData, _params=None):
    """
    generates vehicle database
    index is consecutive number if dataframe
    registered whether drivers have already made sign up decision
    position is random graph node
    event is set to STARTS_DAY
    """

    vehs = generate_vehicles(_inData, _params.nV)
    
    vehs.expected_income = float("NaN")
    vehs['res_wage'] = np.random.normal(_params.evol.drivers.res_wage.mean, _params.evol.drivers.res_wage.std, _params.nV)
    vehs['informed'] = (np.random.rand(_params.nV) < _params.evol.drivers.inform.prob_start)
    vehs['registered'] = (np.random.rand(_params.nV) < _params.evol.drivers.regist.prob_start) & vehs.informed
    vehs.loc[vehs.registered, "expected_income"] = _params.evol.drivers.init_inc_ratio * _params.evol.drivers.res_wage.mean

    return vehs

def pax_mode_choice(*args, **kwargs):
    # returns boolean True if passenger decides not to use (private) ridesourcing for a given day (i.e. low quality offer)
    traveller = kwargs.get('traveller',None)
    sim = traveller.sim
    
    platform_id, offer = list(traveller.offers.items())[0]
    delta = 0.5
    
    pass_walk_time = sim.skims.walk[traveller.pax.pos][traveller.request.origin]
    veh_pickup_time = sim.skims.ride.T[sim.vehs[offer['veh_id']].veh.pos][traveller.request.origin]
    pass_matching_time = sim.env.now - traveller.t_matching
    tt = traveller.request.ttrav
    
    return (max(pass_walk_time, veh_pickup_time) + pass_matching_time) / tt.seconds > delta
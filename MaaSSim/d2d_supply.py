from MaaSSim.driver import driverEvent
from MaaSSim.utils import generate_vehicles
import pandas as pd
import numpy as np
import math
import random

# Generation
def generate_vehicles_d2d(_inData, _params=None):
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
    
    dfd = df.loc[df.event == 'DEPARTS_FROM_PICKUP']
    dfd['fare'] = (dfd.dt * (params.speeds.ride/1000) * params.platforms.fare).add(params.platforms.base_fare)
    dfd['min_fare'] = params.platforms.min_fare
    dfd['revenue'] = (dfd[['fare','min_fare']].max(axis=1) * (1-params.platforms.comm_rate))
    
    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name]=0 #cover all statuss
    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret = ret[['nRIDES','nREJECTED','OUT']+[_.name for _ in driverEvent]].fillna(0) #nans become 0
    
    ret['DRIVING_TIME'] = ret.REJECTS_REQUEST + ret.IS_ACCEPTED_BY_TRAVELLER + ret.DEPARTS_FROM_PICKUP
    ret['DRIVING_DIST'] = ret['DRIVING_TIME'] * (params.speeds.ride/1000)
    ret['REVENUE'] = dfd.groupby(['veh'])['revenue'].sum().reindex(ret.index).fillna(0)
    ret['COST'] = ret['DRIVING_DIST'] * (params.drivers.fuel_costs)
    ret['NET_INCOME'] = ret['REVENUE'] - ret['COST']
    ret = ret[['nRIDES','nREJECTED', 'DRIVING_TIME', 'DRIVING_DIST', 'REVENUE', 'COST', 'NET_INCOME', 'OUT']+[_.name for _ in driverEvent]]
    ret.index.name = 'veh'
    
    #KPIs
    kpi = ret.agg(['sum','mean','std'])
    kpi['nV']=ret.shape[0]
    return {'veh_exp':ret,'veh_kpi':kpi}

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

# Driver decisions
def wom_driver(inData, **kwargs):
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

def platform_regist(inData, end_day, **kwargs):
    "determine probability of registering at platform overnight for all unregistered drivers"
    params = kwargs.get('params', None)
    exp_reg_drivers = end_day.loc[end_day.registered]
    average_perc_income = exp_reg_drivers.new_perc_inc.mean()

    samp = np.random.rand(params.nV) <= params.evol.drivers.regist.samp   # Sample of drivers making registration choice

    # Probability to participate
    util_ptcp = np.ones(params.nV) * params.evol.drivers.particip.beta * average_perc_income
    util_no_ptcp = params.evol.drivers.particip.beta * inData.vehicles.res_wage.to_numpy()
    prob_d_reg = np.exp(util_ptcp) / (np.exp(util_ptcp) + np.exp(util_no_ptcp))
    
    util_reg = params.evol.drivers.regist.beta * average_perc_income * prob_d_reg
    util_not_reg = params.evol.drivers.regist.beta * (inData.vehicles.res_wage.to_numpy() + params.evol.drivers.regist.cost_comp)
    prob_regist = np.exp(util_reg) / (np.exp(util_reg) + np.exp(util_not_reg))
    regist_decision = (np.random.rand(params.nV) < prob_regist) & inData.vehicles.informed & samp

    prev_regist = inData.vehicles.registered.to_numpy()
    registered = (np.concatenate(([prev_regist],[regist_decision]),axis=0).transpose()).any(axis=1)
    regist_res = pd.DataFrame(data = {'registered': registered, 'expected_income': end_day.new_perc_inc})
    regist_res.loc[((~inData.vehicles.registered) & (regist_res.registered)), ['expected_income']] = average_perc_income
    
    return regist_res

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
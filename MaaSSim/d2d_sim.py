import pandas as pd
from dotmap import DotMap

def D2D_summary(**kwargs):
    "create day-to-day stats"
    d2d = kwargs.get('d2d', None)
    evol_micro = DotMap()
    evol_agg = DotMap()

    # Supply
    drivers = d2d.drivers
    inform = pd.concat([drivers[i].informed for i in range(len(drivers))], axis=1)
    inform.columns = list(range(len(drivers)))
    regist = pd.concat([drivers[i].registered for i in range(len(drivers))], axis=1)
    regist.columns = list(range(len(drivers)))
    ptcp = pd.concat([~drivers[i].out for i in range(0, len(drivers))], axis=1)
    ptcp.columns = list(range(len(drivers)))
    init_perc_inc = pd.concat([drivers[i].init_perc_inc for i in range(len(drivers))], axis=1)
    init_perc_inc.columns = list(range(len(drivers)))
    exp_inc = pd.concat([drivers[i].exp_inc for i in range(len(drivers))], axis=1)
    exp_inc.columns = list(range(len(drivers)))
    evol_micro.supply = DotMap()
    evol_micro.supply.inform = inform
    evol_micro.supply.regist = regist
    evol_micro.supply.ptcp = ptcp
    evol_micro.supply.perc_inc = init_perc_inc
    evol_micro.supply.exp_inc = exp_inc
    evol_agg.supply = pd.DataFrame({'inform': evol_micro.supply.inform.sum(), 'regist': evol_micro.supply.regist.sum(),
                                    'particip': evol_micro.supply.ptcp.sum(),
                                    'mean_perc_inc': evol_micro.supply.perc_inc.mean(),
                                    'mean_exp_inc': evol_micro.supply.exp_inc.mean()})
    evol_agg.supply.index.name = 'day'

    # Demand
    travs = d2d.travs
    inform = pd.concat([travs[i].informed for i in range(len(travs))], axis=1)
    inform.columns = list(range(len(travs)))
    requests = pd.concat([travs[i].requests for i in range(len(travs))], axis=1)
    requests.columns = list(range(len(travs)))
    gets_offer = pd.concat([travs[i].gets_offer for i in range(len(travs))], axis=1)
    gets_offer.columns = list(range(len(travs)))
    accepts_offer = pd.concat([travs[i].accepts_offer for i in range(len(travs))], axis=1)
    accepts_offer.columns = list(range(len(travs)))
    wait_time = pd.concat([travs[i].xp_wait for i in range(len(travs))], axis=1)
    wait_time.columns = list(range(len(travs)))
    corr_wait_time = pd.concat([travs[i].corr_xp_wait for i in range(len(travs))], axis=1)
    corr_wait_time.columns = list(range(len(travs)))
    perc_wait = pd.concat([travs[i].init_perc_wait for i in range(len(travs))], axis=1)
    perc_wait.columns = list(range(len(travs)))
    evol_micro.demand = DotMap()
    evol_micro.demand.inform = inform
    evol_micro.demand.requests = requests
    evol_micro.demand.gets_offer = gets_offer
    evol_micro.demand.accepts_offer = accepts_offer
    evol_micro.demand.wait_time = wait_time
    evol_micro.demand.corr_wait_time = corr_wait_time
    evol_micro.demand.perc_wait = perc_wait
    evol_agg.demand = pd.DataFrame(
        {'inform': evol_micro.demand.inform.sum(), 'requests': evol_micro.demand.requests.sum(),
         'gets_offer': evol_micro.demand.gets_offer.sum(), 'accepts_offer': evol_micro.demand.accepts_offer.sum(),
         'mean_wait': evol_micro.demand.wait_time.mean(), 'corr_mean_wait': evol_micro.demand.corr_wait_time.mean(), 'perc_wait': evol_micro.demand.perc_wait.mean()})
    evol_agg.demand.index.name = 'day'

    return evol_micro, evol_agg

def D2D_stop_crit(*args, **kwargs):
    "returns True if simulation will be stopped, False otherwise"
    res = kwargs.get('d2d_res', None)
    params = kwargs.get('params', None)

    if len(res) < params.evol.min_it:
        return False
    ret = (res[len(res)-1].new_perc_inc - res[len(res)-1].init_perc_inc) / res[len(res)-1].init_perc_inc
    return bool(ret.abs().max() <= params.evol.conv)
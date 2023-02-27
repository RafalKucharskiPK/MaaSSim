################################################################################
# Module: performance.py
# Description: Processes raw simulation results into dataframes with network-wide and sinlge pax/veh KPIs
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################



from .traveller import travellerEvent
from .driver import driverEvent
import pandas as pd
import matplotlib.pyplot as plt



def kpi_pax(*args,**kwargs):
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
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['pax', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event

    ret.columns.name = None
    ret = ret.reindex(paxindex)  # update for vehicles with no record

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


def kpi_veh(*args, **kwargs):
    """
    calculate vehicle KPIs (global and individual)
    it bases of duration of each event.
    The time per each event denotes the time spent by vehicle BEFORE that event took place.
    From this we can interpret duration of each segments.
    :param args:
    :param kwargs:
    :return: dictionary with kpi per vehicle and system-wide
    """
    sim =  kwargs.get('sim', None)
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    vehindex = sim.inData.vehicles.index
    df = simrun['rides'].copy()  # results of previous simulation
    DECIDES_NOT_TO_DRIVE = df[df.event == driverEvent.DECIDES_NOT_TO_DRIVE.name].veh  # track drivers out
    dfs = df.shift(-1)  # to map time periods between events
    dfs.columns = [_ + "_s" for _ in df.columns]  # columns with _s are shifted
    df = pd.concat([df, dfs], axis=1)  # now we have time periods
    df = df[df.veh == df.veh_s]  # filter for the same vehicles only
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['veh', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event
    ret.columns.name = None
    ret = ret.reindex(vehindex)  # update for vehicles with no record
    ret['nRIDES'] = df[df.event == driverEvent.ARRIVES_AT_DROPOFF.name].groupby(
        ['veh']).size().reindex(ret.index)
    ret['nREJECTED'] = df[df.event == driverEvent.IS_REJECTED_BY_TRAVELLER.name].groupby(
        ['veh']).size().reindex(ret.index)
    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0  # cover all statuss

    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret = ret[['nRIDES', 'nREJECTED', 'OUT'] + [_.name for _ in driverEvent]].fillna(0)  # nans become 0

    # meaningful names
    ret['TRAVEL'] = ret['ARRIVES_AT_DROPOFF']  # time with traveller (paid time)
    ret['WAIT'] = ret['MEETS_TRAVELLER_AT_PICKUP']  # time waiting for traveller (by default zero)
    ret['CRUISE'] = ret['ARRIVES_AT_PICKUP'] + ret['REPOSITIONED']  # time to arrive for traveller
    ret['OPERATIONS'] = ret['ACCEPTS_REQUEST'] + ret['DEPARTS_FROM_PICKUP'] + ret['IS_ACCEPTED_BY_TRAVELLER']
    ret['IDLE'] = ret['ENDS_SHIFT'] - ret['OPENS_APP'] - ret['OPERATIONS'] - ret['CRUISE'] - ret['WAIT'] - ret['TRAVEL']

    ret['PAX_KM'] = ret.apply(lambda x: sim.inData.requests.loc[sim.runs[0].trips[
        sim.runs[0].trips.veh_id == x.name].pax.unique()].dist.sum() / 1000, axis=1)
#     ret.apply(lambda x: print(sim.inData.platforms.loc[sim.inData.vehicles.loc[x.name].platform]))
#     print(sim.inData.platforms.loc[sim.inData.vehicles.loc['name'].platform])
    
    
    rides = sim.inData.sblts.rides
    profits_idx = []
    for i in range(1, len(ret.index)+1):
        profits_idx.append((max(pd.DataFrame(sim.vehs[i].myrides)['paxes'].to_list())))
    print(profits_idx)
    profits = []
    for i in profits_idx:
        row = sim.inData.sblts.rides['indexes_orig'].apply(lambda x: x == i)

        profits.append(sim.inData.sblts.rides[row.values]['driver_revenue'].to_list()[0] if sim.inData.sblts.rides[row.values]['driver_revenue'].to_list() else 0)

    
    
    
#     This ret['REVENUE'] = ret.apply(lambda x: sim.inData.platforms.loc[sim.inData.vehicles.loc[
#         x.name].platform].fare, axis=1)
    ret['REVENUE'] = profits
   # ret['nREJECTS'] = df[df.event==driverEvent.REJECTS_REQUEST.name].groupby(['veh']).size().reindex(ret.index)
    
    
    ret.index.name = 'veh'
    total_rev = ret['REVENUE'].sum()
# This is a code for plotting
#plotting seaborn
    # plot graph of driver revenue
   # vehicles  = list(sim.vehs.keys())
   # fig, ax = plt.subplots(figsize = (10,5))
    #bars = ax.barh(vehicles, profits)
   # ax.bar_label(bars)
   # for bars in ax.containers:
   #     ax.bar_label(bars)
    

    #plt.xlabel("Revenue")
   # plt.ylabel("Vehicles")
   # plt.title("revenue against driver")
   # plt.show()
    # KPIs
    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nV'] = ret.shape[0]
    return {'veh_exp': ret, 'veh_kpi': kpi, 'all_kpi': total_rev}







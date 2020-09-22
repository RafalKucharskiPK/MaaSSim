import pandas as pd
import math
import random


def prep_shared_rides(_inData, sp, sblt = None,_print = False):
    """
    Determines which requests are shareable,
    computes the matching with ExMAS,
    generates schedule of shared (or non shared) rides
    :param _inData:
    :param sp:
    :param sblt: external ExMas library (import sblt.sblt as sblt) optional
    :param _print:
    :return:
    """
    # prepare the results of calc-sblt for use in MaaSSim

    sp.share = sp.get('share', 0)  # share of shareable rides
    requests = _inData.requests

    if sp.shape == 0:
        requests.shareable = False  # all requests are not shareable
    elif sp.share == 1:
        requests.shareable = True  # all requests are shareable
    else:  # mixed - not fully tested, can be unstable
        requests.shareable = requests.apply(lambda x:
                                            False if random.random() >= sp.share
                                            else True, axis=1)

    if requests[requests.shareable].shape[0] == 0:  # no sharing
        _inData.requests['ride_id'] = _inData.requests.index.copy()  # rides are schedules, we do not share
        _inData.requests['position'] = 0  # everyone has first position
        #_inData.requests['sim_schedule'] = _inData.requests.apply(lambda x: , axis=1)


    else:  # sharing

        if sp.without_matching:  # was shareability graph comouted before?
            _inData = sblt.matching(_inData, sp, plot=False, _print=_print)  # if so, do only matching
        else:
            _inData = sblt.main(_inData, sp, plot=False, _print=_print)  # compute graph and do matching

        # prepare schedules
        schedule = _inData.sblts.schedule
        r = _inData.sblts.requests
        schedule['nodes'] = schedule.apply(lambda s: [None] + list(r.loc[s.indexes_orig].origin.values) +
                                           list(r.loc[s.indexes_dest].destination.values), axis=1)

        schedule['req_id'] = schedule.apply(lambda s: [None] + s.indexes_orig + s.indexes_dest, axis=1)

        _inData.requests['ride_id'] = r.ride_id  # store ride index in requests for simulation
        _inData.requests['position'] = r.position  # store ride index in requests for simulation
        _inData.sblts.schedule['sim_schedule'] = _inData.sblts.schedule.apply(lambda x: make_schedule_shared(x), axis=1)


    def set_sim_schedule(x):
        if not x.shareable:
            return make_schedule_nonshared([x])
        elif 'platform' in x and x.platform == -1:
            return make_schedule_nonshared([x])
        else:
            return _inData.sblts.schedule.loc[x.ride_id].sim_schedule


    _inData.requests['sim_schedule'] = _inData.requests.apply(lambda x: set_sim_schedule(x), axis=1)

    _inData.schedules_queue = pd.DataFrame([[i, _inData.schedules[i].node[1]]
                                           for i in _inData.schedules.keys()],
                                          columns=[0, 'origin']).set_index(0)

    return _inData



    schedules = dict()

    for ride in _inData.requests.ride_id.unique():
        schedules[ride]


    for i, req in requests.iterrows():
        if not req.shareable:
           schedules[reque] = make_schedule_nonshared([req])
        else:
            if not req.ride_id in schedules:
                schedules[req.ride_id] = make_schedule_shared(schedule.loc[req.ride_id])
    _inData.schedules = schedules




    return _inData


def make_schedule_nonshared(requests):
    columns = ['node', 'time', 'req_id', 'od']
    degree = 2 * len(requests) + 1
    df = pd.DataFrame(None, index=range(degree), columns=columns)
    nodes = [None] + [r.origin for r in requests] + [r.destination for r in requests]
    df.node = nodes
    df.req_id = [None] + [r.name for r in requests] * 2

    # times = list()
    # for i in range(len(nodes) - 1):
    #    times.append(sim.skims.ride[nodes[i]][nodes[i + 1]])  # travel times
    # times.append(0)
    # df.time = times
    df.od = [None] + ['o'] * len(requests) + ['d'] * len(requests)

    return df


def make_schedule_shared(row):
    columns = ['node', 'time', 'req_id', 'od']
    degree = int(2 * row.degree + 1)
    schedule = pd.DataFrame(None, index=range(degree), columns=columns)

    nodes = row.nodes
    schedule.req_id = row.req_id
    schedule.node = nodes

    # schedule.time = row.times + [0]
    schedule.od = [None] + ['o'] * row.degree + ['d'] * row.degree

    return schedule

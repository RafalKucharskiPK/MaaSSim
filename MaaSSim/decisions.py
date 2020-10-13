################################################################################
# Module: decision.py
# Description: Agent decision function templates
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################
from math import exp
import random
import pandas as pd
from dotmap import DotMap
from numpy.random.mtrand import choice

from MaaSSim.driver import driverEvent
from MaaSSim.traveller import travellerEvent


#################
#    DUMMIES    #
#################


def dummy_False(*args, **kwargs):
    # dummy function to always return False,
    # used as default function inside of functionality
    # (if the behaviour is not modelled)
    return False


def dummy_True(*args, **kwargs):
    # dummy function to always return True
    return True


def f_dummy_repos(*args, **kwargs):
    # handles the vehiciles when they become IDLE (after comppleting the request or entering the system)
    repos = DotMap()
    repos.flag = False
    # repos.pos = None
    # repos.time = 0
    return repos


################
#    DRIVER    #
################


def f_driver_out(*args, **kwargs):
    # returns boolean True if vehicle decides to opt out
    leave_threshold = 0.25
    back_threshold = 0.5
    unserved_threshold = 0.005
    anneal = 0.2

    veh = kwargs.get('veh', None)  # input
    sim = veh.sim  # input
    flag = False  # output
    if len(sim.runs) == 0: # first day
        msg = 'veh {} stays on'.format(veh.id)
    else:
        last_run = sim.run_ids[-1]
        avg_yesterday = sim.res[last_run].veh_exp.nRIDES.quantile(
            back_threshold)  # how many rides was there on average
        quant_yesterday = sim.res[last_run].veh_exp.nRIDES.quantile(
            leave_threshold)  # what was the lower quantile of rides

        prev_rides = pd.Series([sim.res[_].veh_exp.loc[veh.id].nRIDES for _ in
                                sim.run_ids]).mean()  # how many rides did I have on average before

        rides_yesterday = sim.res[last_run].veh_exp.loc[veh.id].nRIDES # how many rides did I have yesterday

        unserved_demand_yesterday = sim.res[last_run].pax_exp[sim.res[last_run].pax_exp.LOSES_PATIENCE > 0].shape[0] / \
                                    sim.res[last_run].pax_exp.shape[0]  # what is the share of unserved demand
        did_i_work_yesterday = sim.res[last_run].veh_exp.loc[veh.id].ENDS_SHIFT > 0

        if not did_i_work_yesterday:
            if avg_yesterday < prev_rides:
                msg = 'veh {} stays out'.format(veh.id)
                flag = True
            elif unserved_demand_yesterday > unserved_threshold:
                if random.random() < anneal:
                    msg = 'veh {} comes to serve unserved'.format(veh.id)
                    flag = False
                else:
                    msg = 'veh {} someone else come to serve unserved'.format(veh.id)
                    flag = False
            else:
                msg = 'veh {} comes back'.format(veh.id)
                flag = False

            pass
        else:
            if rides_yesterday > quant_yesterday:
                msg = 'veh {} stays in'.format(veh.id)
                flag = False
            else:
                msg = 'veh {} leaves'.format(veh.id)
                flag = True

    sim.logger.info('DRIVER OUT: ' + msg)
    return flag


def f_repos(*args, **kwargs):
    """
    handles the vehiciles when they become IDLE (after comppleting the request or entering the system)
    :param args:
    :param kwargs: vehicle and simulation object (veh.sim)
    :return: structure with flag = bool, position to reposition to and time that it will take to reposition there.
    """

    import random
    repos = DotMap()
    if random.random() > 0.8:  # 20% of cases driver will repos
        driver = kwargs.get('veh', None)
        sim = driver.sim
        neighbors = list(sim.inData.G.neighbors(driver.veh.pos))
        if len(neighbors) == 0:
            # escape from dead-end (teleport)
            repos.pos = sim.inData.nodes.sample(1).squeeze().name
            repos.time = 300
        else:
            repos.pos = random.choice(neighbors)
            repos.time = driver.sim.skims.ride[repos.pos][driver.veh.pos]
        repos.flag = True
    else:
        repos.flag = False

    return repos


def f_decline(*args, **kwargs):
    # determines whether driver will pick up the request or not
    # now it accepts requests only in the first quartile of travel times
    wait_limit = 200
    fare_limit = 0.1
    veh = kwargs.get('veh',None)
    offers = veh.platform.offers
    my_offer = None
    for key, offer in offers.items():
        if offer['status'] == 0 and offer['veh_id'] == veh.id:
            my_offer = offer
            break
    if my_offer is None:
        return False


    wait_time = my_offer['wait_time']
    fare = my_offer['fare']

    flag = False # i do not decline
    if wait_time  >= wait_limit:
        flag = True  # unless I have ot wait a lot
    if fare < fare_limit:
        flag = True  # or fare is low
    #if flag:
    #    veh.sim.logger.critical('Veh {} declined offer with {} wait time and fare {}'.format(veh.id, wait_time,fare))

    return flag


# ######### #
# PLATFORM  #
# ######### #


def f_match(**kwargs):
    """
    for each platfrom, whenever one of the queues changes (new idle vehicle or new unserved request)
    this procedure handles the queue and prepares transactions between drivers and travellers
    it operates based on nearest vehicle and prepares and offer to accept by traveller/vehicle
    :param kwargs:
    :return:
    """

    platform = kwargs.get('platform')  # platform for which we perform matching
    vehQ = platform.vehQ  # queue of idle vehicles
    reqQ = platform.reqQ  # queue of unserved requests
    sim = platform.sim  # reference to the simulation object

    while min(len(reqQ), len(vehQ)) > 0:  # loop until one of queues is empty (i.e. all requests handled)
        requests = sim.inData.requests.loc[reqQ]  # queued schedules of requests
        vehicles = sim.vehicles.loc[vehQ]  # vehicle agents
        skimQ = sim.skims.ride[requests.origin].loc[vehicles.pos].copy().stack()  # travel times between
        # requests and vehicles in the column vector form

        skimQ = skimQ.drop(platform.tabu, errors='ignore')  # drop already rejected matches


        if skimQ.shape[0] == 0:
            sim.logger.warn("Nobody likes each other, "
                            "Qs {}veh; {}req; tabu {}".format(len(vehQ), len(reqQ), len(platform.tabu)))
            break  # nobody likes each other - wait until new request or new vehicle

        vehPos, reqPos = skimQ.idxmin()  # find the closest ones

        mintime = skimQ.min()  # and the travel time
        vehicle = vehicles[vehicles.pos == vehPos].iloc[0]
        veh_id = vehicle.name
        veh = sim.vehs[veh_id]  # vehicle agent

        request = requests[requests.origin == reqPos].iloc[0]
        req_id = request.name
        simpaxes = request.sim_schedule.req_id.dropna().unique()
        simpax = sim.pax[simpaxes[0]]  # first traveller of shared ride (he is a leader and decision maker)

        veh.update(event=driverEvent.RECEIVES_REQUEST)
        for i in simpaxes:
            sim.pax[i].update(event=travellerEvent.RECEIVES_OFFER)

        if simpax.veh is not None:  # the traveller already assigned (to a different platform)
            if req_id in platform.reqQ:  # we were too late, forget about it
                platform.reqQ.pop(platform.reqQ.index(req_id))  # pop this request (vehicle still in the queue)
        else:
            for i in simpaxes:
                offer_id = i
                pax_request = sim.pax[i].request
                if isinstance(pax_request.ttrav, int):
                    ttrav = pax_request.ttrav
                else:
                    ttrav = pax_request.ttrav.total_seconds()
                offer = {'pax_id': i,
                         'req_id': pax_request.name,
                         'simpaxes': simpaxes,
                         'veh_id': veh_id,
                         'status': 0,  # 0 -  offer made, 1 - accepted, -1 rejected by traveller, -2 rejected by veh
                         'request': pax_request,
                         'wait_time': mintime,
                         'travel_time': ttrav,
                         'fare': platform.platform.fare * sim.pax[i].request.dist / 1000}  # make an offer
                platform.offers[offer_id] = offer  # bookkeeping of offers made by platform
                sim.pax[i].offers[platform.platform.name] = offer  # offer transferred to
            if veh.f_driver_decline(veh=veh):  # allow driver reject the request
                veh.update(event=driverEvent.REJECTS_REQUEST)
                platform.offers[offer_id]['status'] = -2
                for i in simpaxes:
                    sim.pax[i].update(event=travellerEvent.IS_REJECTED_BY_VEHICLE)
                    sim.pax[i].offers[platform.platform.name]['status'] = -2
                sim.logger.warning("pax {:>4}  {:40} {}".format(request.name,
                                                             'got rejected by vehicle ' + str(veh_id),
                                                             sim.print_now()))
                platform.tabu.append((vehPos, reqPos))  # they are unmatchable
            else:
                for i in simpaxes:
                    if not sim.pax[i].got_offered.triggered:
                        sim.pax[i].got_offered.succeed()
                vehQ.pop(vehQ.index(veh_id))  # pop offered ones
                reqQ.pop(reqQ.index(req_id))  # from the queues

        platform.updateQs()


# ######### #
# TRAVELLER #
# ######### #

def f_platform_opt_out(*args, **kwargs):
    pax = kwargs.get('pax', None)
    return pax.request.platform == -1


def f_out(*args, **kwargs):
    # it uses pax_exp of a passenger populated in previous run
    # prev_exp is a pd.Series of this pd.DataFrame
    # pd.DataFrame(columns=['wait_pickup','wait_match','tt'])
    # returns boolean True if passanger decides to opt out
    prev_exp = kwargs.get('prev_exp', None)
    if prev_exp is None:
        # no prev exepreince
        return False
    else:
        if prev_exp.iloc[0].outcome == 1:
            return False
        else:
            return True


def f_mode(*args, **kwargs):
    # returns boolean True if passenger decides not to use MaaS (bad offer)
    offer = kwargs.get('offer', None)
    delta = 0.5
    trip = kwargs.get('trip')

    pass_walk_time = trip.pass_walk_time
    veh_pickup_time = trip.sim.skims.ride.T[trip.veh.pos][trip.request.origin]
    pass_matching_time = trip.sim.env.now - trip.t_matching
    tt = trip.request.ttrav
    return (max(pass_walk_time, veh_pickup_time) + pass_matching_time) / tt.seconds > delta


def f_platform_choice(*args, **kwargs):
    traveller = kwargs.get('traveller')
    sim = traveller.sim

    betas = sim.params.platform_choice
    offers = traveller.offers

    # calc utilities
    exps = list()

    add_opt_out = True

    for platform, offer in offers.items():
        if add_opt_out:
            u = offer['wait_time'] * 2 * betas.Beta_wait + \
                offer['travel_time'] * 2 * betas.Beta_time + \
                offer['fare'] / 2 * betas.Beta_cost
            exps.append(exp(u))
            add_opt_out = False

        u = offer['wait_time'] * betas.Beta_wait + \
            offer['travel_time'] * betas.Beta_time + \
            offer['fare'] * betas.Beta_cost
        exps.append(exp(u))

    p = [_ / sum(exps) for _ in exps]
    platform_chosen = choice([-1] + list(offers.keys()), 1, p=p)[0]  # random choice with p

    if platform_chosen == -1:
        sim.logger.info("pax {:>4}  {:40} {}".format(traveller.id, 'chosen to opt out',
                                                     sim.print_now()))
    else:
        sim.logger.info("pax {:>4}  {:40} {}".format(traveller.id, 'chosen platform ' + str(platform_chosen),
                                                     sim.print_now()))
        sim.logger.info("pax {:>4}  {:40} {}".format(traveller.id, 'platform probs: ' + str(p),
                                                     sim.print_now()))

    # handle requests
    for platform_id, offer in offers.items():
        if int(platform_id) == platform_chosen:
            sim.plats[platform_id].handle_accepted(offer['pax_id'])
        else:
            sim.plats[platform_id].handle_rejected(offer['pax_id'])
        sim.logger.info("pax {:>4}  {:40} {}".format(traveller.id,
                                                     "wait: {}, travel: {}, fare: {}".format(offer['wait_time'],
                                                                                             int(offer['travel_time']),
                                                                                             int(offer[
                                                                                                     'fare'] * 100) / 100),
                                                     sim.print_now()))
    return platform_chosen == -1


#############
# SIMULATOR #
#############

def f_stop_crit(*args, **kwargs):
    """
    Decision whether to stop experiment after current iterartion
    :param args:
    :param kwargs: sim object
    :return: boolean flag
    """
    sim = kwargs.get('sim', None)
    convergence_threshold = 0.001
    _ = sim.run_ids[-1]
    sim.logger.warning(sim.res[_].veh_exp[sim.res[_].veh_exp.ENDS_SHIFT > 0].shape[0])
    if len(sim.runs) < 2:
        sim.logger.warning('Early days')
        return False
    else:
        # example of convergence on waiting times
        convergence = abs((sim.res[sim.run_ids[-1]].pax_kpi['MEETS_DRIVER_AT_PICKUP']['mean'] -
                           sim.res[sim.run_ids[-2]].pax_kpi['MEETS_DRIVER_AT_PICKUP']['mean']) /
                          sim.res[sim.run_ids[-2]].pax_kpi['MEETS_DRIVER_AT_PICKUP']['mean'])
        if convergence < convergence_threshold:
            sim.logger.warn('CONVERGED to {} after {} days'.format(convergence, sim.run_ids[-1]))
            return True
        else:
            sim.logger.warn('NOT CONVERGED to {} after {} days'.format(convergence, sim.run_ids[-1]))
            return False

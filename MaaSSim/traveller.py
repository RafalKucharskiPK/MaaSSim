from enum import Enum
import time
from dotmap import DotMap
from .driver import driverEvent
from math import exp
from numpy.random import choice


class travellerEvent(Enum):
    # actions that a traveller can take
    STARTS_DAY = 0
    REQUESTS_RIDE = 1
    RECEIVES_OFFER = 2
    IS_REJECTED_BY_VEHICLE = -2
    ACCEPTS_OFFER = 3
    REJECTS_OFFER = -3
    ARRIVES_AT_PICKUP = 5
    MEETS_DRIVER_AT_PICKUP = 6
    DEPARTS_FROM_PICKUP = 7
    ARRIVES_AT_DROPOFF = 8
    SETS_OFF_FOR_DEST = 9
    ARRIVES_AT_DEST = 10
    PREFERS_OTHER_SERVICE = -1
    LOSES_PATIENCE = -4


class PassengerAgent(object):
    def __init__(self, simData, pax_id):
        self.sim = simData  # reference to the parent Simulator instance
        self.id = pax_id  # reference in the list of simulated processes
        self.pax = self.sim.inData.passengers.loc[self.id].copy()  # reference to a simulated passenger
        self.platform_ids = self.pax.platforms

        self.requests = self.sim.inData.requests[self.sim.inData.requests.pax_id == pax_id]  # assign a requests

        self.request = self.requests.iloc[0]  # for the moment we consider only one request
        self.schedule = self.request.sim_schedule  # schedule serving this requests
        self.schedule_id = self.request.ride_id  # schedule serving this requests
        self.schedule_leader = self.request.position == 0  # orded in pickups - if it is 0 , you will request the ride,
        if self.sim.params.get('debug', False): #  debugging test test
            nodes = list(self.schedule.node.values)
            if self.request.origin != self.request.destination:
                assert nodes.index(self.request.origin) < nodes.index(self.request.destination)

        self.rides = list()  # report of this passenger process, populated while simulating
        # self.outcome = tripOutcome.NOT_PROCESSED.value #return of this process

        # functions from kwargs at the sim level
        self.f_out = self.sim.functions.f_trav_out  # handles process of exiting due to previous experience
        self.f_mode = self.sim.functions.f_trav_mode  # handles the process of exitinng due to low quality of offer
        self.f_platform_choice = self.sim.functions.f_platform_choice  # handles the process of exitinng due to low quality of offer

        # events (https://simpy.readthedocs.io/en/latest/topical_guides/events.html)
        self.action = self.sim.env.process(self.pax_action())  # <--- main process
        self.my_schedule_triggered = self.sim.env.event()
        self.lost_shared_patience = self.sim.env.event()
        self.found_veh = self.sim.env.event()
        self.got_offered = self.sim.env.event()
        self.arrived_at_pick_up = self.sim.env.event()
        self.pickuped = self.sim.env.event()
        self.dropoffed = self.sim.env.event()

        self.veh = None  # vehicle used by passenger (empty at creation)
        self.offer = dict()
        self.offers = dict()
        self.msg = ''
        self.t_matching = None

        self.sharing_not_requesting = False # depreciated

    def update(self, event, pos=None, t=True, db_update=True):
        """call whenever pos or event of vehicle changes
        keeping consistency with Simulator DB during simulation"""
        # self.pax.event = event
        self.pax.event = event
        if pos:
            self.pax.pos = pos  # update position
        if db_update:
            self.sim.passengers.loc[self.pax.name] = self.pax
        self.append_stage()

    def append_stage(self):
        """adds new record to the report"""
        stage = dict()
        stage['pax'] = self.id
        stage['pos'] = self.pax.pos
        stage['t'] = self.sim.env.now
        stage['event'] = self.pax.event.name
        stage['veh_id'] = None if self.veh is None else self.veh.name
        self.rides.append(stage)
        self.disp()

    def disp(self):
        """degugger"""
        if self.sim.params.sleep > 0:
            print(self.sim.print_now(), self.rides[-1])
            time.sleep(self.sim.params.sleep)

    def leave_queues(self):
        self.update(event=travellerEvent.LOSES_PATIENCE)
        for platform_id in self.platform_ids:
            platform = self.sim.plats[platform_id]
            if self.request.name in platform.reqQ:
                platform.reqQ.pop(platform.reqQ.index(self.request.name))
            platform.updateQs()
            # platform.resource.release(self.reqs[platform_id])
            # self.reqs[platform_id].cancel()

    def pax_action(self):

        """main routine of the passenger process,
        passes through the travellerEvent sequence in time and space"""
        self.update(event=travellerEvent.STARTS_DAY)
        if self.f_out(pax=self):
            self.msg = 'decided not to travel with MaaS'
            self.update(event=travellerEvent.PREFERS_OTHER_SERVICE)
        else:
            did_i_opt_out = False
            if self.schedule_leader:  # single ride, or you are requesting a shared ride
                yield self.sim.timeout((self.request.treq - self.sim.t0).seconds,
                                       variability=self.sim.vars.start)  # wait IDLE until the request time
                self.sim.requests.loc[len(self.sim.requests.index) + 1] = self.request  # append request
                # self.trip.request_node, self.trip.pickup_node= self.pax.pos, self.request.origin
                self.update(event=travellerEvent.REQUESTS_RIDE)

                self.t_matching = self.sim.env.now
                for platform_id in self.platform_ids:
                    platform = self.sim.plats[platform_id]
                    platform.appendReq(self.id)

                yield self.sim.timeout(self.sim.params.times.request,
                                       variability=self.sim.vars.request)  # time for transaction

                # wait until either vehicle was found or pax lost his patience
                yield self.got_offered | self.sim.timeout(self.sim.params.times.patience,
                                                        variability=self.sim.vars.patience)

                # print(self.offers)
                if len(self.offers) > 1:
                    did_i_opt_out = self.f_platform_choice(traveller=self, sim=self.sim)
                elif len(self.offers) == 1:
                    platform_id, offer = list(self.offers.items())[0]
                    if self.f_mode():
                        self.sim.plats[platform_id].handle_rejected(offer['pax_id'])
                        did_i_opt_out = True
                    else:
                        self.sim.plats[platform_id].handle_accepted(offer['pax_id'])
                else:
                    self.sim.logger.warn("pax {:>4}  {:40} {}".format(self.id, 'has no offers ',
                                                                      self.sim.print_now()))
                    self.leave_queues()
                    self.msg = 'lost his patience and left the system'
                if len(self.offers) > 0:
                    yield self.found_veh

            else:
                yield self.my_schedule_triggered | self.lost_shared_patience
            if did_i_opt_out:
                self.msg = 'decided not to travel with MaaS'
                self.update(event=travellerEvent.PREFERS_OTHER_SERVICE)
            elif self.veh is None:
                self.leave_queues()
                self.msg = 'lost his patience and left the system'
            else:

                # proper trip
                if self.schedule_leader:
                    self.offer['pass_walk_time'] = self.sim.skims.walk[self.pax.pos][self.request.origin]
                    yield self.sim.timeout(self.sim.params.times.transaction,
                                           variability=self.sim.vars.transaction)  # time for transaction
                    self.sim.vehs[self.veh.name].update(event=driverEvent.IS_ACCEPTED_BY_TRAVELLER)
                    yield self.sim.timeout(self.offer['pass_walk_time'], variability=self.sim.vars.walk)
                else:
                    # TO DO WAIT UNTIL DEPARTURE TIME OF YOUR POINT IN SCHEDULE
                    yield self.sim.timeout(10, variability=self.sim.vars.walk)
                self.arrived_at_pick_up.succeed()
                self.update(event=travellerEvent.ARRIVES_AT_PICKUP, pos=self.request.origin)
                yield self.arrived_at_pick_up & self.sim.vehs[self.veh.name].arrives_at_pick_up[self.id]
                self.update(event=travellerEvent.MEETS_DRIVER_AT_PICKUP)
                yield self.sim.timeout(self.sim.params.times.pickup)  # vehicle waits until you board
                self.pickuped.succeed()
                self.update(event=travellerEvent.DEPARTS_FROM_PICKUP)
                yield self.sim.vehs[self.veh.name].arrives[self.id]  # wait until vehicle arrive
                self.update(event=travellerEvent.ARRIVES_AT_DROPOFF, pos=self.request.destination)
                yield self.sim.timeout(self.sim.params.times.dropoff)  # time needed to dropoff
                self.dropoffed.succeed()
                self.veh = None
                self.update(event=travellerEvent.SETS_OFF_FOR_DEST)
                pass_dest_time = self.sim.skims.walk[self.pax.pos][self.request.destination]
                yield self.sim.timeout(pass_dest_time,
                                       variability=self.sim.vars.walk)  # time needed to walk to destination
                self.update(event=travellerEvent.ARRIVES_AT_DEST)
                self.msg = 'got to dest at'

        self.msg = "pax {:>4}  {:40} {}".format(self.pax.name, self.msg, self.sim.print_now())
        self.sim.logger.info(self.msg)
        # self.update(event = travellerEvent.EXIT)

# ######### #
# FUNCTIONS #
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
    veh_pickup_time = trip.sim.skims.ride[trip.veh.pos][trip.request.origin]
    pass_matching_time = trip.sim.env.now - trip.t_matching
    tt = trip.request.ttrav
    return (max(pass_walk_time, veh_pickup_time) + pass_matching_time) / tt.seconds > delta


def f_platform_choice(*args, **kwargs):
    sim = kwargs.get('sim')
    traveller = kwargs.get('traveller')

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

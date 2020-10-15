################################################################################
# Module: traveller.py
# Description: Traveller agent
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################


from enum import Enum
import time
from .driver import driverEvent


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
    """
    Traveller (Passenger) agent in the simulations.

    Attributes
    ----------
    sim : Object
      reference to the parent Simulator instance

    id : int
        reference in the list of simulated processes

    rides: list
        log of events with their time and position (node)

    veh: Object
        vehicle used by passenger (empty at creation, then self.sim.vehs[self.veh.name])

    pax: pandas Series
        reference to a simulated passenger

    platform_ids: list(int)
        list of platforms to which traveller is assigned

    requests: pandas DataFrame
        travel requests to be completed during the simulation (from inData.requests)

    request: pandas Series
        current travel request to be completed

    schedule:  DotMap
        schedule of a current request (sequence of nodes to be visited)

    schedule_id: int
        id of a schedule of a current request ()

    schedule_leader: Bool
        true for first picked-up traveller in a shared ride (of for a non-shared ride)

    f_out: function
        decision function to handle process of exiting due to previous experience

    f_mode: function
        decision function to handle the process of exiting due to low quality of offer

    f_platform_choice: function
     decision function to handle the process of exiting due to low quality of offer

    pax_action: Simpy process
        main routine of simulated traveller

    my_schedule_triggered: Simpy event
        trigerred when my co traveller already requested a shared rides

    lost_shared_patience: Simpy event
        trigerred when my co traveller lost patience

    found_veh: Simpy event
        trigerred when I have found matching vehicle

    got_offered: Simpy event
        trigerred when I got offered

    arrived_at_pick_up : Simpy event
        trigerred when I arrived for pick up

    pickuped: Simpy event
        trigerred when I got picked up

    dropoffed: Simpy event
        trigerred when I got dropped off
    """

    def __init__(self, simData, pax_id):
        self.sim = simData  # reference to the parent Simulator instance
        self.id = pax_id  # reference in the list of simulated passengers
        self.pax = self.sim.inData.passengers.loc[self.id].copy()  # reference to a simulated passenger
        self.platform_ids = self.pax.platforms  # list of platforms to which traveller is assigned

        self.requests = self.sim.inData.requests[self.sim.inData.requests.pax_id == pax_id]  # assign a requests
        self.request = self.requests.iloc[0]  # for the moment we consider only one request
        self.schedule = self.request.sim_schedule  # schedule serving this requests
        self.schedule_id = self.request.ride_id  # schedule serving this requests
        self.schedule_leader = self.request.position == 0  # orded in pickups - if it is 0 , you will request the ride,

        if self.sim.params.get('debug', False):  # debugging test test
            nodes = list(self.schedule.node.values)
            if self.request.origin != self.request.destination:
                assert nodes.index(self.request.origin) < nodes.index(self.request.destination)

        self.rides = list()  # report of this passenger process, populated while simulating

        # decision functions from kwargs at the sim level
        self.f_out = self.sim.functions.f_trav_out  # handles process of exiting due to previous experience
        self.f_mode = self.sim.functions.f_trav_mode  # handles the process of exitinng due to low quality of offer
        self.f_platform_choice = self.sim.functions.f_platform_choice  # handles the process of exitinng due to low quality of offer

        # events (https://simpy.readthedocs.io/en/latest/topical_guides/events.html)
        self.action = self.sim.env.process(self.pax_action())  # <--- main process
        self.my_schedule_triggered = self.sim.env.event()  # my co traveller already requested a shared rides
        self.lost_shared_patience = self.sim.env.event()  # my co traveller lost patience
        self.found_veh = self.sim.env.event()  # I have found matching vehicle
        self.got_offered = self.sim.env.event()  # I got offered
        self.arrived_at_pick_up = self.sim.env.event()  # I arrived for pick up
        self.pickuped = self.sim.env.event()   # I got picked up
        self.dropoffed = self.sim.env.event()  # I got dropped off

        self.veh = None  # vehicle used by passenger (empty at creation)
        self.offer = dict()  # selected offer
        self.offers = dict()  # received offers (from various platforms)
        self.msg = ''  # log message

        self.t_matching = None  # time for match


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
                    did_i_opt_out = self.f_platform_choice(traveller=self)
                elif len(self.offers) == 1:
                    platform_id, offer = list(self.offers.items())[0]
                    if self.f_mode(traveller = self):
                        self.sim.plats[platform_id].handle_rejected(offer['pax_id'])
                        did_i_opt_out = True
                    else:
                        self.sim.plats[platform_id].handle_accepted(offer['pax_id'])
                else:
                    self.sim.logger.info("pax {:>4}  {:40} {}".format(self.id, 'has no offers ',
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





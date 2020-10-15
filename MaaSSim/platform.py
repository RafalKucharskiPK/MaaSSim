################################################################################
# Module: platform.py
# Platform agent
# Rafal Kucharski @ TU Delft
################################################################################


from .driver import driverEvent
from .traveller import travellerEvent
import simpy
import random


class PlatformAgent(object):
    """
    Handles queues of its own vehicles and serves the requests of the passengers.
    Transactions are either event_based (whenever Q changes) or batched and triggered in intervals.
    Main function f_match is external and can be user defined when passed by reference to the Simulator

    Attributes
    ----------
    sim : Object
      reference to the parent Simulator instance

    id : int
        reference in the list of simulated processes

    platform: pandas Series
        reference to a simulated platform

     f_match: function
        handles process of exiting due to previous experience

    event_based: Bool
        determines the way of handling the requests, inherited from Sim

    batch_time: Bool
        time interval [s] to match the requests for non event_based platforms

    resource: simpy.Resource
        object managing serving the queues of requests to serve the queue

    vehQ: list
        list of ids of queuing vehicles

    reqQ: list
        list of ids of queuing travellers

    Qs: list
        log of Queues

    offers: dict
        list of offers made to travellers

    tabu: list(int, int)
        rejected veh - traveller pairs

    plat_action: Simpy.process
        main process triggered when new request or new vehicle arrives

    f_match: function
        the main matching process
    """

    def __init__(self, simData, platform_id):
        self.sim = simData  # reference to the parent Simulator instance
        self.id = platform_id  # reference in the list of simulated processes
        self.platform = self.sim.inData.platforms.loc[self.id].copy()  # reference to the platform
        self.f_match = self.sim.functions.f_match  # handles process of exiting due to previous experience
        self.event_based = self.sim.defaults.get('event_based', True)  # way of handling the reuqests
        self.batch_time = self.platform.batch_time  # time interval [s] to match the requests
        self.resource = simpy.Resource(self.sim.env, capacity=1000)
        self.vehQ = list()  # list of ids of queuing vehicles
        self.reqQ = list()  # list of ids of queuing travellers
        self.offers = dict()  # list of offers made to travellers
        self.monitor = self.sim.defaults.get('monitor', False)  # do we record queue lengths
        self.Qs = list()
        self.tabu = [(-1, -1)]  # list of rejected matches [veh_id, req_id]
        self.action = self.sim.env.process(self.plat_action())  # <--- main process

    def plat_action(self):
        # two ways how platform may operate
        if self.event_based:  # either do not enter this loop
            yield self.sim.env.timeout(0)  # and handle the queue on the event
        else:  # or operate every batch time
            yield self.sim.env.timeout(random.randint(0, self.batch_time))  # randomized at the begining
            while True:  # infinite loop
                self.f_match(platform=self)  # operate in loop and match requests every batch_time
                yield self.sim.env.timeout(self.batch_time)  # wait for next batch

    def appendReq(self, req):
        """
        Called whenever new requests comes in. It is either matched instantly (self.event_based),
        or simply queued and matched in batch
        :param req: id of request
        :return: None
        """
        self.reqQ.append(req)
        self.trigger_event()

    def appendVeh(self, veh):
        """
        Called whenver vehicle becomes idle and bookable
        :param veh: vehicle id
        :return:
        """
        self.vehQ.append(veh)
        self.trigger_event()

    def trigger_event(self):
        self.updateQs()
        if self.event_based:
            with self.resource.request() as req:
                self.f_match(platform=self)
                self.resource.release(req)

    def updateQs(self):
        """
        Tracks queue lengths whenever they change
        :return:
        """
        if self.monitor:
            self.Qs.append([self.sim.env.now, self.id, len(self.vehQ), len(self.reqQ)])

    def handle_rejected(self, offer_id):
        """
        triggered when offer made earlier gets rejected by traveller
        :param offer_id:
        :return:
        """
        offer = self.offers[offer_id]
        offer['status'] = -1
        veh = self.sim.vehs[offer['veh_id']]
        veh.update(event=driverEvent.IS_REJECTED_BY_TRAVELLER)
        self.tabu.append((offer['veh_id'], offer_id))  # they are unmatchable
        for i in offer['simpaxes']:
            self.sim.pax[i].update(event=travellerEvent.REJECTS_OFFER)
            self.sim.pax[i].msg = 'rejected offer'
        self.sim.logger.info("pax {:>4}  {:40} {}".format(int(offer_id),
                                                          'rejected vehicle ' + str(offer['veh_id']),
                                                          self.sim.print_now()))
        self.appendVeh(offer['veh_id'])  # bring this vehicle back to the queue

    def handle_accepted(self, offer_id):
        """
        triggered when offer made earlier is accepted by traveller
        :param offer_id:
        :return:
        """
        offer = self.offers[offer_id]
        offer['status'] = -1
        veh = self.sim.vehs[offer['veh_id']]

        for i in offer['simpaxes']:
            self.sim.pax[i].update(event=travellerEvent.ACCEPTS_OFFER)
            self.sim.pax[i].found_veh.succeed()
            self.sim.pax[i].my_schedule_triggered.succeed()
            self.sim.pax[i].veh = self.sim.vehicles.loc[offer['veh_id']]  # assigne the vehicle to passenger
        veh.update(event=driverEvent.ACCEPTS_REQUEST)
        # simpax.update(event=travellerEvent.ACCEPTS_OFFER)

        # veh.request = request  # assign request to vehicles
        veh.schedule = self.sim.pax[offer['simpaxes'][0]].schedule
        # simpax.found_veh.succeed()  # raise the event for passenger
        # simpax.veh = vehicle  # assigne the vehicle to passenger

        veh.requested.succeed()  # raise the revent for vehicle



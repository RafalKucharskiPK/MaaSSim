from .driver import driverEvent
from .traveller import travellerEvent
import simpy
import random
import pandas as pd


class PlatformAgent(object):
    """
    Handles queues of its own vehicles and serves the requests of the passengers.
    Transactions are either event_based (whenever Q changes) or batched and triggered in intervals.
    Main function f_match is external and user defined one may be passed by reference to Simulator
    """

    def __init__(self, simData, platform_id):
        self.sim = simData  # reference to the parent Simulator instance
        self.id = platform_id  # reference in the list of simulated processes
        self.platform = self.sim.inData.platforms.loc[self.id].copy()  # reference to platform
        self.f_match = self.sim.functions.f_match  # handles process of exiting due to previous experience
        self.event_based = self.sim.DEFAULTS.get('event_based', True)  # way of handling the reuqests
        self.batch_time = self.platform.batch_time  # time interval [s] to match the requests
        self.resource = simpy.Resource(self.sim.env, capacity=100)
        self.vehQ = list()  # list of ids of queuing vehicles
        self.reqQ = list()  # list of ids of queuing travellers
        self.offers = dict()  # list of offers made to travellers
        self.monitor = self.sim.DEFAULTS.get('monitor', False)  # do we record queue lengths
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
        #= sim.inData.schedules[req_id].req_id.dropna().unique()  # for shared rides there is more travellers
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
                if isinstance(pax_request.ttrav,int):
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
                sim.logger.warn("pax {:>4}  {:40} {}".format(request.name,
                                                             'got rejected by vehicle ' + str(veh_id),
                                                             sim.print_now()))
                platform.tabu.append((veh_id, req_id))  # they are unmatchable
            else:
                for i in simpaxes:
                    if not sim.pax[i].got_offered.triggered:
                        sim.pax[i].got_offered.succeed()
                vehQ.pop(vehQ.index(veh_id))  # pop offered ones
                reqQ.pop(reqQ.index(req_id))  # from the queues

        platform.updateQs()

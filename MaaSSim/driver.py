################################################################################
# Module: driver.py
# Description: Driver agent
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################
from enum import Enum
import time
from numpy import nan
import pandas as pd
from dotmap import DotMap


class driverEvent(Enum):
    """
    sequence of driver events
    """
    STARTS_DAY = 0
    OPENS_APP = 1
    RECEIVES_REQUEST = 2
    ACCEPTS_REQUEST = 3
    REJECTS_REQUEST = 4
    IS_ACCEPTED_BY_TRAVELLER = 5
    IS_REJECTED_BY_TRAVELLER = 6
    ARRIVES_AT_PICKUP = 7
    MEETS_TRAVELLER_AT_PICKUP = 8
    DEPARTS_FROM_PICKUP = 9
    ARRIVES_AT_DROPOFF = 10
    CONTINUES_SHIFT = 11
    STARTS_REPOSITIONING = 12
    REPOSITIONED = 13
    DECIDES_NOT_TO_DRIVE = -1
    ENDS_SHIFT = -2


class VehicleAgent(object):
    """
    Driver Agent
    operating in a loop between his shift_start and shift_end
    serving cyclically enqueing to queue of his platform waiting for a match and serving the reuqest
    """

    def __init__(self, simData, veh_id):
        # ids
        self.sim = simData  # reference to Simulator object
        self.id = veh_id  # unique vehicle id
        self.veh = self.sim.vehicles.loc[veh_id].copy()  # copy of inData vehicle data
        self.platform_id = self.veh.platform  # id of a platform
        self.platform = self.sim.plats[self.platform_id]  # reference to the platform
        # local variables
        self.paxes = list()
        self.schedule = None  # schedule served by vehicle (single request for case of non-shared rides)
        self.exit_flag = False  # raised at the end of the shift
        self.tveh_pickup = None  # travel time from .pos to request first node
        # output reports
        self.myrides = list()  # report of this vehicle process, populated while simulating
        # functions
        self.f_driver_out = self.sim.functions.f_driver_out  # exit from the system due to prev exp
        self.f_driver_decline = self.sim.functions.f_driver_decline  # reject the incoming request
        self.f_driver_repos = self.sim.functions.f_driver_repos  # reposition after you are free again
        # events
        self.requested = self.sim.env.event()  # triggers when vehicle is requested
        self.arrives_at_pick_up = dict()  # list of events for each passengers in the schedule
        self.arrives = dict()  # list of events for each arrival at passenger origin
        # main action
        self.action = self.sim.env.process(self.loop_day())  # main process in simu

    def update(self, event=None, pos=None, db_update=True):
        # call whenever pos or event of vehicle changes
        # keeping consistency with DB during simulation
        if event:
            self.veh.event = event
        if pos:
            self.veh.pos = pos  # update position
        if db_update:
            self.sim.vehicles.loc[self.id] = self.veh
        self.append_ride()

    def append_ride(self):
        ride = dict()
        ride['veh'] = self.id
        ride['pos'] = self.veh.pos
        ride['t'] = self.sim.env.now
        ride['event'] = self.veh.event.name
        ride['paxes'] = list(self.paxes)  # None if self.request is None else self.request.name
        self.myrides.append(ride)

        self.disp()

    def disp(self):
        """degugger"""
        if self.sim.params.sleep > 0:
            self.sim.logger.info(self.myrides[-1])
            time.sleep(self.sim.params.sleep)

    def till_end(self):
        # returns how much time left until end of shift or end of sim
        till_shift_end = self.veh.shift_end - self.sim.env.now
        till_sim_end = self.sim.t1 - self.sim.env.now - 1
        return min(till_shift_end, till_sim_end)

    def clear_me(self):
        self.arrives_at_pick_up = dict()  # init lists
        self.arrives = dict()
        self.requested = self.sim.env.event()  # initialize  request event
        # self.request = None  # initialize the request
        self.schedule = None

    def loop_day(self):
        # main routine of the vehicle process
        self.update(event=driverEvent.STARTS_DAY)
        if self.f_driver_out(veh=self):  # first see if driver wants to work this day (by default he wants)
            self.update(event=driverEvent.DECIDES_NOT_TO_DRIVE)
            msg = "veh {:>4}  {:40} {}".format(self.id, 'opted-out from the system', self.sim.print_now())
            self.sim.logger.info(msg)
            return
        yield self.sim.timeout(self.veh.shift_start, variability=self.sim.vars.shift)  # wait until shift start
        self.update(event=driverEvent.OPENS_APP)  # in the system

        while True:
            # try:  # depreciated since now traveller rejects instantly for simplicity
            repos = self.f_driver_repos(veh=self)  # reposition yourself
            if repos.flag:  # if reposition
                self.update(event=driverEvent.STARTS_REPOSITIONING)
                yield self.sim.timeout(repos.time, variability=self.sim.vars.ride)
                self.update(event=driverEvent.REPOSITIONED, pos=repos.pos)
            self.platform.appendVeh(self.id)  # appended for the queue
            yield self.requested | self.sim.timeout(self.till_end())  # wait until requested or shift end
            if self.schedule is None:
                if self.id in self.sim.vehQ:  # early exit if I quit shift, or sim ends
                    self.platform.vehQ.pop(self.sim.vehQ.index(self.id))
                    self.platform.updateQs()
                self.exit_flag = True
            else:
                # create events for each traveller
                for req in self.schedule.req_id.dropna().unique():  # two events per traveller
                    self.arrives_at_pick_up[req] = self.sim.env.event()  # when vehicle is ready to pick him up
                    self.arrives[req] = self.sim.env.event()  # when vehicle arrives at dropoff
                no_shows = list()
                for i in range(1, self.schedule.shape[0]):  # loop over the schedule
                    stage = self.schedule.loc[i]
                    if stage.req_id in no_shows:
                        break  # we do not serve this gentleman
                    yield self.sim.timeout(self.sim.skims.ride[self.veh.pos][stage.node],
                                           variability=self.sim.vars.ride)  # travel time
                    if stage.od == 'o':  # pickup
                        self.arrives_at_pick_up[stage.req_id].succeed()  # raise event
                        self.update(event=driverEvent.ARRIVES_AT_PICKUP, pos=stage.node)  # vehicle arrived
                        # driver waits until traveller arrives (or patience)
                        yield self.sim.pax[stage.req_id].arrived_at_pick_up | \
                              self.sim.timeout(self.sim.params.times.pickup_patience,
                                               variability=self.sim.vars.ride)
                        if not self.sim.pax[stage.req_id].arrived_at_pick_up:  # if traveller did not arrive
                            no_shows.apppend(stage.req_id)
                            break  # we do not serve this gentleman
                        self.update(event=driverEvent.MEETS_TRAVELLER_AT_PICKUP)
                        yield self.sim.pax[stage.req_id].pickuped  # wait until passenger has boarded
                        self.paxes.append(int(stage.req_id))
                        self.update(event=driverEvent.DEPARTS_FROM_PICKUP)
                    elif stage.od == 'd':
                        self.arrives[stage.req_id].succeed()  # arrived
                        self.update(event=driverEvent.ARRIVES_AT_DROPOFF, pos=stage.node)
                        yield self.sim.pax[stage.req_id].dropoffed  # wait until passenger has left
                        self.paxes.remove(stage.req_id)

                self.clear_me()  # initialize events. clear request
                if self.till_end() <= 1:  # quit shift
                    self.exit_flag = True

            if self.exit_flag:
                # handles end of the sim
                self.update(event=driverEvent.ENDS_SHIFT)
                msg = "veh {:>4}  {:40} {}".format(self.id, 'quitted shift', self.sim.print_now())
                self.sim.logger.info(msg)
                break


# ######### #
# FUNCTIONS #
# ######### #

def f_driver_out(*args, **kwargs):
    # it uses veh_exp of a vehicle populated in previous run
    # returns boolean True if vehicle decides to opt out
    import random
    leave_threshold  = 0.25
    back_threshold = 0.5
    unserved_threshold = 0.005
    anneal = 0.2

    veh = kwargs.get('veh', None)
    sim = veh.sim
    flag = False
    if len(sim.runs)==0:
        msg = 'veh {} runs on'.format(veh.id)
    else:
        last_run = sim.run_ids[-1]
        quant_yesterday = sim.res[last_run].veh_exp.nRIDES.quantile(leave_threshold)
        avg_yesterday = sim.res[last_run].veh_exp.nRIDES.quantile(back_threshold)
        prev_rides = pd.Series([sim.res[_].veh_exp.loc[veh.id].nRIDES for _ in sim.run_ids]).mean()
        rides_yesterday = sim.res[last_run].veh_exp.loc[veh.id].nRIDES
        unserved_demand_yesterday = sim.res[last_run].pax_exp[sim.res[last_run].pax_exp.LOSES_PATIENCE>0].shape[0]/ \
                                    sim.res[last_run].pax_exp.shape[0]
        if sim.res[last_run].veh_exp.loc[veh.id].ENDS_SHIFT == 0:
            print(unserved_demand_yesterday)
            if avg_yesterday < prev_rides:
                msg = 'veh {} stays out'.format(veh.id)
                flag = True
            elif unserved_demand_yesterday>unserved_threshold:
                if random.random()<anneal:
                    print('wracamyyy!')
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

    sim.logger.info('DRIVER OUT: '+msg)
    return flag


def f_repos(*args, **kwargs):
    # handles the vehiciles when they become IDLE (after comppleting the request or entering the system)
    import random
    repos = DotMap()
    if random.random()>0.9:  #10% of cases driver will repos
        driver = kwargs.get('veh',None)
        sim = driver.sim
        if len(list(sim.inData.G.neighbors(sim.inData.nodes.sample(1).squeeze().name)))==0:
            repos.pos = sim.inData.G.nodes.sample(1).squeeze().name
            repos.time = 60
        else:
            repos.pos = random.choice(list(sim.inData.G.neighbors(sim.inData.nodes.sample(1).squeeze().name)))
            repos.time = driver.sim.skims.ride[driver.veh.pos][repos.pos]
        repos.flag = True
    else:
        repos.flag = False

    return repos


def f_dummy_repos(*args, **kwargs):
    # handles the vehiciles when they become IDLE (after comppleting the request or entering the system)
    repos = DotMap()
    repos.flag = False
    #repos.pos = None
    #repos.time = 0
    return repos


def f_decline(*args, **kwargs):
    # determines whether driver will pick up the request or not
    # now it accepts requests only in the first quartile of travel times
    pickup_time = kwargs.get('pickup_time', None)
    request_nodes = kwargs.get('request_nodes', None)
    pos = kwargs.get('pos', None)
    skim = kwargs.get('skim', None)
    quan = 0.5

    limit = skim[pos].loc[request_nodes].quantile(quan)  # mean ditance to all the nodes
    return pickup_time >= limit

################################################################################
# Module: main.py
# Description: Simulator object
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################


from dotmap import DotMap
import pandas as pd
import math
import networkx as nx
import simpy
import time
import numpy as np
import os.path
import zipfile
from pathlib import Path

from MaaSSim.traveller import PassengerAgent, travellerEvent
from MaaSSim.driver import VehicleAgent
from MaaSSim.decisions import f_dummy_repos, f_match, dummy_False
from MaaSSim.platform import PlatformAgent
from MaaSSim.performance import kpi_pax, kpi_veh
from MaaSSim.utils import initialize_df
import sys
import logging

DEFAULTS = dict(f_match=f_match,
                f_trav_out=dummy_False,
                f_driver_learn=dummy_False,
                f_driver_out=dummy_False,
                f_trav_mode=dummy_False,
                f_driver_decline=dummy_False,
                f_platform_choice = dummy_False,
                f_driver_repos=f_dummy_repos,
                f_stop_crit=dummy_False,
                f_timeout=None,
                kpi_pax=kpi_pax,
                kpi_veh=kpi_veh,
                monitor=True)


class Simulator:
    """
    main class of MaaSSim

    used to prepare, populate, run simulations and analyze the results
    """
    # STATICS and kwargs
    # list of functionalities
    # that may be filled with functions to represent desired behaviour
    FNAMES = ['f_match',
              'f_trav_out',
              'f_driver_learn',
              'f_driver_out',
              'f_trav_mode',
              'f_driver_decline',
              'f_platform_choice',
              'f_driver_repos',
              'f_timeout',
              'f_stop_crit',
              'kpi_pax',
              'kpi_veh']


    def __init__(self, _inData, **kwargs):
        # input
        self.inData = _inData.copy()  # copy of data structure for simulations (copy needed for multi-threading)
        self.vehicles = self.inData.vehicles  # input
        self.platforms = self.inData.platforms  # input
        self.defaults = DEFAULTS.copy()  # default configuration of decision functions

        self.myinit(**kwargs)  # part that is called every run
        # output
        self.run_ids = list()  # ids of consecutively executed runs
        self.runs = dict()  # simulation outputs (raw)
        self.res = dict()  # simulation results (processed)
        self.logger = self.init_log(**kwargs)
        self.logger.warning("""Setting up {}h simulation at {} for {} vehicles and {} passengers in {}"""
                            .format(self.params.simTime,
                                    self.t0, self.params.nV, self.params.nP,
                                    self.params.city))

    ##########
    #  PREP  #
    ##########

    def myinit(self, **kwargs):
        # part of init that is repeated every run
        self.update_decisions_and_params(**kwargs)

        self.make_skims()
        self.set_variabilities()
        self.env = simpy.Environment()  # simulation environment init
        self.t0 = self.inData.requests.treq.min()  # start at the first request time
        self.t1 = 60 * 60 * (self.params.simTime + 2)

        self.trips = list()  # report of trips
        self.rides = list()  # report of rides
        self.passengers = self.inData.passengers.copy()
        self.requests = initialize_df(self.inData.requests)  # init requests
        self.reqQ = list()  # queue of requests (traveller ids)
        self.vehQ = list()  # queue of idle vehicles (driver ids)
        self.pax = dict()  # list of passengers
        self.vehs = dict()  # list of vehicles
        self.plats = dict()  # list of platforms
        self.sim_start = None

    def generate(self):
        # generate passengers and vehicles as agents in the simulation (inData stays intact)
        for platform_id in self.platforms.index:
            self.plats[platform_id] = PlatformAgent(self, platform_id)
        for pax_id in self.inData.passengers.index:
            self.pax[pax_id] = PassengerAgent(self, pax_id)
        for veh_id in self.vehicles.index:
            self.vehs[veh_id] = VehicleAgent(self, veh_id)

    #########
    #  RUN  #
    #########

    def simulate(self, run_id=None):
        # run
        self.sim_start = time.time()
        self.logger.info("-------------------\tStarting simulation\t-------------------")

        self.env.run(until=self.t1)  # main run sim time + cool_down
        self.sim_end = time.time()
        self.logger.info("-------------------\tSimulation over\t\t-------------------")
        if len(self.reqQ) >= 0:
            self.logger.info(f"queue of requests {len(self.reqQ)}")
        self.logger.warning(f"simulation time {round(self.sim_end - self.sim_start, 1)} s")
        self.make_res(run_id)
        if self.params.get('assert_me', True):
            self.assert_me()  # test consistency of results

    def make_and_run(self, run_id=None, **kwargs):
        # wrapper for the simulation routine
        self.myinit(**kwargs)
        self.generate()
        self.simulate(run_id=run_id)

    ############
    #  OUTPUT  #
    ############

    def make_res(self, run_id):
        # called at the end of simulation
        if run_id == None:
            if len(self.run_ids) > 0:
                run_id = self.run_ids[-1] + 1
            else:
                run_id = 0
        self.run_ids.append(run_id)
        trips = pd.concat([pd.DataFrame(self.pax[pax].rides) for pax in self.pax.keys()])
        outcomes = [self.pax[pax].rides[-1]['event'] for pax in self.pax.keys()]
        rides = pd.concat([pd.DataFrame(self.vehs[pax].myrides) for pax in self.vehs.keys()])
        queues = pd.concat([pd.DataFrame(self.plats[plat].Qs,
                                         columns=['t', 'platform', 'vehQ', 'reqQ'])
                            for plat in self.plats]).set_index('t')

        self.runs[run_id] = DotMap({'trips': trips, 'outcomes': outcomes, 'rides': rides, 'queues': queues})

    def output(self, run_id=None):
        # called after the run for refined results
        run_id = self.run_ids[-1] if run_id is None else run_id
        ret = self.functions.kpi_pax(sim = self, run_id = run_id)
        veh = self.functions.kpi_veh(sim = self, run_id = run_id)
        ret.update(veh)
        self.res[run_id] = DotMap(ret)

    #########
    # UTILS #
    #########
    def init_log(self, **kwargs):
        logger = kwargs.get('logger', None)
        level = kwargs.get('logger_level', logging.INFO)
        if logger is None:
            logging.basicConfig(stream=sys.stdout, format='%(asctime)s-%(levelname)s-%(message)s',
                                datefmt='%d-%m-%y %H:%M:%S', level=level)

            logger = logging.getLogger()
            logger.setLevel(level)
            return logging.getLogger(__name__)
        else:
            logger.setLevel(level)
            return logger

    def print_now(self):
        return self.t0 + pd.Timedelta(self.env.now, 's')

    def assert_me(self):
        # try:
        # basic checks for results consistency and correctness
        rides = self.runs[0].rides  # vehicles record
        trips = self.runs[0].trips  # travellers record
        for i in self.inData.passengers.sample(min(5, self.inData.passengers.shape[0])).index.to_list():
            r = self.inData.requests[self.inData.requests.pax_id == i].iloc[0].squeeze()  # that is his request
            o, d = r['origin'], r['destination']  # his origin and destination
            trip = trips[trips.pax == i]  # his trip
            assert o in trip.pos.values  # was he at origin
            if travellerEvent.ARRIVES_AT_DEST.name in trip.event.values:
                # succesful trip
                assert d in trip.pos.values  # did he reach the destination
                veh = trip.veh_id.dropna().unique()  # did he travel with vehicle
                assert len(veh) == 1  # was there just one vehicle (should be)
                ride = rides[rides.veh == veh[0]]
                assert i in list(
                    set([item for sublist in ride.paxes.values for item in sublist]))  # was he assigned to a vehicle
                common_pos = list(set(list(ride.pos.values) + list(trip.pos.values)))
                assert len(common_pos) >= 2  # were there at least two points in common
                for pos in common_pos:
                    assert len(set(ride[ride.pos == pos].t.to_list() + trip[
                        trip.pos == pos].t.to_list())) > 0  # were they at the same time at the same place?
                if not self.vars.ride:
                    # check travel times
                    length = int(nx.shortest_path_length(self.inData.G, o, d, weight='length') / self.params.speeds.ride)
                    skim = self.skims.ride[o][d]
                    assert abs(skim - length) < 3

            else:
                # unsuccesful trip
                flag = False
                if travellerEvent.LOSES_PATIENCE.name in trip.event.values:
                    flag = True
                elif travellerEvent.IS_REJECTED_BY_VEHICLE.name in trip.event.values:
                    flag = True
                elif travellerEvent.REJECTS_OFFER.name in trip.event.values:
                    flag = True
                elif travellerEvent.ARRIVES_AT_PICKUP.name in trip.event.values:
                    flag = True  # still to be handled - what happens if traveller waits and simulation is over
                try:
                    assert flag is True
                except AssertionError:
                    print(trip)
                    assert flag is True
        self.logger.warning('assertion tests for simulation results - passed')
        # except:
        #     self.logger.info('assertion tests for simulation results - failed')
        #     swwssw

    def dump(self, path=None, dump_id=None, inputs=True, results=True):
        """
        stores resulting files into .zip folder
        :param path:
        :param id: run id
        :param inputs: store input files (vehicles, passengers, platforms)
        :param results: stor output files (trips, rides, veh, pax KPIs)
        :return: zip file
        """
        if path is None:
            path = os.getcwd()
        Path(path).mkdir(parents=True, exist_ok=True)
        dump_id = self.run_ids[-1] if dump_id is None else dump_id

        with zipfile.ZipFile(os.path.join(path, 'res{}.zip'.format(dump_id)), 'w') as csv_zip:
            if inputs:
                for data in ['vehicles', 'passengers', 'requests', 'platforms']:
                    csv_zip.writestr("{}.csv".format(data), self.inData[data].to_csv())
            if results:
                csv_zip.writestr("{}.csv".format('trips'), self.runs[0].trips.to_csv())
                csv_zip.writestr("{}.csv".format('rides'), self.runs[0].rides.to_csv())
                for key in self.res[0].keys():
                    csv_zip.writestr("{}.csv".format(key), self.res[0][key].to_csv())
        return csv_zip

    def update_decisions_and_params(self, **kwargs):
        self.defaults.update(kwargs)  # update defaults with kwargs
        self.params = self.defaults['params']  # json dict with parameters

        # populate functions
        self.functions = DotMap()
        for f in self.defaults.keys():
            if f in self.FNAMES:
                self.functions[f] = self.defaults[f]

        if self.functions.timeout is None:
            self.functions.timeout = self.timeout

    def make_skims(self):
        # uses distance skim in meters to populate 3 skims used in simulations
        self.skims = DotMap()
        self.skims.dist = self.inData.skim.copy()
        self.skims.ride = self.skims.dist.divide(self.params.speeds.ride).astype(int).T  # <---- here we have travel time
        self.skims.walk = self.skims.dist.divide(self.params.speeds.walk).astype(int).T  # <---- here we have travel time

    def timeout(self, n, variability=False):
        # overwrites sim timeout to add potential stochasticity
        if variability:
            n = np.random.normal(n, math.sqrt(n * variability))  # normal
        return self.env.timeout(n)

    def set_variabilities(self):
        self.vars = DotMap()
        self.vars.walk = False
        self.vars.start = False
        self.vars.request = False
        self.vars.transaction = False
        self.vars.ride = False
        self.vars.pickup = False
        self.vars.dropoff = False
        self.vars.shift = False
        self.vars.pickup_patience = False

    def plot_trip(self, pax_id, run_id=None):
        from MaaSSim.visualizations import plot_trip
        plot_trip(self,pax_id, run_id = run_id)

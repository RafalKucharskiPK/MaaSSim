#!/usr/bin/env python

__author__ = "Rafal Kucharski"
__email__ = "r.m.kucharski@tudelft.nl"
__license__ = "MIT"

import unittest
import networkx as nx
import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add local path for Travis CI


class TestSimulationResults(unittest.TestCase):
    """
    Basic tests
    """

    def test_results(self):
        """
        Runs MaaSSim and inspects the results sim.runs[0].trips *.rides etc.
        """
        from MaaSSim.simulators import simulate as this_simulator

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        # tests if results are as expected
        sim = this_simulator(config=CONFIG_PATH, root_path=os.path.dirname(__file__))  # run simulations

        self.assertIn('trips', sim.runs[0].keys())  # do we have the results

        self.assertLess(sim.sim_end - sim.sim_start, 20)  # does the calculation take a lot of time?

        self.assertGreater(sim.runs[0].trips.pos.nunique(), sim.params.nP)  # do we travel around the city

        self.assertGreater(sim.runs[0].trips.t.max(),
                           0.5 * sim.params.simTime * 60 * 60)  # do we span at least half of simulation time?)

        paxes = sim.runs[0].rides.paxes.apply(lambda x: tuple(x))
        self.assertGreater(paxes.nunique(), sim.params.nP / 2)  # at least half of travellers got the ride

        self.assertGreater(sim.res[0].veh_exp["ARRIVES_AT_PICKUP"].max(), 0)  # is there any vehicle RIDING?

        self.assertIn('pax_exp', sim.res[0].keys())
        del sim

    def test_consistency(self):
        """
        Runs MaaSSim and inspects the consistency of simulation for randomly selected few travellers
        """
        from MaaSSim.simulators import simulate as this_simulator
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_consistency_test.json')
        sim = this_simulator(config=CONFIG_PATH, root_path=os.path.dirname(__file__))  # run simulations

        from MaaSSim.traveller import travellerEvent

        rides = sim.runs[0].rides  # vehicles results
        trips = sim.runs[0].trips  # travellers result
        for i in sim.inData.passengers.sample(min(5, sim.inData.passengers.shape[0])).index.to_list():
            r = sim.inData.requests[sim.inData.requests.pax_id == i].iloc[0].squeeze()  # that is his request
            o, d = r['origin'], r['destination']  # his origin and destination
            trip = trips[trips.pax == i]  # his trip
            self.assertIn(o, trip.pos.values)  # was he at origin
            if travellerEvent.ARRIVES_AT_DEST.name in trip.event.values:
                # successful trip
                self.assertIn(d, trip.pos.values)  # did he reach the destination
                veh = trip.veh_id.dropna().unique()  # did he travel with vehicle
                self.assertAlmostEqual(len(veh), 1)  # was there just one vehicle (should be)
                ride = rides[rides.veh == veh[0]]
                self.assertIn(i, list(
                    set([item for sublist in ride.paxes.values for item in sublist])))  # was he assigned to a vehicle
                common_pos = list(set(list(ride.pos.values) + list(trip.pos.values)))
                self.assertGreaterEqual(len(common_pos), 2)  # were there at least two points in common
                for pos in common_pos:
                    self.assertGreater(len(set(ride[ride.pos == pos].t.to_list() + trip[
                        trip.pos == pos].t.to_list())), 0)  # were they at the same time at the same place?
                if not sim.vars.ride:
                    # check travel times
                    length = int(nx.shortest_path_length(sim.inData.G, o, d, weight='length') /
                                 sim.params.speeds.ride)
                    skim = sim.skims.ride[o][d]
                    assert abs(skim - length) < 3
            else:
                # unsuccessful trip
                flag = False
                if travellerEvent.LOSES_PATIENCE.name in trip.event.values:
                    flag = True
                elif travellerEvent.IS_REJECTED_BY_VEHICLE.name in trip.event.values:
                    flag = True
                elif travellerEvent.REJECTS_OFFER.name in trip.event.values:
                    flag = True
                try:
                    self.assertTrue(flag)
                except AssertionError:
                    print(trip.event.values.unique())
                    self.assertTrue(flag)

        del sim

    def test_prep(self):
        """
        tests if simulation is prepared properly (reuests, passengers, vehicles are generated)
        """
        from MaaSSim.utils import prep_supply_and_demand, get_config, load_G

        from MaaSSim.data_structures import structures as inData
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        inData = load_G(inData, params)  # load network graph
        inData = prep_supply_and_demand(inData, params)  # generate supply and demand
        self.assertEqual(inData.requests.shape[0], params.nP)
        self.assertEqual(inData.passengers.shape[0], params.nP)
        self.assertEqual(inData.vehicles.shape[0], params.nV)

    def test_staticIO(self):
        """
        test if simulation can be restarted from static .csv file and yield the same results
        """
        from MaaSSim.utils import prep_supply_and_demand, get_config, load_G

        from MaaSSim.simulators import simulate as this_simulator
        from MaaSSim.data_structures import structures as inData
        from MaaSSim.utils import read_requests_csv, read_vehicle_positions
        from pandas.testing import assert_frame_equal

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        inData = load_G(inData, params)  # load network graph
        inData = prep_supply_and_demand(inData, params)  # generate supply and demand
        inData.requests.to_csv('requests.csv')
        inData.vehicles.to_csv('vehicles.csv')
        sim1 = this_simulator(params=params, inData=inData)  # simulate
        inData = read_requests_csv(inData, path='requests.csv')
        inData = read_vehicle_positions(inData, path='vehicles.csv')
        sim2 = this_simulator(params=params, inData=inData)  # simulate

        assert_frame_equal(sim2.runs[0].trips, sim2.runs[0].trips)
        assert_frame_equal(sim2.runs[0].rides, sim2.runs[0].rides)
        # self.assertTrue(sim2.runs[0].rides.equals(sim1.runs[0].rides))
        del sim2
        del sim1

    def test_parallel(self):
        """
        running parallel experiments on the multidimensional search space
        """
        from dotmap import DotMap
        from MaaSSim.utils import collect_results
        from MaaSSim.utils import get_config

        from MaaSSim.simulators import simulate_parallel as this_parallel_simulator
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_parallel_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        search_space = DotMap()
        search_space.nP = [20, 40]
        search_space.nV = [20, 40]

        this_parallel_simulator(config=CONFIG_PATH, search_space=search_space, root_path=os.path.dirname(__file__))

        res = collect_results(params.paths.dumps)  # collect the results from multiple experiments
        self.assertNotEqual(res.rides.t.mean(), res.rides.t.std())
        self.assertNotEqual(res.trips.t.mean(), res.trips.t.std())

    def tearDown(self):
        zips = glob.glob('*.{}'.format('zip'))
        for zip in zips:
            os.remove(zip)


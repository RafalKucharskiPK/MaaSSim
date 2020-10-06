#!/usr/bin/env python

__author__ = "Rafal Kucharski"
__email__ = "r.m.kucharski@tudelft.nl"
__license__ = "MIT"

import unittest
import os
import sys
import glob



sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add local path for Travis CI
from MaaSSim.simulators import simulate, simulate_parallel


class TestSimulationResults(unittest.TestCase):

    def test_results(self):
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        # tests if results are as expected
        self.sim = simulate(config=CONFIG_PATH, root_path=os.path.dirname(__file__))  # run simulations

        self.assertIn('trips', self.sim.runs[0].keys())  # do we have the results

        self.assertLess(self.sim.sim_end - self.sim.sim_start, 20)  # does the calculation take a lot of time?

        self.assertGreater(self.sim.runs[0].trips.pos.nunique(), self.sim.params.nP)  # do we travel around the city

        self.assertGreater(self.sim.runs[0].trips.t.max(),
                           0.5 * self.sim.params.simTime * 60 * 60) # do we span at least half of simulation time?)

        paxes = self.sim.runs[0].rides.paxes.apply(lambda x: tuple(x))
        self.assertGreater(paxes.nunique(), self.sim.params.nP/2)  # at least half of travellers got the ride

        self.assertGreater(self.sim.res[0].veh_exp["ARRIVES_AT_PICKUP"].max(), 0)  # is there any vehicle RIDING?

        self.assertIn('pax_exp', self.sim.res[0].keys())

    def test_consistency(self):
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_consistency_test.json')
        self.sim = simulate(config=CONFIG_PATH, root_path=os.path.dirname(__file__))  # run simulations

        from MaaSSim.traveller import travellerEvent

        rides = self.sim.runs[0].rides  # vehicles results
        trips = self.sim.runs[0].trips  # travellers result
        for i in self.sim.inData.passengers.sample(min(5, self.sim.params.nP)).index.to_list():
            r = self.sim.inData.requests[self.sim.inData.requests.pax_id == i].iloc[0].squeeze()  # that is his request
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
            else:
                # unsuccessful trip
                flag = False
                if travellerEvent.LOSES_PATIENCE.name in trip.event.values:
                    flag = True
                elif travellerEvent.IS_REJECTED_BY_VEHICLE.name in trip.event.values:
                    flag = True
                elif travellerEvent.REJECTS_OFFER.name in trip.event.values:
                    flag = True
                self.assertTrue(flag)

    def test_parallel(self):
        from dotmap import DotMap
        search_space = DotMap()
        search_space.nP = [20, 40]
        search_space.nV = [20, 40]
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')

        simulate_parallel(config = CONFIG_PATH, search_space=search_space, root_path=os.path.dirname(__file__))

    def tearDown(self):
        zips = glob.glob('*.{}'.format('zip'))

        for zip in zips:
            os.remove(zip)




    # def test_static_input(self):
    #     CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_static_input.json')
    #     # tests if results are as expected
    #     self.sim = simulate(config=CONFIG_PATH, root_path=os.path.dirname(__file__))  # run simulations


class TestUtils(unittest.TestCase):


    def setUp(self):
        from MaaSSim.data_structures import structures
        from MaaSSim.utils import get_config


        self.inData = structures.copy()  # fresh data
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_utils_test.json')
        self.params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file


    def test_networkIO(self):
        from numpy import inf
        from MaaSSim.utils import load_G, download_G, save_G

        self.params.city = 'Wieliczka, Poland'

        self.params.paths.G = os.path.join(os.path.dirname(__file__), self.params.city.split(",")[0] + ".graphml")  # graphml of a current .city
        self.params.paths.skim = os.path.join(os.path.dirname(__file__), self.params.city.split(",")[0] + ".csv")  # csv with a skim between the nodes of the .city


        self.inData = download_G(self.inData, self.params) # download the graph and compute the skim
        save_G(self.inData, self.params) # save it to params.paths.G
        self.inData = load_G(self.inData, self.params, stats=True)  # download graph for the 'params.city' and calc the skim matrices

        self.assertGreater(self.inData.nodes.shape[0],10)  # do we have nodes
        self.assertGreater(self.inData.skim.shape[0], 10)  # do we have skim
        self.assertLess(self.inData.skim.mean().mean(), inf)  # and values inside
        self.assertGreater(self.inData.skim.mean().mean(),0)  # positive distances

    def tearDown(self):
        os.remove(self.params.paths.G)
        os.remove(self.params.paths.skim)

class TestJupyters(unittest.TestCase):
    def setUp(self):
        from nbconvert.preprocessors import ExecutePreprocessor
        self.ep = ExecutePreprocessor(timeout=600, kernel_name='python3')


    def test_tutorials(self):
        import nbformat

        os.chdir("../docs/tutorials")

        notebooks = glob.glob('*.{}'.format('ipynb'))

        #tutorials
        for notebook in notebooks:
            if notebook[0] == '0':
                print('testing: ',notebook)
                with open(notebook) as f:
                    nb = nbformat.read(f, as_version=4)
                self.ep.preprocess(nb)

        # appendices
        for notebook in notebooks:
            if notebook[0] == 'A':
                print('testing: ', notebook)
                with open(notebook) as f:
                    nb = nbformat.read(f, as_version=4)
                self.ep.preprocess(nb)






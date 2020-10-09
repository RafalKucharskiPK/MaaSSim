#!/usr/bin/env python

__author__ = "Rafal Kucharski"
__email__ = "r.m.kucharski@tudelft.nl"
__license__ = "MIT"

import unittest
import os
import sys
import glob
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add local path for Travis CI
from MaaSSim.simulators import simulate, simulate_parallel
from MaaSSim.utils import prep_supply_and_demand, get_config, load_G, generate_demand, generate_vehicles, initialize_df
from MaaSSim.traveller import f_platform_choice


class TestSimulationResults(unittest.TestCase):
    """
    Basic tests
    """

    def test_results(self):
        """
        Runs MaaSSim and inspects the results sim.runs[0].trips *.rides etc.
        """
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        # tests if results are as expected
        self.sim = simulate(config=CONFIG_PATH, root_path=os.path.dirname(__file__))  # run simulations

        self.assertIn('trips', self.sim.runs[0].keys())  # do we have the results

        self.assertLess(self.sim.sim_end - self.sim.sim_start, 20)  # does the calculation take a lot of time?

        self.assertGreater(self.sim.runs[0].trips.pos.nunique(), self.sim.params.nP)  # do we travel around the city

        self.assertGreater(self.sim.runs[0].trips.t.max(),
                           0.5 * self.sim.params.simTime * 60 * 60)  # do we span at least half of simulation time?)

        paxes = self.sim.runs[0].rides.paxes.apply(lambda x: tuple(x))
        self.assertGreater(paxes.nunique(), self.sim.params.nP / 2)  # at least half of travellers got the ride

        self.assertGreater(self.sim.res[0].veh_exp["ARRIVES_AT_PICKUP"].max(), 0)  # is there any vehicle RIDING?

        self.assertIn('pax_exp', self.sim.res[0].keys())

    def test_consistency(self):
        """
        Runs MaaSSim and inspects the consistency of simulation for randomly selected few travellers
        """
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_consistency_test.json')
        self.sim = simulate(config=CONFIG_PATH, root_path=os.path.dirname(__file__))  # run simulations

        from MaaSSim.traveller import travellerEvent

        rides = self.sim.runs[0].rides  # vehicles results
        trips = self.sim.runs[0].trips  # travellers result
        for i in self.sim.inData.passengers.sample(min(5, self.sim.inData.passengers.shape[0])).index.to_list():
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

    def test_prep(self):
        """
        tests if simulation is prepared properly (reuests, passengers, vehicles are generated)
        """
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
        from MaaSSim.data_structures import structures as inData
        from MaaSSim.utils import read_requests_csv, read_vehicle_positions
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        inData = load_G(inData, params)  # load network graph
        inData = prep_supply_and_demand(inData, params)  # generate supply and demand
        inData.requests.to_csv('requests.csv')
        inData.vehicles.to_csv('vehicles.csv')
        sim1 = simulate(params=params, inData=inData)  # simulate
        inData = read_requests_csv(inData, path='requests.csv')
        inData = read_vehicle_positions(inData, path='vehicles.csv')
        sim2 = simulate(params=params, inData=inData)  # simulate
        self.assertTrue(sim2.runs[0].trips.equals(sim1.runs[0].trips))
        self.assertTrue(sim2.runs[0].rides.equals(sim1.runs[0].rides))

    def test_parallel(self):
        """
        running parallel experiments on the multidimensional search space
        """
        from dotmap import DotMap
        from MaaSSim.utils import collect_results

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_parallel_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        search_space = DotMap()
        search_space.nP = [20, 40]
        search_space.nV = [20, 40]

        simulate_parallel(config=CONFIG_PATH, search_space=search_space, root_path=os.path.dirname(__file__))

        res = collect_results(params.paths.dumps)  # collect the results from multiple experiments
        self.assertNotEqual(res.rides.t.mean(), res.rides.t.std())
        self.assertNotEqual(res.trips.t.mean(), res.trips.t.std())

    def tearDown(self):
        zips = glob.glob('*.{}'.format('zip'))
        for zip in zips:
            os.remove(zip)


class TestFunctionalities(unittest.TestCase):
    """
    Test of MaaSSim capabilities, functionalities, extra features
    """

    def test_batch_platform(self):
        """
        test if platform batches the results properly at matching
        """
        from MaaSSim.utils import initialize_df

        from MaaSSim.data_structures import structures as inData
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        inData = load_G(inData, params)  # load network graph
        inData = prep_supply_and_demand(inData, params)  # generate supply and demand
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0] = [1, 'Uber', 600]  # batch requests every 600 seconds

        sim = simulate(params=params, inData=inData, event_based=False)
        Qs = sim.runs[0].queues
        Qs.groupby('platform')[['vehQ', 'reqQ']].plot(drawstyle='steps-post')

        r = sim.runs[0].trips
        times = r[r.event == 'RECEIVES_OFFER'].t.sort_values(ascending=True).diff().dropna().unique()

        self.assertEqual(len(times), 2)  # are requests batched ony at batch_time
        self.assertEqual(times.max(), inData.platforms.iloc[0].batch_time)  # are they batched at batch_time

    def test_early_ending_shift_and_losing_patience(self):
        """
        test if platform batches the results properly at matching
        """
        from MaaSSim.traveller import travellerEvent
        from MaaSSim.driver import driverEvent

        from MaaSSim.data_structures import structures as inData
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        params.nP = 2
        params.nV = 1
        params.times.patience = 30
        inData = load_G(inData, params)  # load network graph
        inData = prep_supply_and_demand(inData, params)  # generate supply and demand
        inData.vehicles.shift_end = [100]  # vehicle ends early

        sim = simulate(params=params, inData=inData, event_based=False)

        self.assertIn(travellerEvent.LOSES_PATIENCE.name, sim.runs[0].trips.event.values)  # one traveller lost patience

        r = sim.runs[0].rides
        self.assertLess(r[r.event == driverEvent.ENDS_SHIFT.name].t.squeeze(), sim.t1)  # did he really end earlier

    def test_multiple_runs(self):
        # have two runs in the same instance and see if they are different

        from MaaSSim.data_structures import structures as inData
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')

        def rand_reject8(**kwargs):
            # sample function to reject with probability of 80%
            return random.random() >= 0.2

        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file

        params.times.patience = 3600  # 1 hour of simulation
        params.simTime = 1  # 1 hour of simulation
        params.nP = 100  # reuqests (and passengers)
        params.nV = 100  # vehicles

        # A no rejections
        sim = simulate(params=params, inData=inData)
        self.assertEqual(len(sim.runs), 1)
        sim.make_and_run(f_driver_decline=rand_reject8)  # change something and re run
        self.assertEqual(len(sim.runs), 2)
        self.assertNotEqual(sim.runs[0].trips.values, sim.runs[1].trips.values)
        self.assertNotEqual(sim.runs[0].trips.values, sim.runs[1].trips.values)

    def test_rejects(self):
        # make sure that rejection works for drivers and travellers (dummy reject with fixed probability)
        from MaaSSim.utils import dummy_False
        from MaaSSim.traveller import travellerEvent
        from MaaSSim.driver import driverEvent
        from MaaSSim.data_structures import structures as inData
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')

        def rand_reject8(**kwargs):
            # sample function to reject with probability of 80%
            return random.random() >= 0.2

        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file

        params.times.patience = 3600  # 1 hour of simulation
        params.simTime = 1  # 1 hour of simulation
        params.nP = 100  # reuqests (and passengers)
        params.nV = 100  # vehicles

        # A no rejections
        sim = simulate(params=params, inData=inData)
        self.assertNotIn(travellerEvent.IS_REJECTED_BY_VEHICLE.name,
                         sim.runs[0].trips.event.values)  # no rejections
        self.assertNotIn(travellerEvent.REJECTS_OFFER.name,
                         sim.runs[0].trips.event.values)  # no rejections

        self.assertNotIn(driverEvent.REJECTS_REQUEST.name,
                         sim.runs[0].rides.event.values)  # no rejections
        self.assertNotIn(driverEvent.IS_REJECTED_BY_TRAVELLER.name,
                         sim.runs[0].rides.event.values)  # no rejections

        # B vehicle rejects
        sim.make_and_run(f_trav_mode=dummy_False, f_driver_decline=rand_reject8)
        self.assertIn(travellerEvent.IS_REJECTED_BY_VEHICLE.name,
                      sim.runs[1].trips.event.values)  # no rejections
        self.assertNotIn(travellerEvent.REJECTS_OFFER.name,
                         sim.runs[1].trips.event.values)  # no rejections

        self.assertIn(driverEvent.REJECTS_REQUEST.name,
                      sim.runs[1].rides.event.values)  # no rejections
        self.assertNotIn(driverEvent.IS_REJECTED_BY_TRAVELLER.name,
                         sim.runs[1].rides.event.values)  # no rejections

        # # C traveller rejects
        sim.make_and_run(f_trav_mode=rand_reject8, f_driver_decline=dummy_False)
        self.assertNotIn(travellerEvent.IS_REJECTED_BY_VEHICLE.name,
                         sim.runs[2].trips.event.values)  # no rejections
        self.assertIn(travellerEvent.REJECTS_OFFER.name,
                      sim.runs[2].trips.event.values)  # no rejections

        self.assertNotIn(driverEvent.REJECTS_REQUEST.name,
                         sim.runs[2].rides.event.values)  # no rejections
        self.assertIn(driverEvent.IS_REJECTED_BY_TRAVELLER.name,
                      sim.runs[2].rides.event.values)  # no rejections
        #
        # # D both reject
        sim.make_and_run(f_trav_mode=rand_reject8, f_driver_decline=rand_reject8)
        self.assertIn(travellerEvent.IS_REJECTED_BY_VEHICLE.name,
                      sim.runs[3].trips.event.values)  # no rejections
        self.assertIn(travellerEvent.REJECTS_OFFER.name,
                      sim.runs[3].trips.event.values)  # no rejections

        self.assertIn(driverEvent.REJECTS_REQUEST.name,
                      sim.runs[3].rides.event.values)  # no rejections
        self.assertIn(driverEvent.IS_REJECTED_BY_TRAVELLER.name,
                      sim.runs[3].rides.event.values)  # no rejections

    def test_platform_competition(self):
        # make sure when you compete with the prcie, lowering the fare and increasing the fee
        from MaaSSim.data_structures import structures as inData

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_platform_choices.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        params.nP = 50  # reuqests (and passengers)
        params.simTime = 4
        params.nD = 1

        fare = 1
        fleet = 20
        params.nV = 20 + fleet
        inData = load_G(inData, params,
                        stats=True)  # download graph for the 'params.city' and calc the skim matrices
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0] = [1, 'Platform1', 30]
        inData.platforms.loc[1] = [fare, 'Platform2', 30]
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.vehicles.platform = [0] * 20 + [1] * fleet

        inData.passengers.platforms = inData.passengers.apply(lambda x: [0, 1], axis=1)
        sim = simulate(params=params, inData=inData, print=False, f_platform_choice=f_platform_choice)
        ret = sim.res[0].veh_exp
        ret['platform'] = inData.vehicles.platform
        first = ret[ret.platform == 1].ARRIVES_AT_DROPOFF.sum()

        fare = 0.5
        fleet = 30
        params.nV = 20 + fleet
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0] = [1, 'Platform1', 30]
        inData.platforms.loc[1] = [fare, 'Platform2', 30]
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.vehicles.platform = [0] * 20 + [1] * fleet

        inData.passengers.platforms = inData.passengers.apply(lambda x: [0, 1], axis=1)
        sim = simulate(params=params, inData=inData, print=False, f_platform_choice=f_platform_choice)
        ret = sim.res[0].veh_exp
        ret['platform'] = inData.vehicles.platform
        second = ret[ret.platform == 1].ARRIVES_AT_DROPOFF.sum()
        self.assertGreater(second,first)

class TestUtils(unittest.TestCase):
    """
    test input, output, utils, etc.
    """

    def setUp(self):
        from MaaSSim.data_structures import structures
        from MaaSSim.utils import get_config, make_config_paths

        self.inData = structures.copy()  # fresh data
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_utils_test.json')
        self.params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file

    def test_configIO(self):
        from MaaSSim.utils import make_config_paths
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_utils_test.json')

        from MaaSSim.utils import get_config, save_config
        self.params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        params = make_config_paths(self.params, main='test_path', rel=True)
        self.assertEqual(params.paths.G[0:9], 'test_path')
        self.params.testIO = '12'
        save_config(self.params, os.path.join(os.path.dirname(__file__), 'configIO_test.json'))
        params = get_config(os.path.join(os.path.dirname(__file__), 'configIO_test.json'),
                            root_path=os.path.dirname(__file__))  # load from .json file
        self.assertEqual(params.testIO, self.params.testIO)

    def test_networkIO(self):
        from numpy import inf
        from MaaSSim.utils import load_G, download_G, save_G

        self.params.city = 'Wieliczka, Poland'

        self.params.paths.G = os.path.join(os.path.dirname(__file__),
                                           self.params.city.split(",")[0] + ".graphml")  # graphml of a current .city
        self.params.paths.skim = os.path.join(os.path.dirname(__file__), self.params.city.split(",")[
            0] + ".csv")  # csv with a skim between the nodes of the .city

        self.inData = download_G(self.inData, self.params)  # download the graph and compute the skim
        save_G(self.inData, self.params)  # save it to params.paths.G
        self.inData = load_G(self.inData, self.params,
                             stats=True)  # download graph for the 'params.city' and calc the skim matrices

        self.assertGreater(self.inData.nodes.shape[0], 10)  # do we have nodes
        self.assertGreater(self.inData.skim.shape[0], 10)  # do we have skim
        self.assertLess(self.inData.skim.mean().mean(), inf)  # and values inside
        self.assertGreater(self.inData.skim.mean().mean(), 0)  # positive distances

    def tearDown(self):
        pass
        # os.remove(self.params.paths.G)
        # os.remove(self.params.paths.skim)

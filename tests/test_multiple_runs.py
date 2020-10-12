import unittest
import os
import random

class TestMultipleRuns(unittest.TestCase):
    def test_multiple_runs(self):
        # simulate two runs in the same instance and see if they are different

        from MaaSSim.data_structures import structures as inData
        from MaaSSim.simulators import simulate as multiple_simulator_tester
        from MaaSSim.utils import get_config

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
        self.sim = multiple_simulator_tester(params=params, inData=inData)
        self.assertEqual(len(self.sim.runs), 1)
        self.sim.make_and_run(f_driver_decline=rand_reject8)  # change something and re run
        self.assertEqual(len(self.sim.runs), 2)
        self.assertNotEqual(self.sim.runs[0].trips.values, self.sim.runs[1].trips.values)
        self.assertNotEqual(self.sim.runs[0].trips.values, self.sim.runs[1].trips.values)

        del self.sim
        del params
        del inData
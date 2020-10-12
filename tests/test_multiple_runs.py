import unittest
import os
import random
from pandas.testing import assert_frame_equal

def assert_frame_not_equal(*args, **kwargs):
    try:
        assert_frame_equal(*args, **kwargs)
    except AssertionError:
        # frames are not equal
        pass
    else:
        # frames are equal
        raise AssertionError

class TestMultipleRuns(unittest.TestCase):
    def test_multiple_runs(self):
        # simulate two runs in the same instance and see if they are different

        from MaaSSim.data_structures import structures as inData
        from MaaSSim.driver import f_dummy_repos
        from MaaSSim.simulators import simulate as multiple_simulator_tester
        from MaaSSim.utils import get_config

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')

        def rand_reject8(**kwargs):
            # sample function to reject with probability of 80%
            return random.random() >= 0.5

        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file

        params.times.patience = 3600  # 1 hour of simulation
        params.simTime = 1  # 1 hour of simulation
        params.nP = 100  # reuqests (and passengers)
        params.nV = 100  # vehicles


        # A no rejections
        params.times.transaction = 30
        sim = multiple_simulator_tester(params=params, inData=inData, f_driver_repos=f_dummy_repos, event_based=True)
        self.assertEqual(len(sim.runs), 1)
        sim.params.times.transaction = 60
        sim.make_and_run()  # change something and re run
        self.assertEqual(len(sim.runs), 2)
        assert_frame_not_equal(sim.runs[0].trips, sim.runs[1].trips)
        assert_frame_not_equal(sim.runs[0].rides, sim.runs[1].rides)


        del sim
        del params
        del inData
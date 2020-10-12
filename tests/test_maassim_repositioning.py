import unittest
import os

class TestDriverRepos(unittest.TestCase):
    def test_driver_repositioning(self):
        from MaaSSim.simulators import simulate as repos_simulator
        from MaaSSim.driver import f_repos, driverEvent
        from MaaSSim.utils import get_config


        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_platform_choices.json')
        self.params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        self.params.times.patience = 600  # 1 hour of simulation
        self.params.nP = 100  # reuqests (and passengers)
        self.params.nV = 50  # vehicles
        self.params.simTime = 4
        self.params.nD = 1
        self.sim1 = repos_simulator(params=self.params)
        self.assertNotIn(driverEvent.STARTS_REPOSITIONING.name, self.sim1.runs[0].rides.event.values)  # no rejections
        del self.sim1
        self.sim2 = repos_simulator(params=self.params, f_driver_repos=f_repos)
        self.assertIn(driverEvent.STARTS_REPOSITIONING.name, self.sim2.runs[0].rides.event.values)  # no rejections


    def tearDown(self):
        self.sim1 = None
        self.sim2 = None
        self.params = None
#
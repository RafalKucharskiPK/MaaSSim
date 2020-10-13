import unittest
import os

class TestDriverRepos(unittest.TestCase):
    def test_driver_repositioning(self):
        from MaaSSim.simulators import simulate as repos_simulator
        from MaaSSim.driver import driverEvent
        from MaaSSim.decisions import f_repos
        from MaaSSim.utils import get_config


        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_platform_choices.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        params.times.patience = 600  # 1 hour of simulation
        params.nP = 100  # reuqests (and passengers)
        params.nV = 50  # vehicles
        params.simTime = 4
        params.nD = 1
        sim1 = repos_simulator(params=params)
        self.assertNotIn(driverEvent.STARTS_REPOSITIONING.name, sim1.runs[0].rides.event.values)  # no rejections
        del sim1
        sim2 = repos_simulator(params=params, f_driver_repos=f_repos)
        self.assertIn(driverEvent.STARTS_REPOSITIONING.name, sim2.runs[0].rides.event.values)  # no rejections


        del sim2
        del params
#
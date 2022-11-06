import unittest
import os


class TestdriverDecline(unittest.TestCase):
    def test_driver_decline(self):
        from MaaSSim.simulators import simulate as simulator_driver_decl
        from MaaSSim.traveller import travellerEvent
        from MaaSSim.utils import get_config
        from MaaSSim.decisions import dummy_False
        from MaaSSim.decisions import f_decline

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_platform_choices.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        params.times.patience = 600  # 1 hour of simulation
        params.nP = 10  # reuqests (and passengers)
        params.nV = 10  # vehicles
        params.user_controlled_vehicles_count = 0
        params.simTime = 4
        params.nD = 1
        sim = simulator_driver_decl(params=params, f_driver_decline=f_decline)

        del sim

        del params
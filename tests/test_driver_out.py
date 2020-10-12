import unittest
import os

class TestdriverOut(unittest.TestCase):
    def test_driver_out(self):
        from MaaSSim.simulators import simulate as simulator_driver_our
        from MaaSSim.utils import dummy_False, get_config
        from MaaSSim.driver import f_driver_out

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_platform_choices.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        params.times.patience = 600  # 1 hour of simulation
        params.nP = 100  # reuqests (and passengers)
        params.nV = 50  # vehicles
        params.simTime = 4
        params.nD = 1
        sim = simulator_driver_our(params=params, f_driver_out=dummy_False)
        self.assertEqual(sim.res[0].veh_exp[sim.res[0].veh_exp.ENDS_SHIFT == 0].shape[0], 0)
        del sim
        params.nD = 2
        sim = simulator_driver_our(params=params, f_driver_out=f_driver_out)
        self.assertGreater(sim.res[1].veh_exp[sim.res[1].veh_exp.ENDS_SHIFT == 0].shape[0], 0) # did someone
        # end shift at beginning of simulation?

        del sim
        del params
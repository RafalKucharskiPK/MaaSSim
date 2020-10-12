import unittest
import os

class TestMultipleDays(unittest.TestCase):
    def test_multiple_days(self):
        # have two runs in the same instance and see if they are different
        from MaaSSim.data_structures import structures as my_inData
        from MaaSSim.traveller import f_platform_choice
        from MaaSSim.simulators import simulate as multiple_days_simulator
        from MaaSSim.utils import get_config, generate_vehicles, generate_demand, initialize_df, load_G

        from MaaSSim.simulators import simulate as simulator_for_multiple_days

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        self.params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file

        self.params.times.patience = 3600  # 1 hour of simulation
        self.params.simTime = 1  # 1 hour of simulation

        self.params.nP = 30  # reuqests (and passengers)
        self.params.nV = 30  # vehicles
        self.params.nD = 3

        self.inData = my_inData.copy()

        self.inData = load_G(self.inData, self.params,
                        stats=True)  # download graph for the 'params.city' and calc the skim matrices
        self.inData = generate_demand(self.inData, self.params, avg_speed=True)
        self.inData.vehicles = generate_vehicles(self.inData, self.params.nV)

        self.sim = multiple_days_simulator(params=self.params, inData=self.inData, print=False, f_platform_choice=f_platform_choice)

        self.assertEqual(len(self.sim.runs), 3)


    def tearDown(self):
        self.sim = None
        self.params = None

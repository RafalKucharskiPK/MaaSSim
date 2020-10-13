import unittest
import os

class TestMultipleDays(unittest.TestCase):
    def test_multiple_days(self):
        # have two runs in the same instance and see if they are different
        from MaaSSim.data_structures import structures as my_inData
        from MaaSSim.decisions import f_platform_choice
        from MaaSSim.simulators import simulate as multiple_days_simulator
        from MaaSSim.utils import get_config, generate_vehicles, generate_demand, initialize_df, load_G

        from MaaSSim.simulators import simulate as simulator_for_multiple_days

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file

        params.times.patience = 3600  # 1 hour of simulation
        params.simTime = 1  # 1 hour of simulation

        params.nP = 30  # reuqests (and passengers)
        params.nV = 30  # vehicles
        params.nD = 3

        inData = my_inData.copy()

        inData = load_G(inData, params,
                        stats=True)  # download graph for the 'params.city' and calc the skim matrices
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)

        sim = multiple_days_simulator(params=params, inData=inData, print=False, f_platform_choice=f_platform_choice)

        self.assertEqual(len(sim.runs), 3)


        del sim
        del params

import unittest
import os


class TestEarly(unittest.TestCase):
    def test_early_ending_shift_and_losing_patience(self):
        """
        test if platform batches the results properly at matching
        """
        from MaaSSim.traveller import travellerEvent
        from MaaSSim.driver import driverEvent
        from MaaSSim.data_structures import structures as this_inData_early
        from MaaSSim.simulators import simulate as simulator_early
        from MaaSSim.utils import prep_supply_and_demand, get_config, load_G

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        params.nP = 2
        params.nV = 1
        params.times.patience = 30
        this_inData_early = load_G(this_inData_early, params)  # load network graph
        this_inData_early = prep_supply_and_demand(this_inData_early, params)  # generate supply and demand
        this_inData_early.vehicles.shift_end = [100]  # vehicle ends early

        self.sim = simulator_early(params=params, inData=this_inData_early, event_based=False)

        self.assertIn(travellerEvent.LOSES_PATIENCE.name, self.sim.runs[0].trips.event.values)  # one traveller lost patience

        r = self.sim.runs[0].rides
        self.assertLess(r[r.event == driverEvent.ENDS_SHIFT.name].t.squeeze(), self.sim.t1)  # did he really end earlier


    def tearDown(self):
        self.sim = None
        self.params = None






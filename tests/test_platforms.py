import unittest
import os

class TestMultiplatform(unittest.TestCase):
    def test_platform_competition(self):
        # make sure when you compete with the prcie, lowering the fare and increasing the fee
        from MaaSSim.data_structures import structures as local_inData
        from MaaSSim.traveller import f_platform_choice
        from MaaSSim.simulators import simulate as platform_simulator_1
        from MaaSSim.utils import get_config, generate_vehicles, generate_demand, initialize_df, load_G


        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_platform_choices.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        params.nP = 200  # reuqests (and passengers)
        params.simTime = 4
        params.nD = 1

        fare = 1.5
        fleet = 10
        params.nV = 20 + fleet
        self.inData = local_inData.copy()
        self.inData = load_G(self.inData, params,
                        stats=True)  # download graph for the 'params.city' and calc the skim matrices
        self.inData = generate_demand(self.inData, params, avg_speed=True)
        self.inData.vehicles = generate_vehicles(self.inData, params.nV)
        self.inData.platforms = initialize_df(self.inData.platforms)
        self.inData.platforms.loc[0] = [1, 'Platform1', 30]
        self.inData.platforms.loc[1] = [fare, 'Platform2', 30]
        self.inData = generate_demand(self.inData, params, avg_speed=True)
        self.inData.vehicles = generate_vehicles(self.inData, params.nV)
        self.inData.vehicles.platform = [0] * 20 + [1] * fleet

        self.inData.passengers.platforms = self.inData.passengers.apply(lambda x: [0, 1], axis=1)
        self.sim = platform_simulator_1(params=params, inData=self.inData, f_platform_choice=f_platform_choice)
        ret = self.sim.res[0].veh_exp.copy()
        ret['platform'] = self.inData.vehicles.platform
        first = ret[ret.platform == 1].ARRIVES_AT_DROPOFF.sum()

        fare = 0.5
        fleet = 50
        params.nV = 20 + fleet
        self.inData = generate_demand(self.inData, params, avg_speed=True)
        self.inData.vehicles = generate_vehicles(self.inData, params.nV)
        self.inData.platforms = initialize_df(self.inData.platforms)
        self.inData.platforms.loc[0] = [1, 'Platform1', 30]
        self.inData.platforms.loc[1] = [fare, 'Platform2', 30]
        self.inData = generate_demand(self.inData, params, avg_speed=True)
        self.inData.vehicles = generate_vehicles(self.inData, params.nV)
        self.inData.vehicles.platform = [0] * 20 + [1] * fleet

        self.inData.passengers.platforms = self.inData.passengers.apply(lambda x: [0, 1], axis=1)

        from MaaSSim.simulators import simulate as platform_simulator_2
        self.sim2 = platform_simulator_2(params=params, inData=self.inData, f_platform_choice=f_platform_choice)
        ret = self.sim2.res[0].veh_exp
        ret['platform'] = self.inData.vehicles.platform
        second = ret[ret.platform == 1].ARRIVES_AT_DROPOFF.sum()
        self.assertGreater(second, first)

        del self.sim
        del self.sim2
        del params
        del self.inData




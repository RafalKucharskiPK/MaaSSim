import unittest
import os

class TestMultiplatform(unittest.TestCase):
    def test_platform_competition(self):
        # make sure when you compete with the prcie, lowering the fare and increasing the fee
        from MaaSSim.data_structures import structures as inData
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
        inData = load_G(inData, params,
                        stats=True)  # download graph for the 'params.city' and calc the skim matrices
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0] = [1, 'Platform1', 30]
        inData.platforms.loc[1] = [fare, 'Platform2', 30]
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.vehicles.platform = [0] * 20 + [1] * fleet

        inData.passengers.platforms = inData.passengers.apply(lambda x: [0, 1], axis=1)
        sim = platform_simulator_1(params=params, inData=inData, f_platform_choice=f_platform_choice)
        ret = sim.res[0].veh_exp.copy()
        ret['platform'] = inData.vehicles.platform
        first = ret[ret.platform == 1].ARRIVES_AT_DROPOFF.sum()

        fare = 0.5
        fleet = 50
        params.nV = 20 + fleet
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0] = [1, 'Platform1', 30]
        inData.platforms.loc[1] = [fare, 'Platform2', 30]
        inData = generate_demand(inData, params, avg_speed=True)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.vehicles.platform = [0] * 20 + [1] * fleet

        inData.passengers.platforms = inData.passengers.apply(lambda x: [0, 1], axis=1)

        from MaaSSim.simulators import simulate as platform_simulator_2
        sim = platform_simulator_2(params=params, inData=inData, f_platform_choice=f_platform_choice)
        ret = sim.res[0].veh_exp
        ret['platform'] = inData.vehicles.platform
        second = ret[ret.platform == 1].ARRIVES_AT_DROPOFF.sum()
        self.assertGreater(second, first)

        del sim
        del params
        del inData




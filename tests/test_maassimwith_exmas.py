import unittest
import os
import ExMAS

class TestEXMAS(unittest.TestCase):
    def test_maassim_with_exmas(self):
        from MaaSSim.data_structures import structures as inData
        from MaaSSim.utils import get_config, load_G, prep_supply_and_demand, generate_demand, generate_vehicles, \
            initialize_df  # simulator

        from MaaSSim.simulators import simulate
        from MaaSSim.driver import driverEvent
        from MaaSSim.utils import get_config


        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_platform_choices.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        params.times.pickup_patience = 3600  # 1 hour of simulation
        params.simTime = 4  # 1 hour of simulation
        params.nP = 500  # reuqests (and passengers)
        params.nV = 100  # vehicles

        params.shareability.avg_speed = params.speeds.ride
        params.shareability.shared_discount = 0.3
        params.shareability.delay_value = 1
        params.shareability.WtS = 1.3
        params.shareability.price = 1.5  # eur/km
        params.shareability.VoT = 0.0035  # eur/s
        params.shareability.matching_obj = 'u_pax'  # minimize VHT for vehicles
        params.shareability.pax_delay = 0
        params.shareability.horizon = 600
        params.shareability.max_degree = 4
        params.shareability.nP = params.nP
        params.shareability.share = 1
        params.shareability.without_matching = True

        inData = load_G(inData, params)  # load network graph

        inData = generate_demand(inData, params, avg_speed=False)
        inData.vehicles = generate_vehicles(inData, params.nV)
        inData.vehicles.platform = inData.vehicles.apply(lambda x: 0, axis=1)
        inData.passengers.platforms = inData.passengers.apply(lambda x: [0], axis=1)
        inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0],
                                                            axis=1)
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0] = [1, 'Uber', 30]
        params.shareability.share = 1
        params.shareability.without_matching = True

        inData = ExMAS.main(inData, params.shareability, plot=False)  # create shareability graph (ExMAS)

        sim = simulate(params=params, inData=inData)  # simulate

        self.assertGreater(sim.inData.sblts.schedule.kind.max(),10)  # are there shared rides in the solution

        self.assertGreater(sim.runs[0].rides.paxes.apply(lambda x: len(x)).max(), 1)  # did someone travel together


#
import unittest
import os


class TestBatch(unittest.TestCase):
    """
    Test of MaaSSim capabilities, functionalities, extra features
    """

    def test_batch_platform(self):
        """
        test if platform batches the results properly at matching
        """
        from MaaSSim.utils import initialize_df, get_config, load_G, prep_supply_and_demand
        from MaaSSim.data_structures import structures as inData
        from MaaSSim.simulators import simulate as this_simulator


        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_results_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        inData = load_G(inData, params)  # load network graph
        inData = prep_supply_and_demand(inData, params)  # generate supply and demand
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0] = [1, 'Uber', 600]  # batch requests every 600 seconds

        sim = this_simulator(params=params, inData=inData, event_based=False)
        Qs = sim.runs[0].queues
        Qs.groupby('platform')[['vehQ', 'reqQ']].plot(drawstyle='steps-post')

        r = sim.runs[0].trips
        times = r[r.event == 'RECEIVES_OFFER'].t.sort_values(ascending=True).diff().dropna().unique()

        self.assertIn(600, times)  # are requests batched ony at batch_time

        del sim
        del params
        del inData




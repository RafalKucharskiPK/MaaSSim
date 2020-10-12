import unittest
import os

class TestUtils(unittest.TestCase):
    """
    test input, output, utils, etc.
    """

    def test_configIO(self):
        from MaaSSim.utils import make_config_paths
        from MaaSSim.utils import get_config, save_config
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_utils_test.json')


        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file
        params = make_config_paths(params, main='test_path', rel=True)
        self.assertEqual(params.paths.G[0:9], 'test_path')
        params.testIO = '12'
        save_config(params, os.path.join(os.path.dirname(__file__), 'configIO_test.json'))
        params = get_config(os.path.join(os.path.dirname(__file__), 'configIO_test.json'),
                            root_path=os.path.dirname(__file__))  # load from .json file
        self.assertEqual(params.testIO, params.testIO)

    def test_networkIO(self):
        from numpy import inf
        from MaaSSim.utils import load_G, download_G, save_G, get_config
        from MaaSSim.data_structures import structures as inData

        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_utils_test.json')
        params = get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))  # load from .json file

        params.city = 'Wieliczka, Poland'

        params.paths.G = os.path.join(os.path.dirname(__file__),
                                           params.city.split(",")[0] + ".graphml")  # graphml of a current .city
        params.paths.skim = os.path.join(os.path.dirname(__file__), params.city.split(",")[
            0] + ".csv")  # csv with a skim between the nodes of the .city

        inData = download_G(inData, params)  # download the graph and compute the skim
        save_G(inData, params)  # save it to params.paths.G
        inData = load_G(inData, params,
                             stats=True)  # download graph for the 'params.city' and calc the skim matrices

        self.assertGreater(inData.nodes.shape[0], 10)  # do we have nodes
        self.assertGreater(inData.skim.shape[0], 10)  # do we have skim
        self.assertLess(inData.skim.mean().mean(), inf)  # and values inside
        self.assertGreater(inData.skim.mean().mean(), 0)  # positive distances



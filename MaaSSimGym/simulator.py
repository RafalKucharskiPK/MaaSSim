import threading

from MaaSSimGym.gym_api_controller import GymApiControllerState, GymApiController
from MaaSSim.maassim import Simulator
from MaaSSim.shared import prep_shared_rides
from MaaSSim.utils import get_config, read_requests_csv, read_vehicle_positions, load_G, generate_demand, generate_vehicles, initialize_df, empty_series


class GymSimulator(Simulator):
    def __init__(
            self,
            user_controller_action_needed: threading.Event,
            user_controller_action_ready: threading.Event,
            simulation_finished: threading.Event,
            state: GymApiControllerState,
            _inData,
            **kwargs,
    ) -> None:
        self.gym_api_controller = GymApiController(
            user_controller_action_needed=user_controller_action_needed,
            user_controller_action_ready=user_controller_action_ready,
            state=state,
            inData=_inData,
        )
        self.simulation_finished = simulation_finished
        super().__init__(
            _inData,
            f_user_controlled_driver_decline=self.gym_api_controller.incoming_offer_decision,
            f_user_controlled_driver_repos=self.gym_api_controller.reposition_decision,
            **kwargs,
        )

    def make_and_run(self, run_id=None, **kwargs):
        # wrapper for the simulation routine
        super().make_and_run(run_id, **kwargs)
        self.simulation_finished.set()


def prepare_gym_simulator(
        user_controller_action_needed: threading.Event,
        user_controller_action_ready: threading.Event,
        state: GymApiControllerState,
        config="data/gym_config.json",
        inData=None,
        params=None,
        **kwargs,
) -> GymSimulator:
    if inData is None:  # otherwise we use what is passed
        from MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
        params = get_config(config, root_path = kwargs.get('root_path'))  # load from .json file
    if kwargs.get('make_main_path',False):
        from MaaSSim.utils import make_config_paths
        params = make_config_paths(params, main = kwargs.get('make_main_path',False), rel = True)

    if params.paths.get('requests', False):
        inData = read_requests_csv(inData, path=params.paths.requests)

    if params.paths.get('vehicles', False):
        inData = read_vehicle_positions(inData, path=params.paths.vehicles)

    if len(inData.G) == 0:  # only if no graph in input
        inData = load_G(inData, params, stats=True)  # download graph for the 'params.city' and calc the skim matrices
    if len(inData.passengers) == 0:  # only if no passengers in input
        inData = generate_demand(inData, params, avg_speed=True)
    if len(inData.vehicles) == 0:  # only if no vehicles in input
        inData.vehicles = generate_vehicles(inData, params.nV, params.user_controlled_vehicles_count)
    if len(inData.platforms) == 0:  # only if no platforms in input
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0] = empty_series(inData.platforms)
        inData.platforms.fare = [1]

    inData = prep_shared_rides(inData, params.shareability)  # prepare schedules
    sim = GymSimulator(
        _inData=inData,
        params=params,
        user_controller_action_needed=user_controller_action_needed,
        user_controller_action_ready=user_controller_action_ready,
        state=state,
        **kwargs,
    )  # initialize
    return sim

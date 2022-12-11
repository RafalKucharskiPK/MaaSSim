import logging
import threading
from typing import Optional

import time
from gym import spaces
from gym.core import Env
from numpy import float64
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from MaaSSim.controllers.gym_api_controller import GymApiControllerState, ACCEPT, DECLINE
from MaaSSim.decisions import OfferStatus
from MaaSSim.simulators import prepare_gym_simulator


class MaaSSimEnv(Env):
    def render(self, mode="human"):
        pass

    def __init__(self, render_mode=None) -> None:
        self.user_controller_action_needed = threading.Event()
        self.user_controller_action_ready = threading.Event()
        self.simulation_finished = threading.Event()
        self.state = GymApiControllerState()
        self.sim = prepare_gym_simulator(
            config="data/gym_config_delft.json",
            user_controller_action_needed=self.user_controller_action_needed,
            user_controller_action_ready=self.user_controller_action_ready,
            simulation_finished=self.simulation_finished,
            state=self.state,
        )
        self.sim_daemon = None
        self.done = False
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "offer_fare": spaces.Box(low=0., high=300., shape=(1,), dtype=float64),
            "offer_travel_time": spaces.Box(low=0., high=300., shape=(1,), dtype=float64),
            "offer_wait_time": spaces.Box(low=0., high=300., shape=(1,), dtype=float64),
            "vehicle_current_cords": spaces.Box(low=0., high=180., shape=(1, 2), dtype=float64),
            "offer_origin_cords": spaces.Box(low=0., high=180., shape=(1, 2), dtype=float64),
            "offer_target_cords": spaces.Box(low=0., high=180., shape=(1, 2), dtype=float64),
            "is_reposition": spaces.Discrete(2),
        })
        self.action_space = spaces.Discrete(2)
        self.action_to_decision = {
            0: DECLINE,
            1: ACCEPT,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if self.sim_daemon is not None and self.sim_daemon.is_alive():
            self._close_simulation()
        self.sim_daemon = threading.Thread(
            target=self.sim.make_and_run,
        )
        self.done = False
        self.state.reward = 0.
        self.sim_daemon.start()
        self._wait_for_call_to_action()
        return self.state.observation

    def step(self, action: int):
        offer = self.state.current_offer
        self.state.action = self.action_to_decision[action]
        # trigger action ready event
        self.user_controller_action_ready.set()
        self._wait_for_call_to_action()
        if offer['status'] == OfferStatus.ACCEPTED:
            self.state.reward = offer['fare']
        current_observation = self.state.observation
        assert current_observation is not None
        print(current_observation)
        self._cleanup_events()
        return current_observation, self.state.reward, self.done, {}

    def close(self):
        logging.warning("Closing the simulation")
        self._close_simulation()

    def _wait_for_call_to_action(self) -> None:
        # wait for action needed event or simulator finish
        while not (self.user_controller_action_needed.is_set() or self.simulation_finished.is_set()):
            time.sleep(.01)

    def _cleanup_events(self) -> None:
        if self.user_controller_action_needed.is_set():
            # clear action needed event
            self.user_controller_action_needed.clear()
        if self.simulation_finished.is_set():
            self.done = True
            self.simulation_finished.clear()

    def _close_simulation(self) -> None:
        while not self.simulation_finished.is_set():
            time.sleep(.01)
            self.user_controller_action_ready.set()
        self.sim_daemon.join()
        self._cleanup_events()


def test_run() -> None:
    env = MaaSSimEnv()
    check_env(env)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(False)
    env.sim_daemon.join()


def test_train() -> None:
    env = MaaSSimEnv()
    model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log="./dqn_maassim_tensorboard/")
    model.learn(total_timesteps=10)
    env.close()


if __name__ == '__main__':
    test_train()

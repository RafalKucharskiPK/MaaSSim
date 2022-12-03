import threading
from typing import Optional

import time
from gym import spaces
from gym.core import Env
from numpy import float64
from stable_baselines3.common.env_checker import check_env

from MaaSSim.controllers.gym_api_controller import GymApiControllerState, ACCEPT, DECLINE
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
        self.sim_daemon = threading.Thread(
            target=self.sim.make_and_run,
        )
        self.sim_daemon.start()
        self._wait_for_call_to_action()
        return self.state.observation

    def step(self, action: int):
        self.state.action = self.action_to_decision[action]
        if self.state.action == ACCEPT:
            self.state.reward += float(self.state.observation['offer_fare'])
        # trigger action ready event
        self.user_controller_action_ready.set()
        self._wait_for_call_to_action()
        current_observation = self.state.observation
        assert current_observation is not None
        print(current_observation)
        self._cleanup_events()
        return current_observation, self.state.reward, self.done, {}

    def _wait_for_call_to_action(self) -> None:
        # wait for action needed event or simulator finish
        while not (self.user_controller_action_needed.is_set() or self.simulation_finished.is_set()):
            time.sleep(.01)

    def _cleanup_events(self) -> None:
        if self.user_controller_action_needed.is_set():
            # clear action needed event
            self.user_controller_action_needed.clear()
        elif self.simulation_finished.is_set():
            self.done = True
            self.simulation_finished.clear()


if __name__ == '__main__':
    env = MaaSSimEnv()
    check_env(env)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(False)
    env.sim_daemon.join()

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import time
from gym import spaces
from gym.core import Env
from matplotlib import pyplot as plt
from numpy import float64
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from MaaSSimGym.gym_api_controller import GymApiControllerState, ACCEPT, DECLINE, Observation, Action
from MaaSSim.decisions import OfferStatus
from MaaSSimGym.simulator import prepare_gym_simulator


class MaaSSimEnv(Env):
    action_to_decision = {
        0: DECLINE,
        1: ACCEPT,
    }

    def render(self, mode="human"):
        pass

    def __init__(self, render_mode: Optional[str] = None, config_path: str = "data/gym_config_delft_balanced_market.json") -> None:
        self.user_controller_action_needed = threading.Event()
        self.user_controller_action_ready = threading.Event()
        self.simulation_finished = threading.Event()
        self.state = GymApiControllerState()
        self.sim = prepare_gym_simulator(
            config=config_path,
            user_controller_action_needed=self.user_controller_action_needed,
            user_controller_action_ready=self.user_controller_action_ready,
            simulation_finished=self.simulation_finished,
            state=self.state,
        )
        self.sim_daemon = None
        self.done = False
        self.render_mode = render_mode
        self.behaviour_log: List[Tuple[Observation, Action]] = []  # used to analyse agent behaviour

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
        self.behaviour_log.append((self.state.observation, self.action_to_decision[action]))
        self.user_controller_action_ready.set()
        self._wait_for_call_to_action()
        if offer['status'] == OfferStatus.ACCEPTED:
            self.state.reward = offer['fare']
        else:
            self.state.reward = 0.
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


class FigureRecorderCallback(BaseCallback):
    def __init__(self, maassim_env: MaaSSimEnv, verbose=0):
        super(FigureRecorderCallback, self).__init__(verbose)
        self.maassim_env = maassim_env

    def _on_step(self) -> bool:
        if self.num_timesteps % 10 == 0:
            data = pd.DataFrame.from_records([
                self._map_behaviour_to_record(observation, action) for observation, action in
                self.maassim_env.behaviour_log
            ])
            if data.empty:
                logging.error("Something wrong: behaviour log is empty!")
                return True

            offers = data[data['is_reposition'] == 0]
            repositions = data[data['is_reposition'] == 1]
            self.maassim_env.behaviour_log.clear()
            self._log_plot_of_cords("offer_target_cords", offers, "offers")
            self._log_plot_of_cords("offer_origin_cords", offers, "offers")
            self._log_plot_of_cords("vehicle_current_cords", offers, "offers")

            self._log_plot_of_cords("offer_target_cords", repositions, "repositions")
            self._log_plot_of_cords("offer_origin_cords", repositions, "repositions")
            self._log_plot_of_cords("vehicle_current_cords", repositions, "repositions")

            plt.close()
            self.logger.dump(self.num_timesteps)
        return True

    def _log_plot_of_cords(self, cords_name: str, data: pd.DataFrame, directory: str) -> None:
        g = sns.JointGrid(data=data, x=f"{cords_name}_x", y=f"{cords_name}_y", hue='action', space=0)
        g.plot_joint(sns.scatterplot, size=.05)
        g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=50)
        self.logger.record(f"{directory}/{cords_name}", Figure(g.figure, close=True),
                           exclude=("stdout", "log", "json", "csv"))

    @staticmethod
    def _map_behaviour_to_record(observation: Observation, action: Action) -> Dict[str, Any]:
        offer_target_cords_x, offer_target_cords_y = observation['offer_target_cords'][0]
        offer_origin_cords_x, offer_origin_cords_y = observation['offer_origin_cords'][0]
        vehicle_current_cords_x, vehicle_current_cords_y = observation['vehicle_current_cords'][0]
        return {
            "offer_fare": float(observation['offer_fare']),
            "offer_travel_time": float(observation['offer_travel_time']),
            "offer_wait_time": float(observation['offer_wait_time']),
            "offer_target_cords_x": offer_target_cords_x,
            "offer_target_cords_y": offer_target_cords_y,
            "offer_origin_cords_x": offer_origin_cords_x,
            "offer_origin_cords_y": offer_origin_cords_y,
            "vehicle_current_cords_x": vehicle_current_cords_x,
            "vehicle_current_cords_y": vehicle_current_cords_y,
            "is_reposition": observation['is_reposition'],
            "action": "ACCEPT" if action == ACCEPT else "DECLINE"
        }


def test_run() -> None:
    env = MaaSSimEnv()
    check_env(env)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(False)
    env.sim_daemon.join()


def test_train(env_config_path: str, model_prefix: str) -> BaseAlgorithm:
    env = DummyVecEnv([lambda: MaaSSimEnv(config_path=env_config_path)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, norm_obs_keys=[
        "offer_travel_time",
        "offer_wait_time",
    ])
    model_name = datetime.now(tz=timezone.utc).strftime(f"{model_prefix}_%Y%m%dT%H%M%S")
    model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log="MaaSSimGym/dqn_maassim_tensorboard/", learning_starts=2500)
    model = model.learn(total_timesteps=10000)
    model.save(model_name)
    env.close()
    return model


def test_run_model(model: BaseAlgorithm, name: str) -> None:
    day = 0

    def _create_row(observation: Observation, action: np.ndarray, decision_id: int) -> Dict[str, float]:
        return {
            'day': day,
            'decision_id': decision_id,
            'is_reposition': bool(observation['is_reposition']),
            'offer_fare': float(observation['offer_fare']),
            'offer_travel_time': float(observation['offer_travel_time']),
            'offer_wait_time': float(observation['offer_wait_time']),
            'vehicle_cords_x': float(observation['vehicle_current_cords'][0][0][0]),
            'vehicle_cords_y': float(observation['vehicle_current_cords'][0][0][1]),
            'offer_origin_cords_x': float(observation['offer_origin_cords'][0][0][0]),
            'offer_origin_cords_y': float(observation['offer_origin_cords'][0][0][1]),
            'offer_target_cords_x': float(observation['offer_target_cords'][0][0][0]),
            'offer_target_cords_y': float(observation['offer_target_cords'][0][0][1]),
            'action': MaaSSimEnv.action_to_decision.get(int(action)),
        }

    print("Test run of the model initiated")
    env = model.get_env()
    obs = env.reset()
    observations_with_action = []
    for i in range(5000):
        action, _states = model.predict(obs)
        observations_with_action.append(_create_row(obs, action[0], i))
        print(action)
        obs, rewards, dones, info = env.step(action)
        if dones[0]:
            day += 1
    logs_parent = Path("MaaSSimGym/decision_logs")
    logs_parent.mkdir(exist_ok=True)
    with open(logs_parent.joinpath(f"{name}.csv"), "w+") as f:
        pd.DataFrame.from_records(observations_with_action).to_csv(f, index=False)
    env.close()


def main():
    configurations = [
        ("MaaSSimGym/configs/gym_config_delft_balanced_market.json", "dqn_balanced_market"),
        ("MaaSSimGym/configs/gym_config_delft_driver_market.json", "dqn_driver_market"),
        ("MaaSSimGym/configs/gym_config_delft_passenger_market.json", "dqn_passenger_market"),
    ]
    for config_path, model_prefix in configurations:
        model = test_train(config_path, model_prefix)
        test_run_model(model, model_prefix)


if __name__ == '__main__':
    main()

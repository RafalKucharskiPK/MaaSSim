from __future__ import annotations
import random
import threading
from dataclasses import dataclass
from typing import TypedDict, Optional

import numpy as np
from dotmap import DotMap
from numpy import ndarray

from MaaSSim.decisions import Offer
from MaaSSim.driver import VehicleAgent

ACCEPT = False
DECLINE = True


class Observation(TypedDict):
    offer_fare: ndarray
    offer_travel_time: ndarray
    offer_wait_time: ndarray


def create_observation_from_input(veh: VehicleAgent, offer: Offer) -> Observation:
    return Observation(
        offer_fare=np.array(offer['fare']).reshape(1),
        offer_travel_time=np.array(offer['travel_time']).reshape(1),
        offer_wait_time=np.array(offer['wait_time']).reshape(1),
    )


@dataclass
class GymApiControllerState:
    reward: float = 0.
    action: bool = ACCEPT
    observation: Optional[Observation] = None


class GymApiController:
    """
    Class that can be used to enable the user (be it human or RL agent) to interact with maassim simulator
    through passing some of its methods as a decision functions for driver agent
    """

    def __init__(
            self,
            user_controller_action_needed: threading.Event,
            user_controller_action_ready: threading.Event,
            state: GymApiControllerState,
    ) -> None:
        self.user_controller_action_needed = user_controller_action_needed
        self.user_controller_action_ready = user_controller_action_ready
        self.state = state

    def incoming_offer_decision(self, veh: VehicleAgent, offer: Offer) -> bool:
        self.state.observation = create_observation_from_input(veh, offer)
        # signals to GymAPI environment that action needs to be determined
        self.user_controller_action_needed.set()
        # waits for the next action to be determined
        self.user_controller_action_ready.wait()
        # clears the action event
        self.user_controller_action_ready.clear()
        return self.state.action

    @staticmethod
    def reposition_decision(veh: VehicleAgent) -> DotMap:
        # TODO:
        # empty offer is needed here
        # create empty offer for every neighbour and check yield for every one of them
        # below implementation is temporary
        repos = DotMap()
        driver = veh
        sim = driver.sim
        neighbors = list(sim.inData.G.neighbors(driver.veh.pos))
        if len(neighbors) == 0:
            # escape from dead-end (teleport)
            repos.pos = sim.inData.nodes.sample(1).squeeze().name
            repos.time = 300
            repos.flag = True
        else:
            choice = random.randint(0, len(neighbors)+1)
            if choice == 0:
                repos.flag = False
            else:
                repos.pos = neighbors[choice]
                repos.time = driver.sim.skims.ride[repos.pos][driver.veh.pos]
                repos.flag = True
        return repos

    @staticmethod
    def drive_out_today_decision(veh: VehicleAgent) -> bool:
        return ACCEPT

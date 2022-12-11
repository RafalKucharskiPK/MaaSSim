from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TypedDict, Optional

import numpy as np
import pandas as pd
from dotmap import DotMap
from numpy import ndarray, float64

from MaaSSim.decisions import Offer, OfferStatus
from MaaSSim.driver import VehicleAgent

ACCEPT = False
DECLINE = True


class Observation(TypedDict):
    offer_fare: ndarray
    offer_travel_time: ndarray
    offer_wait_time: ndarray
    offer_target_cords: ndarray
    offer_origin_cords: ndarray
    vehicle_current_cords: ndarray
    is_reposition: int


def create_observation_from_input(veh: VehicleAgent, offer: Offer, inData: DotMap) -> Observation:
    return Observation(
        offer_fare=np.array(offer['fare']).reshape(1),
        offer_travel_time=np.array(offer['travel_time']).reshape(1),
        offer_wait_time=np.array(offer['wait_time']).reshape(1),
        vehicle_current_cords=inData.nodes[inData.nodes.index == veh.veh.pos][['x', 'y']].to_numpy(),
        offer_origin_cords=inData.nodes[inData.nodes.index == offer['request']['origin']][['x', 'y']].to_numpy(),
        offer_target_cords=inData.nodes[inData.nodes.index == offer['request']['destination']][['x', 'y']].to_numpy(),
        is_reposition=0,
    )


def create_observation_from_reposition(veh: VehicleAgent, neighbor_position_index: int, inData: DotMap) -> Observation:
    return Observation(
        offer_fare=np.zeros(shape=(1,), dtype=float64),
        offer_travel_time=pd.Timedelta(inData.skim[neighbor_position_index][veh.veh.pos], 's').floor('s').to_numpy(),
        offer_wait_time=np.zeros(shape=(1,), dtype=float64),
        offer_target_cords=inData.nodes[inData.nodes.index == neighbor_position_index][['x', 'y']].to_numpy(),
        offer_origin_cords=inData.nodes[inData.nodes.index == veh.veh.pos][['x', 'y']].to_numpy(),
        vehicle_current_cords=inData.nodes[inData.nodes.index == veh.veh.pos][['x', 'y']].to_numpy(),
        is_reposition=1,
    )


def create_empty_reposition_offer() -> Offer:
    return Offer(
        pax_id=-1,
        req_id=-1,
        simpaxes=[],
        veh_id=-1,
        status=OfferStatus.ACCEPTED,
        request=pd.Series(),
        wait_time=0,
        travel_time=0.,
        fare=0.,
    )


@dataclass
class GymApiControllerState:
    current_offer: Optional[Offer] = None
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
            inData: DotMap,
    ) -> None:
        self.user_controller_action_needed = user_controller_action_needed
        self.user_controller_action_ready = user_controller_action_ready
        self.state = state
        self.inData = inData

    def incoming_offer_decision(self, veh: VehicleAgent, offer: Offer) -> bool:
        self.state.current_offer = offer
        self.state.observation = create_observation_from_input(veh, offer, self.inData)
        # signals to GymAPI environment that action needs to be determined
        self.user_controller_action_needed.set()
        # waits for the next action to be determined
        self.user_controller_action_ready.wait()
        # clears the action event
        self.user_controller_action_ready.clear()
        return self.state.action

    def reposition_decision(self, veh: VehicleAgent) -> DotMap:
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
            choice = 0
            for i, neighbor in enumerate(neighbors):
                self.state.current_offer = create_empty_reposition_offer()
                self.state.observation = create_observation_from_reposition(veh, neighbor, sim.inData)
                self.user_controller_action_needed.set()
                self.user_controller_action_ready.wait()
                self.user_controller_action_ready.clear()
                if self.state.action == ACCEPT:
                    choice = i
            if choice == 0:
                repos.flag = False
            else:
                logging.info("Repositioning")
                repos.pos = neighbors[choice]
                repos.time = driver.sim.skims.ride[repos.pos][driver.veh.pos]
                repos.flag = True
        return repos

    @staticmethod
    def drive_out_today_decision(veh: VehicleAgent) -> bool:
        return ACCEPT

from dotmap import DotMap

from MaaSSim.decisions import Offer
from MaaSSim.driver import VehicleAgent

ACCEPT = False
DECLINE = True


class UserController:
    """
    Class that can be used to enable the user (be it human or RL agent) to interact with massim simulator
    through passing some of its methods as a decision functions for driver agent
    """

    @staticmethod
    def incoming_offer_decision(veh: VehicleAgent, offer: Offer) -> bool:
        print(f"Veh id: {veh.id}")
        print(f"Fare: {offer['fare']}")
        print(f"Travel time: {offer['travel_time']}")
        decision_raw = input("Accept offer [yN]: ")
        decision = ACCEPT if decision_raw.upper() == "Y" else DECLINE
        return decision

    @staticmethod
    def reposition_decision(veh: VehicleAgent) -> DotMap:
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
            print("Choose reposition target: ")
            print(f"[0] {driver.veh.pos} (current position)")
            for index, neighbor in enumerate(neighbors, start=1):
                print(f"[{index}] {neighbor}")

            choice_raw = input("Enter choice: ")
            choice = int(choice_raw)
            if choice == 0:
                repos.flag = False
            else:
                repos.pos = neighbors[choice - 1]
                repos.time = driver.sim.skims.ride[repos.pos][driver.veh.pos]
                repos.flag = True
        return repos

    @staticmethod
    def drive_out_today_decision(veh: VehicleAgent) -> bool:
        print(veh.myrides)
        return ACCEPT

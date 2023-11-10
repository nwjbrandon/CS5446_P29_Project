"""
Handcrafted agents for evaluation purposes.
"""
from gym_warehouse.envs import warehouse_env


class OmniscientAgent:
    """
    An agent that knows about the internal state of the warehouse.
    Not implemented yet.
    """

    def __init__(self, env: warehouse_env.WarehouseEnv):
        self.env = env

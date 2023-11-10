"""
A gym environment representing a chaotic warehouse.
"""
import gym
import os
import json
import numpy as np
import random
import time
import sys
from gym_warehouse.envs.warehouse_objects import (
    Agent, Bin, StagingInArea, StagingOutArea, Obstacle
)
from gym_warehouse.envs.warehouse_renderer import WarehouseRenderer
from .curiosity import Curiosity2
#from .curiosity import Curiosity
import torch.optim as optim
import itertools

# Constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

MOVE_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1)
}


class WarehouseEnv(gym.Env):
    """
    An environment representing a chaotic warehouse.

    Parameters
    ----------
    envname : str
        The name of the warehouse environment.
        There needs to be a file called 'envname.json' in the 'envs_data'
        subdirectory.
    """
    def __init__(self, envname='warehouse', curiosity=False):
        # Load specified environment from .json
        file_dir, _ = os.path.split(__file__)
        path = os.path.join(file_dir, 'envs_data', f'{envname}.json')
        with open(path) as f:
            data = json.load(f)

        # Warehouse parameters
        self.num_rows = data['num_rows']
        self.num_cols = data['num_cols']
        self.num_items = data['num_items']
        self.num_agent_slots = data['num_agent_slots']
        self.num_bin_slots = data['num_bin_slots']
        self.num_bins = data['num_bins']
        self.rewards = data['rewards']
        self.time_limit = data['time_limit']
        self.time_step = 0
        self.time_left = self.time_limit

        # Agent
        self.agent = Agent(data['agent'])

        # Staging areas
        self.staging_in_area = StagingInArea(data['staging_in_area'])
        self.staging_out_area = StagingOutArea(data['staging_out_area'])

        # Bins
        self.bins = [Bin(bin_) for bin_ in data['bins']]

        #Obstacles
        # Store positions of obstacles in set for O(1) lookup
        self.obstacles_only = [Obstacle(obst) for obst in data['obstacles']]
        self.obstacles = {obst.position for obst in self.obstacles_only}
        for bin_ in self.bins:
            self.obstacles.add(bin_.position)
        self.obstacles.add(self.staging_in_area.position)
        self.obstacles.add(self.staging_out_area.position)

        # Map access spots to containers for O(1) lookup
        self.container_map = {}
        for container in self.bins + [self.staging_in_area,
                                      self.staging_out_area]:
            for spot in container.access_spots:
                self.container_map[spot] = container

        # Action space
        self.num_actions = 4 + 2 * self.num_agent_slots * self.num_bin_slots
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_staging_areas = gym.spaces.MultiDiscrete([self.num_items + 1] * self.num_bin_slots)

        # Observation space
        nvec = [self.num_rows, self.num_cols]
        nvec += [self.num_items + 1] * self.num_agent_slots
        nvec += [self.num_items + 1] * (self.num_bin_slots * len(self.bins))
        nvec += [self.num_items + 1] * (self.num_bin_slots * 2)
        nvec += [self.time_limit + 1]
        self.num_observations = len(nvec)
        self.observation_space = gym.spaces.MultiDiscrete(nvec)

        # renderer
        self.renderer = None
        self.reward = 0
        self.rewardCuriosity = 0
        self.action = ''

        # Video
        self.video = False
        self.video_filename = None

        self.done = False
        #random warehouse
        for bin_ in self.bins:
            #bin_.reset()
            bin_.status = self.observation_staging_areas.sample()

        #list of valid spots for the agent
        self.agent_valid_spots = set(itertools.product(range(0, self.num_rows), range(0, self.num_cols)))
        self.agent_valid_spots -= self.obstacles
        self.agent_valid_spots = list(self.agent_valid_spots)

        self.agent.position = random.choice(self.agent_valid_spots)

        self._update_transactions()

        self.state = self._create_observation()

        self.curiosity = curiosity
        if self.curiosity:
            #Version 1 Curiosity (with sigmoid loss)
            #self.curiosity_NN = Curiosity(input_size=len(self.state),output_size=len(self.state)-1,discrete_obs=nvec[:-1])

            #Version 2 Curiosity (Multiclass classification, many softmax)
            curiosity_vec = np.append(nvec[:-1], self.num_actions)
            self.curiosity_NN = Curiosity2(curiosity_vec)

            #self.optimizer = optim.RMSprop(self.curiosity_NN.curiosity_net.parameters())
            self.optimizer = optim.Adam(self.curiosity_NN.curiosity_net.parameters())

    def step(self, action, testing = False):
        """
        Performs the action chosen by the agent and alters the state accordingly.

        Parameters
        ----------
        action : int
            The action performed by the agent.
            - 0 = UP
            - 1 = DOWN
            - 2 = LEFT
            - 3 = RIGHT
            - [4, 5 + num_bin_slots + num_agent_slots) = pick actions
            - [4 + num_bin_slots + num_agent_slots, num_actions) = put actions

        Returns
        -------
        obs : np.array, shape (2 + #agent_slots + (#bins + 2) * #bin_slots,)
            An array of ints representing an observation of the warehouse.
            Contains the position of the agent as well as the status of the
            agent, the bins, and the staging areas.
        reward : int
            A numerical reward for the agent.
        done : bool
            True if the episode has come to an end.
        info : dict
            A dictionary containing information about training. At the moment,
            only an empty dict is returned.
        """
        # Throw exception if action is invalid
        if not self.action_space.contains(action):
            message = f'Action must be int in [0, {self.action_space.n})'
            raise ValueError(message)

        reward = 0

        # Perform action
        if action < 4:
            reward = self._move(action)
        else:
            agent_slot, bin_slot, pick = self._decode_pick_or_put_action(action)
            if pick:
                reward = self._pick(agent_slot, bin_slot)
            else:
                reward = self._put(agent_slot, bin_slot)

        # Update transactions randomly
        #self._update_transactions()

        # Advance time
        self.time_step += 1
        self.time_left -=1

        # Create return tuple
        obs = self._create_observation()
        self.done = False
        info = {}
        self.reward = reward

        if self.time_step >= self.time_limit or self.reward >0:
        #if self.time_step >= self.time_limit:
            #print(self.time_step, self.time_left)
            self.done = True

        # Save state
        if self.video:
            self.renderer.save_state()

        self.rewardCuriosity = 0
        if (testing == False) and (self.curiosity == True):
            self.rewardCuriosity = self.curiosity_NN.train(current_state=self.state[:-1],next_state=obs[:-1], action=action, optimizer=self.optimizer)

        self.state = obs

        #if self.rewardCuriosity == 0:
            #print("Curiosity_reward 0!")

        return obs, self.reward + self.rewardCuriosity, self.done, info

    def render(self, mode='human', sec_per_frame=0.5):
        """
        Renders the environment in the desired mode.

        Parameters
        ----------
        mode : str
            Currently, only 'human' is supported, which will render using
            pygame.
        """
        if mode == 'human':
            if self.renderer is None:
                self.renderer = WarehouseRenderer(self)
            self.renderer.render()
            time.sleep(sec_per_frame)
        else:
            super().render(mode=mode)

    def reset(self, testing = False):
        """
        Resets the environment to its initial state and returns a fresh
        observation

        Returns
        -------
        obs : np.array, shape (2 + #agent_slots + (#bins + 2) * #bin_slots,)
            An array of ints representing an oberservation of the warehouse.
            Contains the position of the agent as well as the status of the
            agent, the bins, and the staging areas.
        """
        if not testing:

            self.agent.reset()
            self.agent.position = random.choice(self.agent_valid_spots)

            self.staging_in_area.reset()
            self.staging_out_area.reset()
            for bin_ in self.bins:
                #bin_.reset()
                bin_.status = self.observation_staging_areas.sample()

        self._update_transactions()

        self.time_step = 0
        self.time_left = self.time_limit
        self.done = False
        self.state = self._create_observation()
        return self.state

    def _create_observation(self):
        """
        Creates an observation from the current state.

        Returns
        -------
        obs : np.array, shape (2 + #agent_slots + (#bins + 2) * #bin_slots,)
            An array of ints representing an observation of the warehouse.
            Contains the position of the agent as well as the status of the
            agent, the bins, and the staging areas.
        """
        observation = np.zeros(self.num_observations, dtype=np.int64)
        # Agent position
        start = 0
        end = 2
        observation[start:end] = self.agent.position
        # Agent status
        start = end
        end += self.num_agent_slots
        observation[start:end] = self.agent.status
        # Staging-in area
        start = end
        end += self.num_bin_slots
        observation[start:end] = self.staging_in_area.status
        # Staging-out area
        start = end
        end += self.num_bin_slots
        observation[start:end] = self.staging_out_area.status
        # Bins
        for bin_ in self.bins:
            start = end
            end += self.num_bin_slots
            observation[start:end] = bin_.status

        observation[-1] = self.time_left
        return observation

    def _move(self, action):
        """
        Tries to move the agent in the direction indicated by action.

        Parameters
        ----------
        action : int
            A move action. Must be in [0, 3].

        Returns
        -------
        reward : int
            The reward the agent gets for this action.
        """
        row, col = self.agent.position
        delta_row, delta_col = MOVE_DELTAS[action]
        row, col = row + delta_row, col + delta_col

        if self._within_bounds(row, col) and (row, col) not in self.obstacles:
            self.agent.position = (row, col)

        # Rendering
        self.action = f'Move -> ({row}, {col})'
        return self.rewards['default']

    def _within_bounds(self, row, col):
        """
        Checks whether the given position is within the bounds of the
        warehouse.

        Parameters
        ----------
        row : int
            The row to be checked.
        col : int
            The column to be checked.

        Returns
        -------
        bool :
            True if the position is within bounds.
        """
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols

    def _decode_pick_or_put_action(self, action):
        """
        Decode a pick or put action, i.e. extract the slot of the agent, the
        slot of the bin, and whether it is a pick or a put action.

        Parameters
        ----------
        action : int
            A pick or put action. Must be in [5, num_actions).

        Returns
        -------
        agent_slot : int
            The slot of the agent encoded in the action.
        bin_slot : int
            The slot of the bin encoded in the action.
        pick : bool
            True if the action is a pick action, and False if it's a put action.
        """
        action -= 4
        pick = True
        if action >= self.num_agent_slots * self.num_bin_slots:
            action -= self.num_agent_slots * self.num_bin_slots
            pick = False
        agent_slot = action // self.num_bin_slots
        bin_slot = action % self.num_bin_slots

        return agent_slot, bin_slot, pick

    def _pick(self, agent_slot, bin_slot):
        """
        Tries to perform a pick action from the bin slot into the agent slot.

        Parameters
        ----------
        agent_slot : int
            The slot of the agent to pick into.
        bin_slot : int
            The slot of the bin to pick from.

        Returns
        -------
        reward : int
            The reward the agent gets for trying to perform this action.
        """
        # Check whether agent is in an access spot
        if self.agent.position not in self.container_map:
            return self.rewards['default']

        # Get container associated with the access spot
        container = self.container_map[self.agent.position]

        # Can't pick from staging-out area
        if container is self.staging_out_area:
            return self.rewards['default']

        # Try pick from container
        if not container.free(bin_slot) and self.agent.free(agent_slot):
            self.agent.put(agent_slot, container.pick(bin_slot))
            # Check we made progress on inbound transaction
            if container is self.staging_in_area:
                return self.rewards['success']

        # Render
        self.action = f'Pick {agent_slot} <- {bin_slot}'

        return self.rewards['default']

    def _put(self, agent_slot, bin_slot):
        """
        Tries to perform a put action from the agent slot into the bin slot.

        Parameters
        ----------
        agent_slot : int
            The slot of the agent containing the item to put.
        bin_slot : int
            The slot of the bin to put the item into.

        Returns
        -------
        reward : int
            The reward the agent gets for trying to perform this action.
        """
        # Check whether agent is in an access spot
        if self.agent.position not in self.container_map:
            return self.rewards['default']

        # Get container associated with the access spot
        container = self.container_map[self.agent.position]

        # Can't put into staging-in area
        if container is self.staging_in_area:
            return self.rewards['default']

        # Can only put required items into staging-out area
        if container is self.staging_out_area:
            if container.requires(bin_slot, self.agent.status[agent_slot]):
                container.put(bin_slot, self.agent.pick(agent_slot))
                #return self.rewards['success']
                #Check if transaction completed to give reward and generate a new one
                if (np.sum(self.agent.status)==0 and np.sum(self.staging_in_area.status)==0 and
                        np.sum(self.staging_out_area.status)==0):

                    #self._update_transactions()
                    return self.rewards['completion'] #* self.time_left / self.time_limit

            return self.rewards['default']


        # Try to put into regular bin
        if container.free(bin_slot) and not self.agent.free(agent_slot):
            container.put(bin_slot, self.agent.pick(agent_slot))
            #check if staging in, staging out and agent are empty to generate new transaction
            if (np.sum(self.agent.status) == 0 and np.sum(self.staging_in_area.status) == 0 and
                    np.sum(self.staging_out_area.status) == 0):
                #self._update_transactions()
                return self.rewards['completion'] #* self.time_left / self.time_limit

        # Render
        self.action = f'Put {agent_slot} -> {bin_slot}'

        return self.rewards['default']

    def _update_transactions(self, p=0.5):
        """
        Randomly adds new items to staging in area or staging out area.
        When this method is called, with probability p/2 a new inbound item is
        added, with probability p/2 a new outbound item is added, and with
        probability 1 - p nothing is added. If a new item is to be added but
        the corresponding area is full, nothing happens.

        Parameters
        ----------
        p : float
            The probability of a new transaction-item being added.
        """
        flip = random.random()

        bin_items = np.array([])
        for bin_ in self.bins:
            bin_items = np.append(bin_items, bin_.status)
        #bin_items = np.append(bin_items,self.agent.status)
        bin_items = bin_items.flatten()

        y = lambda x: True if x != 0 else False
        bin_items_bool = [y(i) for i in bin_items]

        warehouse_full = all(bin_items_bool)

        emptyslots = self.num_bin_slots * self.num_bins - np.sum(bin_items_bool)
        #if emptyslots ==0:
        #    warehouse_full = True

        if (flip < p or np.sum(bin_items) == 0) and not (warehouse_full):
            while True:
                self.staging_in_area.status = self.observation_staging_areas.sample()

                # verifies that the number of slots is enough for the incoming in transaction
                sampleslots = np.sum([y(i) for i in self.staging_in_area.status])
                if np.sum(self.staging_in_area.status) != 0 and emptyslots >= sampleslots:
                    break

        else:
            missing_items = True;
            bin_counts = np.zeros(self.num_items + 1)
            for i in bin_items:
                bin_counts[int(i)] += 1
            while (np.sum(self.staging_out_area.status) == 0 or missing_items):
                self.staging_out_area.status = self.observation_staging_areas.sample()
                # checking that an outbound transaction always has the "required" items in bins.
                staging_counts = np.zeros(self.num_items + 1)
                for i in self.staging_out_area.status:
                    staging_counts[i] += 1
                missing_items = not (all(bin_counts[1:] >= staging_counts[1:]))

    def init_video(self):
        """
        Initialize renderer if not done yet. It will save the state history
        throughout the complete simulation. Thus, this function needs to be
        called before the simulation starts.
        """
        self.video = True

        # Initialize renderer if not done yet
        if self.renderer is None:
            self.renderer = WarehouseRenderer(self)

    def save_video(self, video_filename):
        """
        Save rendered environment as video in mp4 format.

        Parameters
        ----------
        video_filename : str
            Plain filename of the video; filename extension (.mp4) will be
            added automatically.
        """
        self.video_filename = ".".join([video_filename, "mp4"])

        if self.video:
            self.renderer.save_video()
        else:
            print("init_video has to be called before in order to save video; "
                  "aborting")

    def save_as_gif(self, filename_prefix = 'quick_test', save_dir = './', sec_per_frame=0.5):
        """
        Save rendered environment as an animation in gif format.

        Parameters
        ----------
        gif_filename : str
            Plain filename of the gif; filename extension (.gif) will be
            added automatically.
        """

        self.gif_filename = save_dir+'/'+filename_prefix+"_agent_interaction.gif"

        if self.renderer:
            self.renderer.save_as_gif(self.gif_filename, sec_per_frame)
        else:
            print("env.render(mode='human') has to be called before in order to save gif;\naborting...")

    def save_as_mp4(self, filename_prefix = 'quick_test', save_dir = './', sec_per_frame=0.5):
        """
        Save rendered environment as an mp4 video.

        Parameters
        ----------
        gif_filename : str
            Plain filename of the mp4; filename extension (.mp4) will be
            added automatically.
        """

        self.mp4_filename = save_dir+'/'+filename_prefix+"_agent_interaction.mp4"

        if self.renderer:
            self.renderer.save_as_mp4(self.mp4_filename, sec_per_frame)
        else:
            print("env.render(mode='human') has to be called before in order to save mp4;\naborting...")

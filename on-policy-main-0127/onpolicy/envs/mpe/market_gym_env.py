import gym
import pandas as pd
from gym import spaces
from gym.spaces import Discrete, Box
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from scipy.optimize import linprog
from pettingzoo.test import api_test
from pettingzoo.utils import random_demo


def wrapped_env(max_step, custom_agent_num):
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = MA_ELEC_Market_ENV(max_step, custom_agent_num)
    # env = wrappers.CaptureStdoutWrapper(env)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class MA_ELEC_Market_ENV(gym.Env):
    """
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self, max_step, custom_agent_num, gate_num = 15):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        # electronic need
        # episode length
        self.gate_num = gate_num
        self.custom_matrix = np.random.randint(2, size=(custom_agent_num, custom_agent_num))
        self.custom_agent_num = custom_agent_num
        csv_data_l = pd.read_excel('/home/ouazusakou/New_disk/on-policy-main/onpolicy/envs/mpe/20230201code+data/data_l_118.xlsx')

        self.L_list_day_array = csv_data_l
        # self.L_list = self.L_list_day_array[:,:,i]
        self.L_list = np.array(csv_data_l.values[:, 0:custom_agent_num])
        # print('l_list', self.L_list.shape)
        # print(L.shape)
        self.Total_L = np.sum(self.L_list) / custom_agent_num
        csv_data_p = pd.read_excel('/home/ouazusakou/New_disk/on-policy-main/onpolicy/envs/mpe/20230201code+data/data_p_118.xlsx')

        self.P_array = np.array(csv_data_p.values[:, 0:custom_agent_num])
        # print('p_list', self.P_array.shape)
        self.P_total = 0
        self.p_ug = 1
        self.sell_price_list = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 1.3, 1.3, 1.3, 1.3, 0.8, 0.8, 0.8,
                                0.8, 1.3, 1.3, 1.3, 0.8, 0.8, 0.4]
        # = [0.15,0.12,0.10,0.09,0.09,0.08,0.083,0.078,0.098,0.11,0.123,0.139,0.145,0.165,0.163,0.145,0.162,0.143,0.143,0.166,0.241,0.242,0.214,0.204]
        self.L_max = np.max(self.L_list)
        self.L_min = np.min(self.L_list)
        self.P_max = np.max(self.P_array)
        self.P_min = np.min(self.P_array)
        self.Pes_max = self.Total_L * 0.5 * 0.5
        # self.action_2_high_limit = self.L_max + self.Pes_max
        self.action_2_high_limit = 30 + 50
        # print(self.action_2_high_limit)

        # self.action_2_low_limit = -(self.P_max+self.Pes_max)
        self.action_2_low_limit = -(50 + 50)
        print('high', self.action_2_high_limit)
        print('low', self.action_2_low_limit)
        # print(self.action_2_low_limit)
        self.beta_p2p = 1
        self.buy_price = 0.35
        self.sell_price = self.sell_price_list[0]
        self.max_step = max_step
        self.agent_num = custom_agent_num + 1
        # action range for agent
        self.real_action_range = [60, 80, 100]
        self.possible_agents = ["player_" + str(r) for r in range(self.agent_num)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # elec parameter calculation

        # print(self.agent_name_mapping)
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        # Pes G Tr
        self.Pv_dic = {agent: None for agent in self.possible_agents}
        self.action_space =  [Box(low=-1, high=1, shape=(3,)) for agent in self.possible_agents]
        self.observation_space = [Box(low=-1, high=1, shape=(3,)) for agent in self.possible_agents]
        share_obs_dim = 3*custom_agent_num
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.custom_agent_num)]

        # self.share_observation_space = [spaces.Box(
        #     low=-np.inf, high=+np.inf, shape=(3*custom_agent_num,), dtype=np.float32) for _ in self.possible_agents]

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        # if len(self.agents) == 2:
        #     string = ("Current state: Agent1: {} , Agent2: {}".format(MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]))
        # else:
        #     string = "Game over"
        # print(string)
        pass

    def get_avaliable_action(self):
        mask = np.random.randint(2, size=(self.gate_num,))
        return mask


    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def reset(self):
        '''
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the stat/home/ouazusakou/Documents/rl_lib/mybaseline dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        '''

        self.P_total = 0
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.action_state = {agent: None for agent in self.agents}
        self.real_state = {agent: None for agent in self.agents}
        # self.allocation_state = {agent: None for agent in self.agents[:3]}
        # self.sale_price_state = None
        self.observations = {agent: None for agent in self.agents}
        self.P_u = {agent: None for agent in self.agents}
        self.a = 0.00059
        self.b = 0.302
        self.c = 0
        for i in self.agents:
            # print(self.agent_name_mapping[i])
            # self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
            # Pv L Es
            self.observations[i] = np.zeros((3,))
            self.real_state[i] = np.zeros((3,))

        self.num_moves = 0
        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''


        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        observation_list = []
        for agent in self.agents:
            observation_list.append(self.observations[agent])


        return observation_list

    def step(self, actions):
        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''
        dones = []
        observations = []
        rewards = []
        infos = []
        infos = [[] for agent in self.agents]
        dones = [self.num_moves >= (self.max_step - 1) for agent in self.agents]

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0


        # stores action of current agent
        # if self.agent_selection == 'player_0':
        #     self.action_state[self.agent_selection] = np.array(
        #         [action[0] * self.real_action_range[0] / 2 + self.real_action_range[0] / 2,
        #          action[1] * 250 + 1250])
        # elif self.agent_selection == 'player_1':
        #     self.action_state[self.agent_selection] = np.array(
        #         [action[0] * self.real_action_range[1] / 2 + self.real_action_range[1] / 2,
        #          action[1] * 250 + 1250])
        # elif self.agent_selection == 'player_2':
        #     self.action_state[self.agent_selection] = np.array(
        #         [action[0] * self.real_action_range[2] / 2 + self.real_action_range[2] / 2,
        #          action[1] * 250 + 1250])
        # else:
        #     if action[0] == -1:
        #         self.action_state[self.agent_selection] = np.array([0, 0])
        #     else:
        #         self.action_state[self.agent_selection] = np.array([1, 1250 + action[1] * 250])
        #
        # print(action)
        for i in range(self.agent_num):
            action = actions[i]
            self.agent_selection = self.agents[i]
            self.action_state[self.agent_selection] = action
            self.action_state[self.agent_selection][0] = action[0] * self.Pes_max
            self.action_state[self.agent_selection][1] = ((action[1] + 1) / 2) * np.max(self.P_array)
            self.action_state[self.agent_selection][2] = ((action[2] + 1) / 2) * (
                        self.action_2_high_limit - self.action_2_low_limit) + self.action_2_low_limit

            # print('total', self.Total_L * 0.5)
            if (self.real_state[self.agent_selection][2] - self.action_state[self.agent_selection][0]) < 0:
                self.action_state[self.agent_selection][0] = self.real_state[self.agent_selection][2]
            if self.real_state[self.agent_selection][2] - self.action_state[self.agent_selection][0] > (
                    self.Total_L * 0.5 * 0.2):
                self.action_state[self.agent_selection][0] = self.real_state[self.agent_selection][2] - (
                            self.Total_L * 0.5 * 0.2)

        # self.TR_state[self.agent_name_mapping[self.agent_selection]] = sum(action[])
        # collect reward if it is the last agent to act


        self.sell_price = self.sell_price_list[self.num_moves]

        self.num_moves += 1
        # The dones dictionary must be updated for all players.
        self.dones = {agent: self.num_moves >= (self.max_step - 1) for agent in self.agents}
        self.tr_dic = {agent: None for agent in self.agents}
        buy_sum = 0
        sell_sum = 0
        for agent in self.agents:
            if self.action_state[agent][2] > 0:
                buy_sum += self.action_state[agent][2]
            else:
                sell_sum += self.action_state[agent][2]

        if (buy_sum + sell_sum) > 0:
            for agent in self.agents:
                amount = self.action_state[agent][2]
                if amount > 0:
                    self.tr_dic[agent] = (amount / buy_sum) * sell_sum
                else:
                    self.tr_dic[agent] = amount
        else:
            for agent in self.agents:
                amount = self.action_state[agent][2]
                if amount < 0:
                    self.tr_dic[agent] = (amount / abs(sell_sum)) * buy_sum
                else:
                    self.tr_dic[agent] = amount

        self.D_sum = 0
        self.G_sum = 0
        self.total_r = 0
        # observe the current state
        for i in self.agents:
            # print(self.agent_name_mapping[i])
            # self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
            # current state consist of last electronic need, now electronic need
            # last allocation, last price given from agent, last price given from administrator.

            self.real_state[i] = [self.P_array[self.num_moves - 1, self.agent_name_mapping[i]],
                                  self.L_list[self.num_moves - 1, self.agent_name_mapping[i]],
                                  self.real_state[i][2] - self.action_state[i][0]
                                  ]
            # print('act', self.action_state[i])
            # print('state', self.real_state[i])
            # print(self.real_state[i][2])
            self.observations[i] = [self.range_norm(bottom_bound=np.min(self.P_array), up_bound=self.P_max,
                                                    value=self.real_state[i][0]),
                                    self.range_norm(bottom_bound=self.L_min, up_bound=self.L_max,
                                                    value=self.real_state[i][1]),
                                    self.range_norm(bottom_bound=0, up_bound=self.Total_L * 0.5,
                                                    value=self.real_state[i][2])
                                    ]
            # print('obs',self.observations[i][2])
            if self.action_state[i][0] > 0:

                self.P_u[i] = self.real_state[i][1] + self.action_state[i][0] + self.tr_dic[i] - \
                              self.action_state[i][1] - self.real_state[i][0]
            else:

                self.P_u[i] = self.real_state[i][1] - self.action_state[i][0] + \
                              self.tr_dic[i] - \
                              self.action_state[i][1] - self.real_state[i][0]
            # print(self.allocation_state)
            # print(self.electronic_need[self.num_moves-1])
            # rewards for all agents are placed in the .rewards dictionary
            self.Ftr = 0.02
            self.rewards[i] = - self.sell_price * max(0, self.P_u[i]) - self.buy_price * min(0, self.P_u[
                i]) + self.beta_p2p * self.tr_dic[i] - \
                              1 / 2 * self.Ftr - (
                                          self.a * (self.action_state[i][1] ** 2) + self.b * self.action_state[i][
                                      1] + self.c)
            # print('r', self.rewards[i])
            self.D_sum = self.D_sum + max(0, self.P_u[i])
            self.G_sum = self.G_sum - min(0, self.P_u[i])
            self.total_r += self.rewards[i]

        # self.D_sum = max(0, self.D_sum)
        # self.G_sum = - min(0, self.G_sum)
        self.P_total_h = - 0.5 * self.p_ug * min(self.D_sum, self.G_sum) + self.total_r
        self.P_total += self.P_total_h

        rewards_list = []
        observation_list = []
        for agent in self.agents:
            observation_list.append(self.observations[agent])
            rewards_list.append([self.rewards[agent]])


        # selects the next agent.
        # self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        # self._accumulate_rewards()
        return observation_list,rewards_list, dones, infos

    def allocation_price_cal(self):
        """
        decide the sale price and allocation for every agent.
        :return:
        """
        c = [self.action_state['player_0'][1], self.action_state['player_1'][1],
             self.action_state['player_2'][1]]
        # print(c)
        A = np.array([1, 1, 1]).reshape(1, 3)
        b = np.array([self.electronic_need[self.num_moves]]).reshape(1, 1)

        x0_bounds = [0, self.real_action_range[0]]
        x1_bounds = [0, self.real_action_range[1]]
        x2_bounds = [0, self.real_action_range[2]]

        res = linprog(c, A_eq=A, b_eq=b, bounds=[x0_bounds, x1_bounds, x2_bounds])

        # self.allocation_state[0],self.allocation_state[1],self.allocation_state[2] = \
        #     [res.x[0], res.x[1], res.x[2]]
        for agent in self.agents[:3]:
            self.allocation_state[agent] = res.x[self.agent_name_mapping[agent]]
        self.sale_price_state = max(res.x)
        return

    def range_norm(self, bottom_bound, up_bound, value):
        normed_value = (value - (bottom_bound + up_bound) / 2) / ((up_bound - bottom_bound) / 2)
        return normed_value


if __name__ == '__main__':
    # csv_data_l = pd.read_excel('./20230201code+data/data_l_33.xlsx')
    # # print(csv_data_l.values)
    # L = csv_data_l.values[:, 1:]
    # # print(L.shape)
    # csv_data_p = pd.read_excel('./20230201code+data/data_p_33.xlsx')
    # P = csv_data_p.values[:,1:]
    # print(P.shape)
    env = MA_ELEC_Market_ENV(max_step=24,custom_agent_num=118)
    # env = wrapped_env(max_step=24, custom_agent_num=33)
    # # api_test(env, num_cycles=10, verbose_progress=False)
    # random_demo(env, render=True, episodes=1)
    # env.reset()
    # for agent in env.agent_iter():
    #     observation, reward, done, info = env.last()
    #     action = policy(observation, agent)
    #     env.step(action)

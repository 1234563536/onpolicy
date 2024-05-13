import gym
import pandas as pd
from gym import spaces
from gym.spaces import Discrete, Box, MultiBinary
import numpy as np
import os
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from scipy.optimize import linprog
from pettingzoo.test import api_test
from pettingzoo.utils import random_demo
# from stable_baselines3.common.env_checker import check_env
from scipy.optimize import minimize
import matlab.engine
import scipy.io as scio
from copy import deepcopy as dc
from onpolicy.envs.mpe.distflow import PyDistflow

eng = matlab.engine.start_matlab()

MainPath = r'D:\powerdist\on-policy-main\on-policy-main-0127'  # os.getcwd()
eng.addpath(MainPath + '\onpolicy\scripts\train')
df = eng.environment()


def transtype(data):
    if type(data) is list:
        return matlab.double(data)
    else:
        return matlab.double(data.tolist())


bus = scio.loadmat(MainPath + "/onpolicy/envs/mpe/case118zh/busdata.mat")["busdata"]
# branch = scio.loadmat(r"D:\powerdist\on-policy-main\onpolicy\envs\mpe\case118zh\branchdata.mat")['branchdata']
# print('branch',branch)
elec_graph = '''
    1	2	0.036	0.01296	0	0	0	0	0	0	1	-360	360;
	2	3	0.033	0.01188	0	0	0	0	0	0	1	-360	360;
	2	4	0.045	0.0162	0	0	0	0	0	0	1	-360	360;
	4	5	0.015	0.054	0	0	0	0	0	0	1	-360	360;
	5	6	0.015	0.054	0	0	0	0	0	0	1	-360	360;
	6	7	0.015	0.0125	0	0	0	0	0	0	1	-360	360;
	7	8	0.018	0.014	0	0	0	0	0	0	1	-360	360;
	8	9	0.021	0.063	0	0	0	0	0	0	1	-360	360;
	2	10	0.166	0.1344	0	0	0	0	0	0	1	-360	360;
	10	11	0.112	0.0789	0	0	0	0	0	0	1	-360	360;
	11	12	0.187	0.313	0	0	0	0	0	0	1	-360	360;
	12	13	0.142	0.1512	0	0	0	0	0	0	1	-360	360;
	13	14	0.18	0.118	0	0	0	0	0	0	1	-360	360;
	14	15	0.15	0.045	0	0	0	0	0	0	1	-360	360;
	15	16	0.16	0.18	0	0	0	0	0	0	1	-360	360;
	16	17	0.157	0.171	0	0	0	0	0	0	1	-360	360;
	11	18	0.218	0.285	0	0	0	0	0	0	1	-360	360;
	18	19	0.118	0.185	0	0	0	0	0	0	1	-360	360;
	19	20	0.16	0.196	0	0	0	0	0	0	1	-360	360;
	20	21	0.12	0.189	0	0	0	0	0	0	1	-360	360;
	21	22	0.12	0.0789	0	0	0	0	0	0	1	-360	360;
	22	23	1.41	0.723	0	0	0	0	0	0	1	-360	360;
	23	24	0.293	0.1348	0	0	0	0	0	0	1	-360	360;
	24	25	0.133	0.104	0	0	0	0	0	0	1	-360	360;
	25	26	0.178	0.134	0	0	0	0	0	0	1	-360	360;
	26	27	0.178	0.134	0	0	0	0	0	0	1	-360	360;
	4	28	0.015	0.0296	0	0	0	0	0	0	1	-360	360;
	28	29	0.012	0.0276	0	0	0	0	0	0	1	-360	360;
	29	30	0.12	0.2766	0	0	0	0	0	0	1	-360	360;
	30	31	0.21	0.243	0	0	0	0	0	0	1	-360	360;
	31	32	0.12	0.054	0	0	0	0	0	0	1	-360	360;
	32	33	0.178	0.234	0	0	0	0	0	0	1	-360	360;
	33	34	0.178	0.234	0	0	0	0	0	0	1	-360	360;
	34	35	0.154	0.162	0	0	0	0	0	0	1	-360	360;
	30	36	0.187	0.261	0	0	0	0	0	0	1	-360	360;
	36	37	0.133	0.099	0	0	0	0	0	0	1	-360	360;
	29	38	0.33	0.194	0	0	0	0	0	0	1	-360	360;
	38	39	0.31	0.194	0	0	0	0	0	0	1	-360	360;
	39	40	0.13	0.194	0	0	0	0	0	0	1	-360	360;
	40	41	0.28	0.15	0	0	0	0	0	0	1	-360	360;
	41	42	1.18	0.85	0	0	0	0	0	0	1	-360	360;
	42	43	0.42	0.2436	0	0	0	0	0	0	1	-360	360;
	43	44	0.27	0.0972	0	0	0	0	0	0	1	-360	360;
	44	45	0.339	0.1221	0	0	0	0	0	0	1	-360	360;
	45	46	0.27	0.1779	0	0	0	0	0	0	1	-360	360;
	35	47	0.21	0.1383	0	0	0	0	0	0	1	-360	360;
	47	48	0.12	0.0789	0	0	0	0	0	0	1	-360	360;
	48	49	0.15	0.0987	0	0	0	0	0	0	1	-360	360;
	49	50	0.15	0.0987	0	0	0	0	0	0	1	-360	360;
	50	51	0.24	0.1581	0	0	0	0	0	0	1	-360	360;
	51	52	0.12	0.0789	0	0	0	0	0	0	1	-360	360;
	52	53	0.405	0.1458	0	0	0	0	0	0	1	-360	360;
	53	54	0.405	0.1458	0	0	0	0	0	0	1	-360	360;
	29	55	0.391	0.141	0	0	0	0	0	0	1	-360	360;
	55	56	0.406	0.1461	0	0	0	0	0	0	1	-360	360;
	56	57	0.406	0.1461	0	0	0	0	0	0	1	-360	360;
	57	58	0.706	0.5461	0	0	0	0	0	0	1	-360	360;
	58	59	0.338	0.1218	0	0	0	0	0	0	1	-360	360;
	59	60	0.338	0.1218	0	0	0	0	0	0	1	-360	360;
	60	61	0.207	0.0747	0	0	0	0	0	0	1	-360	360;
	61	62	0.247	0.8922	0	0	0	0	0	0	1	-360	360;
	1	63	0.028	0.0418	0	0	0	0	0	0	1	-360	360;
	63	64	0.117	0.2016	0	0	0	0	0	0	1	-360	360;
	64	65	0.255	0.0918	0	0	0	0	0	0	1	-360	360;
	65	66	0.21	0.0759	0	0	0	0	0	0	1	-360	360;
	66	67	0.383	0.138	0	0	0	0	0	0	1	-360	360;
	67	68	0.504	0.3303	0	0	0	0	0	0	1	-360	360;
	68	69	0.406	0.1461	0	0	0	0	0	0	1	-360	360;
	69	70	0.962	0.761	0	0	0	0	0	0	1	-360	360;
	70	71	0.165	0.06	0	0	0	0	0	0	1	-360	360;
	71	72	0.303	0.1092	0	0	0	0	0	0	1	-360	360;
	72	73	0.303	0.1092	0	0	0	0	0	0	1	-360	360;
	73	74	0.206	0.144	0	0	0	0	0	0	1	-360	360;
	74	75	0.233	0.084	0	0	0	0	0	0	1	-360	360;
	75	76	0.591	0.1773	0	0	0	0	0	0	1	-360	360;
	76	77	0.126	0.0453	0	0	0	0	0	0	1	-360	360;
	64	78	0.559	0.3687	0	0	0	0	0	0	1	-360	360;
	78	79	0.186	0.1227	0	0	0	0	0	0	1	-360	360;
	79	80	0.186	0.1227	0	0	0	0	0	0	1	-360	360;
	80	81	0.26	0.139	0	0	0	0	0	0	1	-360	360;
	81	82	0.154	0.148	0	0	0	0	0	0	1	-360	360;
	82	83	0.23	0.128	0	0	0	0	0	0	1	-360	360;
	83	84	0.252	0.106	0	0	0	0	0	0	1	-360	360;
	84	85	0.18	0.148	0	0	0	0	0	0	1	-360	360;
	79	86	0.16	0.182	0	0	0	0	0	0	1	-360	360;
	86	87	0.2	0.23	0	0	0	0	0	0	1	-360	360;
	87	88	0.16	0.393	0	0	0	0	0	0	1	-360	360;
	65	89	0.669	0.2412	0	0	0	0	0	0	1	-360	360;
	89	90	0.266	0.1227	0	0	0	0	0	0	1	-360	360;
	90	91	0.266	0.1227	0	0	0	0	0	0	1	-360	360;
	91	92	0.266	0.1227	0	0	0	0	0	0	1	-360	360;
	92	93	0.266	0.1227	0	0	0	0	0	0	1	-360	360;
	93	94	0.233	0.115	0	0	0	0	0	0	1	-360	360;
	94	95	0.496	0.138	0	0	0	0	0	0	1	-360	360;
	91	96	0.196	0.18	0	0	0	0	0	0	1	-360	360;
	96	97	0.196	0.18	0	0	0	0	0	0	1	-360	360;
	97	98	0.1866	0.122	0	0	0	0	0	0	1	-360	360;
	98	99	0.0746	0.318	0	0	0	0	0	0	1	-360	360;
	1	100	0.0625	0.0265	0	0	0	0	0	0	1	-360	360;
	100	101	0.1501	0.234	0	0	0	0	0	0	1	-360	360;
	101	102	0.1347	0.0888	0	0	0	0	0	0	1	-360	360;
	102	103	0.2307	0.1203	0	0	0	0	0	0	1	-360	360;
	103	104	0.447	0.1608	0	0	0	0	0	0	1	-360	360;
	104	105	0.1632	0.0588	0	0	0	0	0	0	1	-360	360;
	105	106	0.33	0.099	0	0	0	0	0	0	1	-360	360;
	106	107	0.156	0.0561	0	0	0	0	0	0	1	-360	360;
	107	108	0.3819	0.1374	0	0	0	0	0	0	1	-360	360;
	108	109	0.1626	0.0585	0	0	0	0	0	0	1	-360	360;
	109	110	0.3819	0.1374	0	0	0	0	0	0	1	-360	360;
	110	111	0.2445	0.0879	0	0	0	0	0	0	1	-360	360;
	110	112	0.2088	0.0753	0	0	0	0	0	0	1	-360	360;
	112	113	0.2301	0.0828	0	0	0	0	0	0	1	-360	360;
	100	114	0.6102	0.2196	0	0	0	0	0	0	1	-360	360;
	114	115	0.1866	0.127	0	0	0	0	0	0	1	-360	360;
	115	116	0.3732	0.246	0	0	0	0	0	0	1	-360	360;
	116	117	0.405	0.367	0	0	0	0	0	0	1	-360	360;
	117	118	0.489	0.438	0	0	0	0	0	0	1	-360	360;
	46	27	0.5258	0.2925	0	0	0	0	0	0	0	-360	360;
	17	27	0.5258	0.2916	0	0	0	0	0	0	0	-360	360;
	8	24	0.4272	0.1539	0	0	0	0	0	0	0	-360	360;
	54	43	0.48	0.1728	0	0	0	0	0	0	0	-360	360;
	62	49	0.36	0.1296	0	0	0	0	0	0	0	-360	360;
	37	62	0.57	0.572	0	0	0	0	0	0	0	-360	360;
	9	40	0.53	0.3348	0	0	0	0	0	0	0	-360	360;
	58	96	0.3957	0.1425	0	0	0	0	0	0	0	-360	360;
	73	91	0.68	0.648	0	0	0	0	0	0	0	-360	360;
	88	75	0.4062	0.1464	0	0	0	0	0	0	0	-360	360;
	99	77	0.4626	0.1674	0	0	0	0	0	0	0	-360	360;
	108	83	0.651	0.234	0	0	0	0	0	0	0	-360	360;
	105	86	0.8125	0.2925	0	0	0	0	0	0	0	-360	360;
	110	118	0.7089	0.2553	0	0	0	0	0	0	0	-360	360;
	25	35	0.5000	0.5000	0	0	0	0	0	0	0	-360	360'''

allbranch = scio.loadmat(MainPath + "/onpolicy/envs/mpe/allbranch.mat")["allbranch"]  # 读取支路数据
params = dict()
params["busNum"] = 118
params["baseMVA"] = 10  # 功率基准值为10MVA
params["basekV"] = 10  # 电压基准值为11kV
params["baseI"] = params["baseMVA"] / params["basekV"]  # 电流基准值
params["I_max"] = (10 * 1 / params["baseI"]) ** 2  # 电流最大值
params["V_max"] = 1.07 ** 2  # 节点电压上限
params["V_min"] = 0.93 ** 2  # 节点电压下限
params["c"] = [0.00059, 0.302, 0]  # 发电成本
params["bus_num"] = np.array([i + 1 for i in range(params["busNum"])])
G_Data = scio.loadmat(MainPath + "/onpolicy/envs/mpe/gendata.mat")["gendata"]  # 读取发电机节点数据
params["gen_num"] = G_Data[:, 0].astype('int')  # 发电机节点编号
params["genP_max"] = dict(zip(params["gen_num"], G_Data[:, 8]))  # 发电机有功上限
# params["genP_max"] = 20
params["genQ_max"] = dict(zip(params["gen_num"], G_Data[:, 3]))  # 发电机无功上限
# print(params["genP_max"])
# params["genQ_max"] = 20
params["commen_num"] = list(set(params["bus_num"].tolist()) - set(params["gen_num"].tolist()))  # 非发电机节点
params["allbranch"] = allbranch


# print(allbranch.shape)


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


class MA_ELEC_Market_Graph_ENV(gym.Env):
    """
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self, max_step, custom_agent_num, gate_num=10):
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
        # self.gate_map = np.zeros((self.gate_num,2))
        self.sl_penalty = 0.01
        self.random_day = np.random.randint(100, 101, size=1) / 100
        self.gate_num = gate_num
        self.gate_map = np.zeros((self.gate_num, 2))
        self.gate_map = np.array(
            [[17, 27], [21, 22], [27, 46], [29, 38], [99, 77], [71, 72], [88, 75], [102, 103], [108, 109], [118, 110]])
        self.except_gate = np.array(
            [[8, 24], [25, 35], [9, 40], [43, 54], [37, 62], [49, 62], [58, 96], [91, 73], [83, 108], [86, 105]])

        self.custom_matrix = np.random.randint(2, size=(custom_agent_num, custom_agent_num))
        self.custom_agent_num = custom_agent_num
        csv_data_l = pd.read_excel(MainPath + '/onpolicy/envs/mpe/20230201code+data/data_l_118.xlsx')
        # print('custom',self.custom_agent_num)
        self.bus_data_selection = pd.read_excel(MainPath + '/onpolicy/envs/mpe/118bus_data.xlsx', sheet_name=None)

        self.L_list_day_array = csv_data_l
        # self.L_list = self.L_list_day_array[:,:,i]
        self.L_list_ori = np.array(csv_data_l.values[:, 0:custom_agent_num]) / 1e4
        # print('l_list', self.L_list.shape)
        # print('l_list_ori', self.L_list_ori.shape)

        # print(L.shape)
        # self.Total_L = np.sum(self.L_list_ori)/ custom_agent_num
        # print('l_list_ori', self.Total_L)

        csv_data_p = pd.read_excel(MainPath + '/onpolicy/envs/mpe/20230201code+data/data_p_118.xlsx')
        self.P_array_ori = np.array(csv_data_p.values[:, 0:custom_agent_num]) / 1e4
        # print('p_shape', self.P_array_ori.shape)
        # print('p_data', self.P_array_ori)

        self.P_total = 0
        self.p_ug = 1
        self.sell_price_list = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 1.3, 1.3, 1.3, 1.3, 0.8, 0.8, 0.8,
                                0.8, 1.3, 1.3, 1.3, 0.8, 0.8, 0.4]
        # = [0.15,0.12,0.10,0.09,0.09,0.08,0.083,0.078,0.098,0.11,0.123,0.139,0.145,0.165,0.163,0.145,0.162,0.143,0.143,0.166,0.241,0.242,0.214,0.204]
        self.E_max = np.sum(self.L_list_ori, axis=0) * 0.2
        # print('E_max', self.E_max)

        self.Pes_max = self.E_max * 0.33
        # print('bus_data',self.bus_data_selection.keys())
        pes_index = self.bus_data_selection['储能节点编号'].values.reshape(-1).tolist()
        pes_index = [i - 1 for i in pes_index]
        # print(pes_index)

        pes_indes = set(range(118)) - set(pes_index)
        # print(pes_indes)

        self.Pes_max[list(pes_indes)] = 0
        # print('self.Pes_max',self.Pes_max)

        self.L_sum_i = np.sum(self.L_list_ori, axis=0)
        # print('L_max_i',self.L_sum_i)

        self.L_max = np.max(self.L_list_ori)
        self.L_min = np.min(self.L_list_ori)
        self.L_max_ori = np.array([self.L_max] * custom_agent_num)

        # print('L_max',self.L_max)
        # print('L_min',self.L_min)

        dg_index = self.bus_data_selection['DG节点编号'].values.reshape(-1).tolist()
        dg_index = [i - 1 for i in dg_index]
        dg_index = set(range(118)) - set(dg_index)
        self.L_max_ori[list(dg_index)] = 0
        # print('self.L_max_ori', self.L_max_ori)

        self.sl_sum = self.L_sum_i * 0.2
        self.sl_max_i = np.max(self.L_list_ori, axis=0) * 0.7
        self.sl_min_i = 0
        # print('sl_max_i', self.sl_max_i)

        self.action_3_low_limit = self.sl_min_i
        self.action_3_high_limit = self.sl_max_i

        # print('self.L_max',self.L_max)
        self.P_max = np.max(self.P_array_ori) * 1.5
        self.P_min = np.min(self.P_array_ori) * 1.5
        # print('self.P_max',self.P_max)

        # self.action_2_high_limit = (30 + 50)/1e4
        # print(self.action_2_high_limit)
        self.F = scio.loadmat(MainPath + "/onpolicy/envs/mpe/20230201code+data/FL.mat")['FL']
        # print('F',np.array(self.F).shape)
        # self.action_2_low_limit = -(self.P_max+self.Pes_max)
        # self.action_2_low_limit = (-(50 + 50))/1e4
        # print('high', self.action_2_high_limit)
        # print('low', self.action_2_low_limit)
        # print(self.action_2_low_limit)

        # 按照分号换行进行切割，得到每一行的字符串
        rows = elec_graph.split(";")
        # print('elec_graph', elec_graph)

        # 使用列表推导式，将每一行的字符串按照空格进行分割，并转换为整数
        matrix = [[float(num) for num in row.split()] for row in rows]
        #    print('matrix', matrix)

        # print(matrix)
        self.elec_graph_data = np.array(matrix)
        self.edge_num_max = self.elec_graph_data.shape[0]
        self.elec_graph = self.init_graph(self.elec_graph_data)
        # print(self.elec_graph_data)
        self.get_avaliable_action_list()
        self.elec_action_num = len(self.avaliable_action_list)
        self.elec_dim = self.gate_num + 2 + 23
        # print('elec_dim',self.elec_dim)
        # print('gate_num', self.gate_num)
        # print('edge_num_max',self.edge_num_max)
        # print('avaliable_action_list',self.avaliable_action_list)
        # print('elec_action_num',self.elec_action_num)

        self.beta_p2p = 1
        self.buy_price = 0.35
        self.sell_price = self.sell_price_list[0]
        self.max_step = max_step
        self.agent_num = custom_agent_num + 1
        # action range for agent
        # self.real_action_range = [60, 80, 100]
        self.possible_agents = ["player_" + str(r) for r in range(self.agent_num)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        # print('agent_num',self.agent_num)
        # elec parameter calculation

        # print(self.agent_name_mapping)
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        # Pes G Tr
        self.Pv_dic = {agent: None for agent in self.possible_agents[:-1]}
        # print('PV', self.Pv_dic)

        self.action_space = [Box(low=-1, high=1, shape=(4,)) for agent in self.possible_agents[:-1]]

        self.action_space.append(Discrete(self.elec_action_num))
        self.observation_space = [Box(low=-1, high=1, shape=(3,)) for agent in self.possible_agents[:-1]]
        self.observation_space.append(Box(low=-1, high=1, shape=(self.gate_num + 2 + 23,)))
        share_obs_dim = 3 * custom_agent_num + self.gate_num + 2 + 23
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in
            range(self.custom_agent_num + 1)]
        # print('elec_action',(self.elec_action_num))

        self.l_up_bound = np.zeros((self.custom_agent_num, self.custom_agent_num))
        self.v_up_bound = np.zeros((self.custom_agent_num))
        self.v_bottom_bound = np.zeros((self.custom_agent_num))
        self.p_up_bound = np.zeros((self.custom_agent_num))
        self.p_bottom_bound = np.zeros((self.custom_agent_num))
        self.q_up_bound = np.zeros((self.custom_agent_num))
        self.q_bottom_bound = np.zeros((self.custom_agent_num))

        # self.share_observation_space = [spaces.Box(
        #     low=-np.inf, high=+np.inf, shape=(3*custom_agent_num,), dtype=np.float32) for _ in self.possible_agents]

    def generate_binary_combinations(self, n):
        combinations = []
        max_value = 2 ** n

        for i in range(max_value):
            binary_string = bin(i)[2:].zfill(n)  # 转换为二进制字符串，并填充到指定长度
            combinations.append(binary_string)
        return combinations

    def is_tree(self, cmb):
        # print(len(cmb))
        current_graph = self.elec_graph.copy()

        for i in range(len(cmb)):
            if cmb[i] == '1':
                # print(cmb[i])
                # print(self.gate_map[i,0]-1)
                current_graph[self.gate_map[i, 0] - 1, self.gate_map[i, 1] - 1] = 1
                current_graph[self.gate_map[i, 1] - 1, self.gate_map[i, 0] - 1] = 1
            else:
                # print(self.gate_map[i, 0])
                current_graph[self.gate_map[i, 0] - 1, self.gate_map[i, 1] - 1] = 0
                current_graph[self.gate_map[i, 1] - 1, self.gate_map[i, 0] - 1] = 0

        if self.is_tree_matrix(current_graph.astype('int').tolist()):
            return True
        else:
            return False

    def is_tree_matrix(self, adj_matrix):
        num_nodes = len(adj_matrix)

        # Check if the number of edges is equal to (num_nodes - 1)
        num_edges = sum(sum(row) for row in adj_matrix) // 2
        if num_edges != num_nodes - 1:
            return False

        visited = [False] * num_nodes

        # Check if the graph is acyclic and connected
        def dfs(node, parent):
            visited[node] = True
            for neighbor in range(num_nodes):
                if adj_matrix[node][neighbor] == 1:
                    if visited[neighbor] and neighbor != parent:
                        return False
                    if not visited[neighbor] and not dfs(neighbor, node):
                        return False
            return True

        # Start the DFS from an arbitrary node
        if not dfs(0, -1):
            return False

        # Check if all nodes are visited
        if not all(visited):
            return False

        return True

    def is_rooted_tree(self, adjMatrix):
        n = len(adjMatrix)
        # print(adjMatrix[110-1])
        # print(adjMatrix)
        # print('n',n)
        # 条件1：检查是否存在环路
        visited = [False] * n
        stack = [1]  # 从1号节点开始深度优先遍历
        while stack:
            node = stack.pop()
            if visited[node - 1]:
                print(node)
                print('circle')
                return False
            visited[node - 1] = True
            neighbors = [i for i in range(1, n + 1) if adjMatrix[node - 1][i - 1] != 0]
            print('node', node)
            print(neighbors)
            stack.extend(neighbors)

        # 条件2：检查是否存在一条从1号节点到达其他所有节点的路径
        visited = [False] * n
        stack = [1]  # 从1号节点开始深度优先遍历
        while stack:
            node = stack.pop()
            visited[node - 1] = True
            neighbors = [i for i in range(1, n + 1) if adjMatrix[node - 1][i - 1] != 0]
            stack.extend(neighbors)

        if False in visited:
            print('vis')
            return False

        # 条件3：检查是否存在多个入度为1的节点
        in_degrees = [0] * n
        for i in range(n):
            for j in range(n):
                if adjMatrix[i][j] != 0:
                    in_degrees[j] += 1
                    if in_degrees[j] >= 2:
                        print('num')
                        return False
        # print(in_degrees)

        return True

    def get_avaliable_action_list(self):
        action_list = []

        cmb_list = self.generate_binary_combinations(self.gate_num)
        # print(cmb_list)
        for cmb in cmb_list:
            if self.is_tree(cmb):
                action_list.append(cmb)

        self.avaliable_action_list = action_list

        # print('list', len(self.avaliable_action_list))

        return

    def init_graph(self, graph_data):

        graph = np.zeros((self.custom_agent_num, self.custom_agent_num))
        # print(graph_data)
        for i in range(graph_data.shape[0]):
            if int(graph_data[i, -3]) == 1:
                graph[int(graph_data[i, 0]) - 1, int(graph_data[i, 1]) - 1] = 1
                graph[int(graph_data[i, 1]) - 1, int(graph_data[i, 0]) - 1] = 1
        # for i in range(self.except_gate.shape[0]):
        #     graph[int(self.except_gate[i,0])-1, int(self.except_gate[i,1])-1] = 0
        #     graph[int(self.except_gate[i, 1]) - 1, int(self.except_gate[i, 0]) - 1] = 0

        return graph

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
        day_cof = np.random.choice(self.random_day)
        self.L_list = self.L_list_ori * day_cof
        self.P_array = self.P_array_ori * day_cof
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.action_state = {agent: None for agent in self.agents}
        self.real_state = {agent: None for agent in self.agents}
        self.s_l_state = {agent: 0 for agent in self.agents}
        # self.allocation_state = {agent: None for agent in self.agents[:3]}
        # self.sale_price_state = None
        self.observations = {agent: None for agent in self.agents}
        self.P_u = {agent: None for agent in self.agents[:-1]}
        self.a = 0.00059
        self.b = 0.302
        self.c = 0
        for i in self.agents[:-1]:
            # print(self.agent_name_mapping[i])
            # self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
            # Pv L Es
            self.observations[i] = np.zeros((3,))
            self.real_state[i] = np.zeros((3,))
            # self.s_l_state[i] = np.zeros((1,))

        self.observations[self.agents[-1]] = np.zeros((self.elec_dim,))
        self.real_state[self.agents[-1]] = np.zeros((self.elec_dim,))

        # self.s_l_max = {agent: 0 for agent in self.agents}
        # self.s
        self.num_moves = 0
        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        # self.init_graph(self.elec_graph_data)
        self.elec_graph = self.init_graph(self.elec_graph_data)
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
        infos = []
        dones = [self.num_moves >= (self.max_step - 1) for agent in self.agents]
        """
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
        # elsße:
        #     if action[0] == -1:
        #         self.action_state[self.agent_selection] = np.array([0, 0])
        #     else:
        #         self.action_state[self.agent_selection] = np.array([1, 1250 + action[1] * 250])
        #
        # print(action)
        """
        for i in range(self.agent_num - 1):
            action = dc(actions[i])
            self.agent_selection = self.agents[i]
            self.action_state[self.agent_selection] = dc(action)
            self.action_state[self.agent_selection][0] = action[0] * self.Pes_max[i]
            if self.action_state[self.agent_selection][0] > self.Pes_max[i]:
                self.action_state[self.agent_selection][0] = self.Pes_max[i]
            if self.action_state[self.agent_selection][0] < -self.Pes_max[i]:
                self.action_state[self.agent_selection][0] = -self.Pes_max[i]

            self.action_state[self.agent_selection][1] = ((action[1] + 1) / 2) * self.L_max_ori[i] * 0.8
            if self.action_state[self.agent_selection][1] > self.L_max_ori[i] * 0.8:
                self.action_state[self.agent_selection][1] = self.L_max_ori[i] * 0.8
            if self.action_state[self.agent_selection][1] < 0:
                self.action_state[self.agent_selection][1] = 0
            # self.action_state[self.agent_selection][2] = ((action[2] + 1) / 2) * (
            #             self.action_2_high_limit - self.action_2_low_limit) + self.action_2_low_limit
            self.action_state[self.agent_selection][3] = ((action[3] + 1) / 2) * (
                    self.action_3_high_limit[i] - self.action_3_low_limit) + self.action_3_low_limit

            if self.action_state[self.agent_selection][3] > self.action_3_high_limit[i]:
                self.action_state[self.agent_selection][3] = self.action_3_high_limit[i]
            if self.action_state[self.agent_selection][3] < self.action_3_low_limit:
                self.action_state[self.agent_selection][3] = self.action_3_low_limit

            self.s_l_state[self.agent_selection] = self.s_l_state[self.agent_selection] + \
                                                   self.action_state[self.agent_selection][3]
            if self.s_l_state[self.agent_selection] > self.sl_sum[i]:
                # print()
                self.action_state[self.agent_selection][3] = self.action_state[self.agent_selection][3] - \
                                                             self.s_l_state[self.agent_selection] + self.sl_sum[i]

            # print('total', self.Total_L * 0.5)
            if (self.real_state[self.agent_selection][2] - 1.05 * self.action_state[self.agent_selection][0]) < 0:
                self.action_state[self.agent_selection][0] = self.real_state[self.agent_selection][2] / 1.05
            if self.real_state[self.agent_selection][2] - 0.95 * self.action_state[self.agent_selection][0] > (
                    self.E_max[i]):
                self.action_state[self.agent_selection][0] = (self.real_state[self.agent_selection][2] - (
                    self.E_max[i])) / 0.95

        print('s_l_state', self.s_l_state)
        print('action_state', self.action_state)

        action = dc(actions[-1])
        self.agent_selection = self.agents[-1]
        self.action_state[self.agent_selection] = action
        # self.TR_state[self.agent_name_mapping[self.agent_selection]] = sum(action[])
        # collect reward if it is the last agent to act

        self.sell_price = self.sell_price_list[self.num_moves]

        self.num_moves += 1
        # The dones dictionary must be updated for all players.
        self.dones = {agent: self.num_moves >= (self.max_step - 1) for agent in self.agents}
        self.tr_dic = {agent: None for agent in self.agents[:-1]}
        # print('gate', list(self.avaliable_action_list[int(self.action_state[self.agents[-1]])]))
        self.graph_trans()
        T = np.zeros((self.custom_agent_num - 1 + 15,))

        for i in range(self.elec_graph_data.shape[0]):
            if self.elec_graph[int(self.elec_graph_data[i, 0]) - 1, int(self.elec_graph_data[i, 1]) - 1] == 1:
                T[i] = 1
        self.Gate_T = T
        # T[-10:] = list(self.avaliable_action_list[int(self.action_state[self.agents[-1]])])
        # np.save('T' + str(self.num_moves), T)
        # print("T",T)
        ## matlab gurobi
        T_mat = transtype(T)

        # print('T',T)
        for i in self.agents[:-1]:
            self.tr_dic[i] = self.real_state[i][0] + self.action_state[i][0] \
                             + self.action_state[i][1] - self.real_state[i][1]

        buy_sum = 0
        sell_sum = 0
        for agent in self.agents[:-1]:
            if self.tr_dic[agent] > 0:
                sell_sum += self.tr_dic[agent]
            else:
                buy_sum += self.tr_dic[agent]

        buy_sum = np.float64(buy_sum)
        sell_sum = np.float64(sell_sum)

        # print('buy_sum',buy_sum)
        # print('sell_sum',sell_sum)

        if (buy_sum + sell_sum) > 0:
            for agent in self.agents[:-1]:
                amount = self.tr_dic[agent]
                if amount > 0:
                    self.tr_dic[agent] = (amount / (-sell_sum)) * buy_sum
                else:
                    self.tr_dic[agent] = amount
        else:
            for agent in self.agents[:-1]:
                amount = self.tr_dic[agent]
                if amount > 0:
                    self.tr_dic[agent] = amount
                else:
                    self.tr_dic[agent] = (-amount / buy_sum) * abs(sell_sum)

        # print('tr_sum', sum(self.tr_dic.values()))
        print('trading', self.tr_dic)
        # print('self.action_state.values()',self.action_state.values())
        action_stateDict = dc(self.action_state)
        trDict = dc(self.tr_dict)
        infos.append([i.tolist() for i in action_stateDict.values()])
        infos.append(list(trDict.values()))
        # print('infos',infos)

        ptr_vec = []
        for agent in self.agents[:-1]:
            ptr_vec.append(self.tr_dic[agent])
        # np.save('ptr_vec'+str(self.num_moves),ptr_vec)
        # p.save('')
        # print('ptr',ptr_vec)

        ptr_vec = transtype(np.array(ptr_vec).reshape(-1, 1))
        # ptr_vec = transtype(np.array([0.001] * 59 + [-0.001] * 59).reshape(-1, 1))
        [ptr_matrix, FTR, g] = eng.Calc_P2P(T_mat, ptr_vec, df, nargout=3)
        # print('FTR',np.array(FTR).shape)
        # print('ptr_matrix',np.array(ptr_matrix))
        i_ = 0
        for agent in self.agents[:-1]:
            self.tr_dic[agent] = sum(np.array(ptr_matrix)[i_, :])
            # print('tr_dic',self.tr_dic[agent])
            i_ += 1

        self.D_sum = 0
        self.G_sum = 0
        self.total_r = 0
        # observe the current state
        j_ = 0
        k_num = 0
        for i in self.agents[:-1]:
            # print(self.agent_name_mapping[i])
            # self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
            # current state consist of last electronic need, now electronic need
            # last allocation, last price given from agent, last price given from administrator.

            self.real_state[i] = [self.P_array[self.num_moves - 1, self.agent_name_mapping[i]],
                                  self.L_list[self.num_moves - 1, self.agent_name_mapping[i]] + self.action_state[i][3],
                                  self.real_state[i][2] - self.action_state[i][0]
                                  ]
            # print('act', self.action_state[i])
            # print('state', self.real_state[i])
            # print(self.real_state[i][2])
            self.observations[i] = [self.range_norm(bottom_bound=np.min(self.P_array), up_bound=self.P_max,
                                                    value=self.real_state[i][0]),
                                    self.range_norm(bottom_bound=self.L_min, up_bound=self.L_max,
                                                    value=self.real_state[i][1]),
                                    self.range_norm(bottom_bound=0, up_bound=self.E_max[k_num],
                                                    value=self.real_state[i][2])
                                    ]

            # print('obs',self.observations[i][2])
            # if self.action_state[i][0] > 0:

            self.P_u[i] = self.real_state[i][1] - self.action_state[i][0] + self.tr_dic[i] - \
                          self.action_state[i][1] - self.real_state[i][0]
            # else:
            #
            #     self.P_u[i] = self.real_state[i][1] - self.action_state[i][0] + \
            #                   self.tr_dic[i] - \
            #                   self.action_state[i][1] - self.real_state[i][0]

            # print(self.allocation_state)
            # print(self.electronic_need[self.num_moves-1])
            # rewards for all agents are placed in the .rewards dictionary
            self.Ftr = np.array(FTR)[0, :]

            self.beta_p2p = 0.5 * (self.buy_price + self.sell_price)
            self.rewards[i] = - self.sell_price * max(0, self.P_u[i]) * 1e4 - self.buy_price * min(0, self.P_u[
                i]) * 1e4 + self.beta_p2p * self.tr_dic[i] * 1e4 - \
                              1 / 2 * self.Ftr[k_num] - (
                                      self.a * ((self.action_state[i][1] * 1e4) ** 2) + self.b * self.action_state[i][
                                  1] * 1e4 + self.c)

            if self.num_moves >= self.max_step:
                self.rewards[i] += self.sl_penalty * abs(self.s_l_state[i] > self.sl_sum[j_])
            j_ += 1
            # print('r', self.rewards[i])
            k_num += 1
            self.D_sum = self.D_sum + max(0, self.P_u[i])
            self.G_sum = self.G_sum - min(0, self.P_u[i])
            self.total_r += self.rewards[i]
        # self.elec_flow()

        # PN = transtype(self.P_u)
        print('P_u', self.P_u)
        print('trade', self.tr_dic)
        PN = +np.array([(self.P_u[agent]) - (self.tr_dic[agent]) for agent in self.agents[:-1]])

        # if sum(PN)>5:
        #     print('sum_PN',sum(PN))

        # np.save('PN'+str(self.num_moves),PN)
        # print('PN',PN)

        # gurobi matlab
        # PN_mat = transtype(PN)

        # print('QN',len(bus[:,3]/10./2.))
        QN = +self.L_list_ori[self.num_moves - 1, :] * 0.3
        # print('QN',QN)

        # gurobi matlab
        # QN = transtype(bus[:,3]/1e4)
        # np.save('QN'+str(self.num_moves),bus[:,3]/1e4)
        # np.save('T'+str(self.num_moves),T)
        # print('T_dot', np.array([1] * 117 + [0] * 15).shape)
        # T = transtype(np.array([1] * 117 + [0] * 15))
        # 2、PN: 节点有功负荷
        # PN = transtype(bus[:, 2] / 10. / 2.)
        # print('PN', len(bus[:, 2]))
        # # 3、QN: 节点无功负荷
        # print('QN', len(bus[:, 3]))
        # QN = transtype(bus[:, 3] / 10. / 2.)

        # one day
        # [Q, R, L, V, X, Pg] = eng.Calc_Distflow(T, PN, QN, df, nargout=6)
        # 读取常量

        ################### 30 days#############

        # params={}
        # busNum = params["busNum"]
        # baseMVA = params["baseMVA"]  # 功率基准值为10MVA
        # basekV = params["basekV"]  # 电压基准值为11kV
        # baseI = params["baseI"]  # 电流基准值
        # I_max = params["I_max"]  # 电流最大值
        # V_max = params["V_max"]  # 节点电压上限
        # V_min = params["V_min"]  # 节点电压下限
        # c = params["c"]  # 发电成本
        #
        # allbranch = params["allbranch"]
        # bus_num = params["bus_num"]
        # gen_num = params["gen_num"]
        # genP_max = params["genP_max"]
        # genQ_max = params["genQ_max"]
        # commen_num = params["commen_num"]
        #
        #
        #
        #
        # params={}
        #
        # print('T', T.shape)
        print('PN', PN)
        print('QN', QN.shape)
        print('hours', self.num_moves - 1)
        print('T', T)
        # pd.DataFrame(QN).to_csv('QN'+str(self.num_moves)+'.csv')
        V, L, R, X, Pg = PyDistflow(T, PN, QN, params)

        #########################################################
        self.observations[self.agents[-1]] = self.get_net_observation()
        self.rewards[self.agents[-1]] = self.get_net_reward(R, L)

        # self.D_sum = max(0, self.D_sum)
        # self.G_sum = - min(0, self.G_sum)
        self.p_ug = self.sell_price_list[int(self.num_moves - 1)] - self.buy_price
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

        return observation_list, rewards_list, dones, infos

    def graph_trans(self):

        gate_list = list(self.avaliable_action_list[int(self.action_state[self.agents[-1]])])
        for i in range(len(gate_list)):
            if gate_list[i] == '1':
                self.elec_graph[self.gate_map[i, 0] - 1, self.gate_map[i, 1] - 1] = 1
                self.elec_graph[self.gate_map[i, 1] - 1, self.gate_map[i, 0] - 1] = 1
            else:
                self.elec_graph[self.gate_map[i, 0] - 1, self.gate_map[i, 1] - 1] = 0
                self.elec_graph[self.gate_map[i, 1] - 1, self.gate_map[i, 0] - 1] = 0
        return

    def get_net_reward(self, R, L):
        reward = 0
        R = np.array(R)
        L = np.array(L)
        # print('L',L.shape)
        # orig_pen = np.sum(self.elec_graph[:,0])
        parent_pen = 0
        # print('L',L[:,2])

        for i in range(self.elec_graph_data.shape[0]):
            if self.elec_graph[int(self.elec_graph_data[i, 0]) - 1, int(self.elec_graph_data[i, 1]) - 1] == 1:
                l_data = 0.0
                for j in range(L.shape[0]):
                    # print('L_0',L[j,0])
                    # print('L_1', L[j, 1])
                    if int(L[j, 0]) == (int(self.elec_graph_data[i, 0]) - 1) and int(L[j, 1]) == (
                            int(self.elec_graph_data[i, 1]) - 1):
                        l_data = L[j, 2]
                        # print('i_reward', l_data)
                        break
                    elif int(L[j, 1]) == (int(self.elec_graph_data[i, 0]) - 1) and int(L[j, 0]) == (
                            int(self.elec_graph_data[i, 1]) - 1):
                        l_data = L[j, 2]
                        # print('i_reward', l_data)
                        break
                reward += - (R[int(self.elec_graph_data[i, 0]) - 1, int(self.elec_graph_data[i, 1]) - 1] * l_data)

        # print('reward',reward)
        return reward

    def get_net_observation(self):
        obs = []
        obs = np.zeros((self.elec_dim))
        obs[:self.gate_num] = list(self.avaliable_action_list[int(self.action_state[self.agents[-1]])])
        # print(obs[:self.gate_num])
        # obs[self.gate_num:self.custom_agent_num+self.gate_num] = self.Gate_T

        # print('num_moves',self.num_moves)
        obs[-25:-1] = self.sell_price_list
        obs[-1] = 0.35
        # print('obs',obs)
        return obs

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

    def elec_flow(self, p_, q_, r_, x_, custom_agent_num):

        self.p_ = np.zeros((self.custom_agent_num,))
        self.q_ = np.zeros((self.custom_agent_num,))
        self.r_ = np.zeros((self.custom_agent_num, self.custom_agent_num))
        self.x_ = np.zeros((self.custom_agent_num, self.custom_agent_num))

        #
        # self.p_p = np.zeros((self.custom_agent_num,self.custom_agent_num))
        # self.q_q = np.zeros((self.custom_agent_num,self.custom_agent_num))
        # self.v_ = np.zeros((self.custom_agent_num,))
        #
        #
        # self.eq_constraint_p = np.zeros((self.custom_agent_num,))
        # self.eq_constraint_v = np.zeros((self.custom_agent_num, self.custom_agent_num))
        #
        # for j in range(self.custom_agent_num):
        #     self.eq_constraint_p[j] = self.p_p[j,:] @ self.elec_graph[j:].T - (self.p_p[:,j] - self.r_[:,j]*self.l_[:,j]).T @ self.elec_graph[:,j] - self.p_[j]
        #     self.eq_constraint_q[j] = self.q_q[j,:] @ self.elec_graph[j:].T - (self.q_q[:,j] - self.x_[:,j]*self.l_[:,j]).T @ self.elec_graph[:,j] - self.q_[j]
        #
        # for i in range(self.custom_agent_num):
        #     for j in range(self.custom_agent_num):
        #         self.eq_constraint_v[i,j] = self.v_[i] - 2 * (self.r_[i,j]*self.p_p[i,j] + self.x_[i,j]* self.q_q[i,j]) + \
        #                                     (self.r_[i,j]*self.r_[i,j] + self.x_[i,j]*self.x_[i,j]) \
        #                                     * self.l_[i,j] - self.v_[j]

        # 目标函数
        def objective(x):
            return x[0] ** 2 + x[1] ** 2

        # 线性约束
        def linear_constraint(x):
            return x[0] + x[1] - 1

        # 非线性约束
        def nonlinear_constraint(x):
            # print('x_dtype',type(x))
            p_p = x[:self.custom_agent_num * self.custom_agent_num].reshape(
                (self.custom_agent_num, self.custom_agent_num))
            q_q = x[
                  self.custom_agent_num * self.custom_agent_num:self.custom_agent_num * self.custom_agent_num * 2].reshape(
                self.custom_agent_num, self.custom_agent_num)
            v_ = x[
                 self.custom_agent_num * self.custom_agent_num * 2:self.custom_agent_num * self.custom_agent_num * 2 + self.custom_agent_num].reshape(
                self.custom_agent_num, )
            l_ = x[
                 self.custom_agent_num * self.custom_agent_num * 2 + self.custom_agent_num:self.custom_agent_num * self.custom_agent_num * 2 + self.custom_agent_num + self.custom_agent_num * self.custom_agent_num].reshape(
                self.custom_agent_num, self.custom_agent_num)

            eq_constraint_p = np.zeros((self.custom_agent_num,))
            eq_constraint_q = np.zeros((self.custom_agent_num,))
            eq_constraint_v = np.zeros((self.custom_agent_num, self.custom_agent_num))
            neq_constraint_pqv = np.zeros((self.custom_agent_num, self.custom_agent_num))

            for j in range(self.custom_agent_num):
                eq_constraint_p[j] = p_p[j, :] @ self.elec_graph[j, :].T - (
                        p_p[:, j] - self.r_[:, j] * l_[:, j]).T @ self.elec_graph[:, j] - self.p_[j]
                eq_constraint_q[j] = q_q[j, :] @ self.elec_graph[j, :].T - (
                        q_q[:, j] - self.x_[:, j] * l_[:, j]).T @ self.elec_graph[:, j] - self.q_[j]

            for i in range(self.custom_agent_num):
                for j in range(self.custom_agent_num):
                    eq_constraint_v[i, j] = v_[i] - 2 * (
                            self.r_[i, j] * p_p[i, j] + self.x_[i, j] * q_q[i, j]) + \
                                            (self.r_[i, j] * self.r_[i, j] + self.x_[i, j] * self.x_[i, j]) \
                                            * l_[i, j] - v_[j]
                    neq_constraint_pqv[i, j] = l_[i, j] + v_[i] - np.linalg.norm(
                        [2 * q_q[i, j], 2 * p_p[i, j], l_[i, j] - v_[i]])

            return np.concatenate(
                [eq_constraint_p, eq_constraint_q, eq_constraint_v.reshape(-1), neq_constraint_pqv.reshape(-1)])

        # 定义约束条件
        # linear_cons = {'type': 'ineq', 'fun': linear_constraint}
        nonlinear_cons = {'type': 'eq', 'fun': nonlinear_constraint}

        # 初始猜测
        x0 = np.zeros((
                      self.custom_agent_num * self.custom_agent_num * 2 + self.custom_agent_num + self.custom_agent_num * self.custom_agent_num,))
        # print('x_0',x0.shape)
        # 调用minimize函数
        res = minimize(objective, x0, constraints=[nonlinear_cons])

        return Q, R, L, V


if __name__ == '__main__':
    # csv_data_l = pd.read_excel('./20230201code+data/data_l_33.xlsx')
    # # print(csv_data_l.values)
    # L = csv_data_l.values[:, 1:]
    # # print(L.shape)
    # csv_data_p = pd.read_excel('./20230201code+data/data_p_33.xlsx')
    # P = csv_data_p.values[:,1:]
    # print(P.shape)
    env = MA_ELEC_Market_Graph_ENV(max_step=24, custom_agent_num=118)

    num = 0
    # print(env.is_rooted_tree(adjMatrix=[[0,1,0],[0,0,1],[0,0,0]]))

    env.get_avaliable_action_list()

    # check_env(env)
    # env = wrapped_env(max_step=24, custom_agent_num=33)
    # # api_test(env, num_cycles=10, verbose_progress=False)
    # random_demo(env, render=True, episodes=1)
    env.reset()
    while True:
        num += 1
        actions = []

        for i in range(118):
            action = env.action_space[i].sample()
            action[-1] = -abs(action[-1])
            actions.append(action)

        print('num', num)
        # print('action',env.action_state)
        # print('action_limit0',env.Pes_max)
        # print('action_limit1',env.L_max_ori*0.8)
        # print('action_limit3_high',env.action_3_high_limit)
        # print('action_limit3_low', env.action_3_low_limit)
        observation, reward, done, info = env.step(actions)
        print('reward', reward)

        if num > 23:
            # break
            env.reset()

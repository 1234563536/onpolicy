import numpy as np
from itertools import chain
import pandas as pd
import torch
from onpolicy.runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MarketRunner(Runner):
    def __init__(self, config, total_num_steps=0):
        super(MarketRunner, self).__init__(config)
        self.run_tag = total_num_steps

    def run(self):
        self.warmup()
        print('Begin Run Data Mode...')
        self.run_data()

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        # print('obs', obs.shape)
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)
        # print('share_obs', share_obs.shape)
        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def run_data(self):
        print("Running...")
        eval_episode_rewards = []
        eval_obs = self.run_data_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_episode_actions = []
        eval_episode_trade = []
        for eval_step in range(self.episode_length):
            # for eval_step in range(1):
            # print('episode_length', self.episode_length)
            eval_temp_actions_env = []
            trade_actions = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)
                eval_action = eval_action.detach().cpu().numpy()
                eval_temp_actions_env.append(eval_action)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
            # trade_actions.append(self.eval_envs.action_state)
            eval_actions_env = []
            trade_action = []
            for i in range(self.n_eval_rollout_threads):
                thread_action = []
                for agent_id in range(self.num_agents):
                    thread_action.append(eval_temp_actions_env[agent_id][i, :])
                eval_actions_env.append(thread_action)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)

            eval_episode_rewards.append(eval_rewards)
            eval_episode_actions.append(eval_actions_env)
            eval_episode_trade.append(eval_infos)
            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_episode_actions = np.array(eval_episode_actions).reshape(24, -1)
        eval_episode_trade = np.array(eval_episode_trade).reshape(24, -1)
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'rundata_average_episode_rewards': eval_average_episode_rewards})
            # print("rundata average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        np.save('rundata_episode_actions_' + str(self.run_tag) + '.npy', eval_episode_actions)
        saved_df = pd.DataFrame(eval_episode_trade)
        trade_list = []
        action_list = []
        for row in saved_df.iloc[:, 1]:
            trade_list.append(row)
        for row in saved_df.iloc[:, 0]:
            # print('row',type(row[0]))
            k = []
            for i in row:
                k = k + i
            action_list.append(k)
        pd.DataFrame(trade_list).to_csv('rundata_episode_trade_' + str(self.run_tag) + '.csv')
        # print(np.array(action_list).reshape(24,-1).shape)
        # print(np.array(action_list).reshape(24,-1).tolist())
        pd.DataFrame(np.array(action_list).reshape(24, -1).tolist()).to_csv(
            'rundata_episode_action_state_' + str(self.run_tag) + '.csv')
        print("Finish Run Data Mode!")

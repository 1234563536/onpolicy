import time
# import wandb
import os
import numpy as np
from itertools import chain

import pandas as pd
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class MarketRunner(Runner):
    def __init__(self, config):
        super(MarketRunner, self).__init__(config)

    def run(self):
        self.warmup()
        print('eval_interval', self.eval_interval)
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            print('episode', episode)
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, available_actions = self.collect(
                    step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
                try:
                    self.envs.init_every_day()
                except:
                    pass

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            for count, info in enumerate(infos):
                                if 'individual_reward' in infos[count][agent_id].keys():
                                    idv_rews.append(infos[count][agent_id].get('individual_reward', 0))
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update(
                            {"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                elif self.env_name == "Market":
                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update(
                            {"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # eval

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

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
            # if agent_id == (self.num_agents -1):
            #     self.buffer[agent_id].available_actions[0] = self.envs.get_avaliable_action()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []
        available_actions = None
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            if agent_id != (self.num_agents - 1):
                value, action, action_log_prob, rnn_state, rnn_state_critic \
                    = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                                self.buffer[agent_id].obs[step],
                                                                self.buffer[agent_id].rnn_states[step],
                                                                self.buffer[agent_id].rnn_states_critic[step],
                                                                self.buffer[agent_id].masks[step])
                # available_actions.append(None)
            else:
                # print('id',agent_id)
                # available_action=self.envs.get_avaliable_action()
                value, action, action_log_prob, rnn_state, rnn_state_critic \
                    = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                                self.buffer[agent_id].obs[step],
                                                                self.buffer[agent_id].rnn_states[step],
                                                                self.buffer[agent_id].rnn_states_critic[step],
                                                                self.buffer[agent_id].masks[step],
                                                                # available_actions=available_action
                                                                )
                # available_actions.append(available_action)

                # print('exp',action.shape)
                # print('mask',self.envs.get_avaliable_action().shape)
                # masked_action_prob = np.exp(action_log_prob.detach().cpu().numpy()) * self.envs.get_avaliable_action()
                #
                # masked_action_prob /= np.sum(masked_action_prob,dim=1)
                #
                # masked_action_log_prob = np.log(masked_action_prob)
                #
                # action = torch.distributions.Bernoulli(masked_action_log_prob).sample()

            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action

            actions.append(action)
            # temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        # print('actions',actions)
        actions_env = []
        for i in range(self.n_rollout_threads):
            thread_action = []
            for agent_id in range(self.num_agents):
                thread_action.append(actions[agent_id][i, :])

            actions_env.append(thread_action)

        # print(actions_env)

        values = np.array(values).transpose(1, 0, 2)
        # print('values',values.shape)
        # actions = np.array(actions).transpose(1, 0, 2)
        actions = actions_env
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)
        # print('actions',actions_env)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, available_actions

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            agent_action = []

            for i in range(self.n_rollout_threads):
                agent_action.append(actions[i][agent_id])
            # if agent_id != self.num_agents -1:
            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         agent_action,
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id],
                                         )
            # else:
            #     self.buffer[agent_id].insert(share_obs,
            #                                  np.array(list(obs[:, agent_id])),
            #                                  rnn_states[:, agent_id],
            #                                  rnn_states_critic[:, agent_id],
            #                                  agent_action,
            #                                  action_log_probs[:, agent_id],
            #                                  values[:, agent_id],
            #                                  rewards[:, agent_id],
            #                                  masks[:, agent_id],
            #                                  available_actions=available_actions[-1])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_episode_actions = []
        eval_episode_trade = []
        for eval_step in range(self.episode_length):
            print('episode_length', self.episode_length)
            eval_temp_actions_env = []
            trade_actions = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                # if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                #     for i in range(self.eval_envs.action_space[agent_id].shape):
                #         eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                #             eval_action[:, i]]
                #         if i == 0:
                #             eval_action_env = eval_uc_action_env
                #         else:
                #             eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                # elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                #     eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                # else:
                #     raise NotImplementedError
                #
                # eval_temp_actions_env.append(eval_action_env)

                # rearrange action

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
                # trade_action =

            # [envs, agents, dim]
            # eval_actions_env = []
            # for i in range(self.n_eval_rollout_threads):
            #     eval_one_hot_action_env = []
            #     for eval_temp_action_env in eval_temp_actions_env:
            #         eval_one_hot_action_env.append(eval_temp_action_env[i])
            #     eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

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
        print('eval_spidoe_trade', eval_episode_trade.shape)
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        np.save('eval_episode_actions' + str(total_num_steps) + '.npy', eval_episode_actions)
        # pd.DataFrame(eval_episode_actions).to_csv('eval_episode_actions'+str(total_num_steps))
        saved_df = pd.DataFrame(eval_episode_trade)
        trade_list = []
        action_list = []
        for row in saved_df.iloc[:, 1]:
            print('row', type(row))
            trade_list.append(row)
        for row in saved_df.iloc[:, 0]:
            # print('row',type(row[0]))
            k = []
            for i in row:
                '''
                if type(i) is list:
                    assert all(map(lambda x: x >= 0, sum(i,[]))), "Please Check market_runner !!!"
                '''
                k = k + i
            action_list.append(k)

        pd.DataFrame(trade_list).to_csv('eval_episode_trade' + str(total_num_steps)+'.csv')
        # print(np.array(action_list).reshape(24,-1).shape)
        # print(np.array(action_list).reshape(24,-1).tolist())
        pd.DataFrame(np.array(action_list).reshape(24, -1).tolist()).to_csv(
            'test_eval_episode_action_state' + str(total_num_steps) + '.csv')
        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            # if self.all_args.save_gifs:
            #     image = self.envs.render('rgb_array')[0][0]
            #     all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                actions = []
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                          rnn_states[:, agent_id],
                                                                          masks[:, agent_id],
                                                                          deterministic=True)

                    action = action.detach().cpu().numpy()

                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)

                        # raise NotImplementedError

                    # temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                    actions.append(action)
                # [envs, agents, dim]
                # actions_env = []
                # for i in range(self.n_rollout_threads):
                #     one_hot_action_env = []
                #     for temp_action_env in temp_actions_env:
                #         one_hot_action_env.append(temp_action_env[i])
                #     actions_env.append(one_hot_action_env)

                actions_env = []
                for i in range(self.n_rollout_threads):
                    actions_env.append(np.array(actions)[:, i, :])

                # Obser reward and next obs
                # print('env_step', self.envs.envs[0].num_moves)
                # print('actions', actions_env)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                # if self.all_args.save_gifs:
                #     image = self.envs.render('rgb_array')[0][0]
                #     all_frames.append(image)
                #     calc_end = time.time()
                #     elapsed = calc_end - calc_start
                #     if elapsed < self.all_args.ifi:
                #         time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

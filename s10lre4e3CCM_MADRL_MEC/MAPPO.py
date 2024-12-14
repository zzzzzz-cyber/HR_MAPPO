import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np
import random
import pickle
from copy import deepcopy
from numpy import savetxt
from numpy import loadtxt
from utils import to_tensor_var
from Model_mu_std import *
from replay_buffer1 import ReplayBuffer
from prioritized_memory import Memory
from mec_env import ENV_MODE, K_CHANNEL, S_E, N_UNITS, MAX_STEPS, LAMBDA_E, LAMBDA_T

MSE = nn.MSELoss(reduction='none')
PATH = "./model/MAPPO/"

class MAPPO(object):
    def __init__(self, InfdexofResult, env, env_eval, n_agents, state_dim, action_dim, action_lower_bound, action_higher_bound,
                 target_tau=0.1, reward_gamma=0.9, reward_scale=1., done_penalty=None,
                 actor_output_activation=torch.tanh, actor_lr=1e-4, critic_lr=1e-3,
                 optimizer_type="adam", max_grad_norm=None, batch_size=64, episodes_before_train=64,
                 epsilon_start=1, epsilon_end=0.01, epsilon_decay=None, use_cuda=False):
        self.n_agents = n_agents
        self.env = env
        self.env_eval = env_eval
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lower_bound = action_lower_bound
        self.action_higher_bound = action_higher_bound
        self.memory = Memory(20000)

        #  ========actor========
        # 使用RNN网络
        self.use_VAE = False

        self.use_Actor_RNN = False

        self.use_Actor_RNN_soft1 = False

        self.use_Actor_RNN_soft2 = False
        # 使用soft处理离散动作，tanh处理两个连续动作（action1, action2_mu, action2_std）
        self.use_soft_1_tanh = True
        # 使用soft处理离散动作，tanh处理两个连续动作（action1, action2_mu, action2_std, action3_mu, action3_std）
        self.use_soft_2_tanh = False
        # tanh处理三个连续动作（action_mu, action_std）
        self.use_3_tanh = False
        #  ========actor========


        #  ========critic========

        self.use_value_clip = False

        self.use_critic_RNN = False
        # 使用一个整体critic (1个输入)
        self.use_A_ciritic = False

        self.use_A_ciritic_noise = False

        self.use_S_ciritic_noise = False

        self.use_S_ciritic = True
        #  ========critic========

        self.env_state = env.reset_mec()
        self.n_episodes = 0
        self.preheating_number = 0
        self.n_episodes_batch = 0
        self.roll_out_n_steps = 1
        self.lamda = 0.95
        self.eps = 0.2
        self.actors_epochs = 10
        self.critics_epochs = 10

        self.replay_buffer = ReplayBuffer(self.env)

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.episodes_data = []
        self.actor_output_activation = actor_output_activation
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train

        self.rnn_batch_size = self.batch_size / MAX_STEPS
        self.rnn_batch_size = int(self.rnn_batch_size)

        self.preheating = False
        self.Max_preheating = 1000
        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        if epsilon_decay == None:
            print("epsilon_decay is None")
            exit()
        else:
            self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.target_tau = target_tau

        critic_state_dim = self.n_agents * self.state_dim
        critic_action_dim = self.n_agents * self.action_dim

        #  ====================actor====================
        if self.use_Actor_RNN:
            self.actors = [ActorNetwork_RNN(self.state_dim, self.action_dim, self.actor_output_activation)] * 1
        elif self.use_Actor_RNN_soft1:
            self.actors = [ActorNetwork_soft_RNN1(self.state_dim, self.action_dim, self.actor_output_activation)] * 1
        elif self.use_Actor_RNN_soft2:
            self.actors = [ActorNetwork_soft_RNN2(self.state_dim, self.action_dim, self.actor_output_activation)] * 1
        elif self.use_3_tanh:
            self.actors = [ActorNetwork(self.state_dim, self.action_dim, self.actor_output_activation)] * 1
        elif self.use_soft_1_tanh:
            self.actors = [ActorNetwork_soft(self.state_dim, self.action_dim, self.actor_output_activation)] * 1
        elif self.use_soft_2_tanh:
            self.actors = [ActorNetwork_soft_tanh(self.state_dim, self.action_dim, self.actor_output_activation)] * 1
        #  ====================actor====================
        self.new_actors = deepcopy(self.actors)

        #  ====================critic====================
        self.noise_dim = 3
        if self.use_critic_RNN:
            self.critics = [CriticNetwork_RNN(self.state_dim)] * 1
        elif self.use_A_ciritic:
            self.critics = [CriticNetwork(self.state_dim)] * 1
        elif self.use_A_ciritic_noise:
            self.critics = [CriticNetwork_noise(self.state_dim, self.noise_dim)] * 1
        elif self.use_S_ciritic_noise:
            self.critics = [CriticNetwork_overall_noise(critic_state_dim, self.noise_dim)] * 1
        elif self.use_S_ciritic:
            self.critics = [CriticNetwork_single(critic_state_dim)] * 1
        #  ====================critic====================
        self.new_critics = deepcopy(self.critics)

        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.new_actors]
            self.critics_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.new_critics]
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.new_actors]
            self.critics_optimizer = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.new_critics]

        if self.use_cuda:
            # for i in range(self.n_agents):
            self.actors[0].cuda()
            self.new_actors[0].cuda()
            self.critics[0].cuda()

        self.eval_episode_rewards = []
        self.eval_episode_loss = []

        self.server_episode_constraint_exceeds = []
        self.energy_episode_constraint_exceeds = []
        self.time_episode_constraint_exceeds = []

        self.EC_episode = []
        self.LT_episode = []
        self.EX_episode = []
        self.LX_episode = []

        self.eval_step_rewards = []
        self.eval_step_loss = []
        self.mean_rewards = []
        self.mean_loss = []

        self.episodes = []
        self.Training_episodes = []

        self.Training_episode_rewards = []
        self.Training_step_rewards = []

        self.max_reward = 0

        self.InfdexofResult = InfdexofResult
        #self.save_models('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
        self.results = []
        self.loss = []
        self.Training_results = []
        self.serverconstraints = []
        self.energyconstraints = []
        self.timeconstraints = []

        self.result_EC = []
        self.result_EX = []
        self.result_LT = []
        self.result_LX = []

    def interact(self, MAX_EPISODES, EPISODES_BEFORE_TRAIN, NUMBER_OF_EVAL_EPISODES):
        while self.n_episodes < MAX_EPISODES:
            self.env_state = self.env.reset_mec()
            if self.n_episodes > 0 and self.replay_buffer.episode_num == 0:  # 在交换环境一定次数后开始评估
                self.evaluate(NUMBER_OF_EVAL_EPISODES)
                self.evaluateAtTraining(NUMBER_OF_EVAL_EPISODES)
            self.Data_collection()

            if self.replay_buffer.episode_num == self.batch_size:  # 在交换环境一定次数后开始训练
                self.n_episodes += 1
                print(self.n_episodes)
                self.train()
                self.replay_buffer.reset_buffer()
                #pass
        self.save_models(PATH)

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def Data_collection(self):
        done = False
        if self.use_critic_RNN:
            self.critics[0].rnn_hidden = None
        if self.use_Actor_RNN or self.use_Actor_RNN_soft1 or self.use_Actor_RNN_soft2:
            self.actors[0].rnn_hidden = None
        while not done:  # 每10次作为一个episodes
            state = self.env_state
            actor_action, log_actor_action, critic_action, hybrid_action = self.choose_action(state, False)
            min_max_state = self.env.preprocessing(state)
            next_state, reward, loss_n, original_reward, done, _, _ = self.env.step_mec(hybrid_action, False)
            original_next_state = next_state.copy()
            min_max_next_state = self.env.preprocessing(original_next_state)

            self.Training_step_rewards.append(np.mean(reward))
            if done:
                last_next_state = next_state
                self.Training_episode_rewards.append(np.sum(np.array(self.Training_step_rewards)))
                self.Training_step_rewards = []
                if self.done_penalty is not None:
                    reward = self.done_penalty
                self.n_episodes_batch += 1
            else:
                self.env_state = next_state

            # =====================global=====================
            global_min_max_state = min_max_state.flatten()
            global_min_max_next_state = min_max_next_state.flatten()
            global_reward = (np.ones_like(loss_n) * np.mean(loss_n)).reshape(-1, 1)
            global_action = actor_action.flatten()
            local_reward = reward.reshape(-1, 1)
            # =====================global=====================

            self.replay_buffer.store_transition(min_max_state, global_min_max_state,
                                                local_reward, original_reward, global_reward,
                                                actor_action, critic_action, log_actor_action, global_action, actor_action,
                                                min_max_next_state, global_min_max_next_state, critic_action,
                                                done)
            self.replay_buffer.episode_num += 1

    def train(self):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.n_episodes / self.epsilon_decay)
        noise = np.random.randn(self.batch_size, self.n_agents, self.noise_dim) * epsilon
        noise = to_tensor_var(noise, self.use_cuda)
        batch = self.replay_buffer.get_training_data(self.use_cuda)
        if self.use_critic_RNN:
            loc_current_obs_n = batch['loc_current_obs_n'].reshape(self.rnn_batch_size, MAX_STEPS, self.n_agents, -1)
            loc_obs_next_n = batch['loc_obs_next_n'].reshape(self.rnn_batch_size, MAX_STEPS, self.n_agents, -1)

            value = []
            self.critics[0].rnn_hidden = None
            for vt1 in range(MAX_STEPS):
                tmp_value = self.critics[0](loc_current_obs_n[:, vt1].reshape(self.rnn_batch_size * self.n_agents, -1))
                tmp_value = tmp_value.reshape(self.rnn_batch_size, self.n_agents, -1)
                value.append(tmp_value)
            value = torch.stack(value, dim=1)
            value = value.reshape(self.batch_size, self.n_agents, -1)

            next_value = []
            self.critics[0].rnn_hidden = None
            for vt2 in range(MAX_STEPS):
                tmp_next_value = self.critics[0](loc_obs_next_n[:, vt2].reshape(self.rnn_batch_size * self.n_agents, -1))
                tmp_next_value = tmp_next_value.reshape(self.rnn_batch_size, self.n_agents, -1)
                next_value.append(tmp_next_value)
            next_value = torch.stack(next_value, dim=1)
            next_value = next_value.reshape(self.batch_size, self.n_agents, -1)

            deltas = self.reward_scale * batch['loc_reward_n'] + self.reward_gamma * next_value - value
        elif self.use_A_ciritic:
            value = self.critics[0](batch['loc_current_obs_n'])
            next_value = self.critics[0](batch['loc_obs_next_n'])
            deltas = self.reward_scale * batch['glo_reward_n'] + self.reward_gamma * next_value - value
        elif self.use_A_ciritic_noise:
            value = self.critics[0](batch['loc_current_obs_n'], noise)
            next_value = self.critics[0](batch['loc_obs_next_n'], noise)
            deltas = self.reward_scale * batch['glo_reward_n'] + self.reward_gamma * next_value - value
        elif self.use_S_ciritic_noise:
            glo_state = batch['glo_current_obs_1'].reshape(self.batch_size, 1, -1)
            glo_state = glo_state.repeat(1, self.n_agents, 1)

            next_glo_state = batch['glo_obs_next_1'].reshape(self.batch_size, 1, -1)
            next_glo_state = next_glo_state.repeat(1, self.n_agents, 1)

            value = self.critics[0](glo_state, noise)
            next_value = self.critics[0](next_glo_state, noise)
            deltas = self.reward_scale * batch['glo_reward_n'] + self.reward_gamma * next_value - value
        else:
            glo_state = batch['glo_current_obs_1'].reshape(self.batch_size, 1, -1)
            glo_state = glo_state.repeat(1, self.n_agents, 1)

            next_glo_state = batch['glo_obs_next_1'].reshape(self.batch_size,  1, -1)
            next_glo_state = next_glo_state.repeat(1, self.n_agents, 1)

            value = self.critics[0](glo_state)
            next_value = self.critics[0](next_glo_state)
            deltas = self.reward_scale * batch['glo_reward_n'] + self.reward_gamma * next_value - value

        gae = 0
        advantage_list = []
        deltas = deltas.reshape(-1, MAX_STEPS, self.n_agents, 1)
        for t1 in reversed(range(self.env.steps)):
            gae = deltas[:, t1] + self.reward_gamma * self.lamda * gae
            advantage_list.insert(0, gae)
        advantage = torch.stack(advantage_list, dim=1)
        advantage = advantage.reshape(self.batch_size, self.n_agents, -1)
        v_target = advantage + next_value

        # Trick 1: advantage normalization
        advantage_actor = ((advantage - advantage.mean()) / (advantage.std() + 1e-10)).detach()
        # advantage_actor = advantage.detach()

        for _ in range(self.critics_epochs):
            # critic updates
            if self.use_critic_RNN:
                loc_current_obs_n = batch['loc_current_obs_n'].reshape(self.rnn_batch_size, MAX_STEPS, self.n_agents, -1)
                new_value = []
                self.new_critics[0].rnn_hidden = None
                for t in range(MAX_STEPS):
                    tmp_new_value = self.new_critics[0](loc_current_obs_n[:, t].reshape(self.rnn_batch_size * self.n_agents, -1))
                    tmp_new_value = tmp_new_value.reshape(self.rnn_batch_size, self.n_agents, -1)
                    new_value.append(tmp_new_value)
                new_value = torch.stack(new_value, dim=1)
                new_value = new_value.reshape(self.batch_size, self.n_agents, -1)

            elif self.use_A_ciritic:
                new_value = self.new_critics[0](batch['loc_current_obs_n'])

            elif self.use_A_ciritic_noise:
                new_value = self.new_critics[0](batch['loc_current_obs_n'], noise)

            elif self.use_S_ciritic_noise:
                glo_state = batch['glo_current_obs_1'].reshape(self.batch_size, 1, -1)
                glo_state = glo_state.repeat(1, self.n_agents, 1)
                new_value = self.new_critics[0](glo_state, noise)

            else:
                glo_state = batch['glo_current_obs_1'].reshape(self.batch_size, 1, -1)
                glo_state = glo_state.repeat(1, self.n_agents, 1)
                new_value = self.new_critics[0](glo_state)

            if self.use_value_clip:
                values_error_clip = torch.clamp(new_value - value.detach(), -self.eps, self.eps) + value.detach() - v_target.detach()
                values_error_original = new_value - v_target.detach()
                critic_loss = torch.mean(torch.max(values_error_clip ** 2, values_error_original ** 2))
            else:
                critic_loss = torch.mean(MSE(new_value, v_target.detach()))
            self.critics_optimizer[0].zero_grad()
            critic_loss.backward()
            self.critics_optimizer[0].step()

            self._soft_update_target(self.critics[0], self.new_critics[0])

        for _ in range(self.actors_epochs):
            # actor update
            old_log_prob = batch['loc_current_log_act_n']
            if self.use_3_tanh or self.use_Actor_RNN:
                comb_old_log_prob = old_log_prob.sum(dim=-1, keepdims=True)
            else:
                old_log_prob1 = old_log_prob[:, :, 0].reshape(self.batch_size, self.n_agents, -1)
                old_log_prob2 = old_log_prob[:, :, 1:]
                comb_old_log_prob = old_log_prob1 + old_log_prob2

            if self.use_soft_1_tanh:
                # 每一轮更新一次策略网络预测的状态
                new_action1, new_action2_mu, new_action2_std = self.new_actors[0](batch['loc_current_obs_n'])
                new_policy_dist1 = torch.distributions.Categorical(new_action1)
                new_log_prob1 = new_policy_dist1.log_prob(batch['loc_current_act_n'][:, :, 0]).reshape(self.episodes_before_train, self.n_agents, -1)

                new_policy_dist2 = torch.distributions.Normal(new_action2_mu, new_action2_std)
                new_log_prob2 = new_policy_dist2.log_prob(batch['loc_current_act_n'][:, :, 1:])

                comb_new_log_prob = new_log_prob1 + new_log_prob2
                ratio1 = (comb_new_log_prob - comb_old_log_prob).exp()

                # 近端策略优化裁剪目标函数公式的左侧项
                surr11 = ratio1 * advantage_actor
                surr12 = torch.clamp(ratio1, 1-self.eps, 1+self.eps) * advantage_actor
                actor_loss = - torch.mean(torch.min(surr11, surr12))

            elif self.use_Actor_RNN_soft1:
                self.new_actors[0].rnn_hidden = None
                comb_new_log_prob = []
                for t2 in range(MAX_STEPS):
                    new_action1, new_action2_mu, new_action2_std = self.new_actors[0](batch['loc_current_obs_n'][:, t2].reshape(self.batch_size * self.n_agents, -1))

                    new_action1 = new_action1.reshape(self.batch_size, self.n_agents, -1)
                    new_policy_dist1 = torch.distributions.Categorical(new_action1)
                    new_log_prob1 = new_policy_dist1.log_prob(batch['loc_current_act_n'][:, t2, :, 0]).reshape(self.batch_size, self.n_agents, -1)

                    new_action2_mu = new_action2_mu.reshape(self.batch_size, self.n_agents, -1)
                    new_action2_std = new_action2_std.reshape(self.batch_size, self.n_agents, -1)
                    new_policy_dist = torch.distributions.Normal(new_action2_mu, new_action2_std)
                    new_log_prob2 = new_policy_dist.log_prob(batch['loc_current_act_n'][:, t2, :, 1:])

                    tmp_comb_new_log_prob = new_log_prob1 + new_log_prob2
                    comb_new_log_prob.append(tmp_comb_new_log_prob)
                comb_new_log_prob = torch.stack(comb_new_log_prob, dim=1)

                ratio = (comb_new_log_prob - comb_old_log_prob).exp()
                # 近端策略优化裁剪目标函数公式的左侧项
                surr1 = ratio * advantage_actor
                surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage_actor
                # 策略网络的损失函数
                actor_loss = - torch.mean(torch.min(surr1, surr2))

            elif self.use_Actor_RNN_soft2:
                self.new_actors[0].rnn_hidden = None
                comb_new_log_prob = []
                for t2 in range(MAX_STEPS):
                    new_action1, new_action2_mu, new_action2_std, new_action3_mu, new_action3_std = self.new_actors[0](batch['loc_current_obs_n'][:, t2].reshape(self.batch_size * self.n_agents, -1))

                    new_action1 = new_action1.reshape(self.batch_size, self.n_agents, -1)
                    new_policy_dist1 = torch.distributions.Categorical(new_action1)
                    new_log_prob1 = new_policy_dist1.log_prob(batch['loc_current_act_n'][:, t2, :, 0]).reshape(self.batch_size, self.n_agents, -1)

                    new_action2_mu = new_action2_mu.reshape(self.batch_size, self.n_agents, -1)
                    new_action2_std = new_action2_std.reshape(self.batch_size, self.n_agents, -1)
                    new_policy_dist2 = torch.distributions.Normal(new_action2_mu, new_action2_std)
                    new_log_prob2 = new_policy_dist2.log_prob(batch['loc_current_act_n'][:, t2, :, 1:].reshape(self.batch_size, self.n_agents, -1))

                    tmp_comb_new_log_prob = new_log_prob1 + new_log_prob2
                    comb_new_log_prob.append(tmp_comb_new_log_prob)
                comb_new_log_prob = torch.stack(comb_new_log_prob, dim=1)

                ratio = (comb_new_log_prob - comb_old_log_prob).exp()
                # 近端策略优化裁剪目标函数公式的左侧项
                surr1 = ratio * advantage_actor
                surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage_actor
                # 策略网络的损失函数
                actor_loss = - torch.mean(torch.min(surr1, surr2))

            elif self.use_soft_2_tanh:
                new_action1, new_action2_mu, new_action2_std, new_action3_mu, new_action3_std = self.new_actors[0](batch['loc_current_obs_n'])
                new_policy_dist1 = torch.distributions.Categorical(new_action1)
                new_log_prob1 = new_policy_dist1.log_prob(batch['loc_current_act_n'][:, :, 0]).reshape(self.batch_size, self.n_agents, -1)

                new_policy_dist2 = torch.distributions.Normal(new_action2_mu, new_action2_std)
                new_log_prob2 = new_policy_dist2.log_prob(batch['loc_current_act_n'][:, :, 1:].reshape(self.batch_size, self.n_agents, -1))

                comb_new_log_prob = new_log_prob1 + new_log_prob2
                ratio1 = (comb_new_log_prob - comb_old_log_prob).exp()
                # 近端策略优化裁剪目标函数公式的左侧项
                surr1 = ratio1 * advantage_actor
                surr2 = torch.clamp(ratio1, 1-self.eps, 1+self.eps) * advantage_actor
                actor_loss = - torch.mean(torch.min(surr1, surr2))


            elif self.use_Actor_RNN:
                self.new_actors[0].rnn_hidden = None
                comb_new_log_prob = []
                for t2 in range(MAX_STEPS):
                    new_action_mu, new_action_std = self.new_actors[0](batch['loc_current_obs_n'][:, t2].reshape(self.batch_size * self.n_agents, -1))
                    new_action_mu = new_action_mu.reshape(self.batch_size, self.n_agents, -1)
                    new_action_std = new_action_std.reshape(self.batch_size, self.n_agents, -1)

                    new_policy_dist = torch.distributions.Normal(new_action_mu, new_action_std)
                    new_log_prob = new_policy_dist.log_prob(batch['loc_current_act_n'][:, t2])

                    tmp_comb_new_log_prob = new_log_prob.sum(dim=-1, keepdims=True)
                    comb_new_log_prob.append(tmp_comb_new_log_prob)
                comb_new_log_prob = torch.stack(comb_new_log_prob, dim=1)

                comb_old_log_prob = old_log_prob.sum(dim=-1, keepdims=True)
                ratio = (comb_new_log_prob - comb_old_log_prob).exp()
                # 近端策略优化裁剪目标函数公式的左侧项
                surr1 = ratio * advantage_actor
                surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage_actor
                # 策略网络的损失函数
                actor_loss = - torch.mean(torch.min(surr1, surr2))

            else:
                new_action_mu, new_action_std = self.new_actors[0](batch['loc_current_obs_n'])
                new_policy_dist = torch.distributions.Normal(new_action_mu, new_action_std)

                new_log_prob = new_policy_dist.log_prob(batch['loc_current_act_n'])
                comb_new_log_prob = new_log_prob.sum(dim=-1, keepdims=True)

                comb_old_log_prob = old_log_prob.sum(dim=-1, keepdims=True)
                ratio = (comb_new_log_prob - comb_old_log_prob).exp()
                # 近端策略优化裁剪目标函数公式的左侧项
                surr1 = ratio * advantage_actor
                surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage_actor
                # 策略网络的损失函数
                actor_loss = - torch.mean(torch.min(surr1, surr2))

            # 梯度清0
            self.actors_optimizer[0].zero_grad()
            # 反向传播
            actor_loss.backward()
            # 梯度更新
            self.actors_optimizer[0].step()

            # self.actors[0].load_state_dict(self.new_actors[0].state_dict())
            self._soft_update_target(self.actors[0], self.new_actors[0])

            self.lr_decay(self.n_episodes)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.actor_lr * (1 - total_steps / self.epsilon_decay)
        for p in self.actors_optimizer[0].param_groups:
            p['lr'] = lr_now

        '''
        checkpoint = torch.load('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')    
        # Check for parameter differences in actors
        changes = []
        for agent_id in range(self.n_agents):
            ce = self.check_parameter_difference(self.actors[agent_id], checkpoint['actors'][agent_id])
            changes.append(ce)
        # Check for parameter differences in critics
        for agent_id in range(1):
            ce = self.check_parameter_difference(self.critics[agent_id], checkpoint['critics'][agent_id])
            changes.append(ce)
        if sum(changes) >1:
            #print("Model update detected", changes)
            self.save_models('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
        elif sum(changes) == 1:
            print("No actor model update detected", changes)
            self.save_models('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
            #exit()
        else:
            print("No model update detected", changes)
            self.save_models('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
            #exit()
        '''
    def save_models(self, path):
        torch.save(self.actors[0].state_dict(), path + str(self.InfdexofResult) + '-Actor.pth')

    def load_models(self, path):
        self.actors[0].load_state_dict(torch.load(path + str(self.InfdexofResult) + "-Actor.pth"))

    def save_buffer(self, path):
        torch.save(self.memory.buffer, path + '_vae_buffer.pickle')

    def check_parameter_difference(self, model, loaded_state_dict):
        current_state_dict = model.state_dict()
        for name, param in current_state_dict.items():
            if name in loaded_state_dict:
                if not torch.equal(param, loaded_state_dict[name]):
                    #print(f"Parameter '{name}' has changed since the last checkpoint.")
                    return 1
                else:
                    #print(f"Parameter '{name}' has not changed since the last checkpoint.")
                    return 0
            else:
                print(f"Parameter '{name}' is not present in the loaded checkpoint.")
                exit()

    def getactionbound(self, a, b, x, i):
        x = (x - a) * (self.action_higher_bound[i] - self.action_lower_bound[i]) / (b - a) \
            + self.action_lower_bound[i]
        return x

    # choose an action based on state with random noise added for exploration in training
    def choose_action(self, original_state, evaluation):
        '''
        checkpoint = torch.load('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
        for agent_id in range(self.n_agents):
            self.actors[agent_id].load_state_dict(checkpoint['actors'][agent_id])
            self.actors_target[agent_id].load_state_dict(checkpoint['actors_target'][agent_id])
            if agent_id == 0:
                self.critics[agent_id].load_state_dict(checkpoint['critics'][agent_id])
                self.critics_target[agent_id].load_state_dict(checkpoint['critics_target'][agent_id])
        '''
        state = original_state.copy()
        state_var = self.env.preprocessing(state)
        state_var = to_tensor_var(state_var, self.use_cuda)
        # get actor_action
        actor_action = np.zeros((self.n_agents, self.action_dim))
        log_actor_action = np.zeros((self.n_agents, self.action_dim))
        log_actor_action_discrete = np.zeros((self.n_agents, 2))
        critic_action = np.zeros((self.n_agents))
        hybrid_action = np.zeros((self.n_agents, self.action_dim))

        # if not evaluation:
        #     epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        #               np.exp(-1. * self.n_episodes / self.epsilon_decay)


        if self.use_soft_1_tanh or self.use_Actor_RNN_soft1:
            action_var1, action_var2, action_std2 = self.actors[0](state_var)
            policy_dist1 = torch.distributions.Categorical(action_var1)
            action_var1 = policy_dist1.sample().reshape(-1, 1)

            policy_dist2 = torch.distributions.Normal(action_var2, action_std2)
            action_var2 = policy_dist2.sample()

            action_var2 = torch.clamp(action_var2, -1, 1)
            log_action_tanh = policy_dist2.log_prob(action_var2)

            action_var = torch.cat([action_var1, action_var2], -1)
            if self.use_cuda:
                actor_action = action_var.data.cpu().numpy()
                log_actor_action_discrete = policy_dist1.logits.data.cpu().numpy()
                log_actor_action[:, 1:] = log_action_tanh.data.cpu().numpy()
            else:
                actor_action = action_var.data.numpy()
                log_actor_action_discrete = policy_dist1.logits.data.numpy()
                log_actor_action[:, 1:] = log_action_tanh.data.numpy()

        elif self.use_soft_2_tanh or self.use_Actor_RNN_soft2:
            action_var1, action_mu2, action_std2, action_mu3, action_std3 = self.actors[0](state_var)
            policy_dist1 = torch.distributions.Categorical(action_var1)
            action_var1 = policy_dist1.sample().reshape(-1, 1)

            policy_dist2 = torch.distributions.Normal(action_mu2, action_std2)
            action_var2 = policy_dist2.sample()
            action_var2 = torch.clamp(action_var2, -1, 1)
            log_action_tanh2 = policy_dist2.log_prob(action_var2)

            policy_dist3 = torch.distributions.Normal(action_mu3, action_std3)
            action_var3 = policy_dist3.sample()
            action_var3 = torch.clamp(action_var3, -1, 1)
            log_action_tanh3 = policy_dist3.log_prob(action_var3)

            log_action_tanh = torch.cat([log_action_tanh2, log_action_tanh3], -1)
            action_var = torch.cat([action_var1, action_var2, action_var3], -1)

            if self.use_cuda:
                actor_action = action_var.data.cpu().numpy()
                log_actor_action_discrete = policy_dist1.logits.data.cpu().numpy()
                log_actor_action[:, 1:] = log_action_tanh.data.cpu().numpy()
            else:
                actor_action = action_var.data.numpy()
                log_actor_action_discrete = policy_dist1.logits.data.numpy()
                log_actor_action[:, 1:] = log_action_tanh.data.numpy()
        else:
            action_var, action_std = self.actors[0](state_var)
            policy_dist = torch.distributions.Normal(action_var, action_std)
            action_var = policy_dist.sample()
            action_var = torch.clamp(action_var, -1, 1)
            log_action_tanh = policy_dist.log_prob(action_var)

            if self.use_cuda:
                actor_action = action_var.data.cpu().numpy()
                log_actor_action = log_action_tanh.data.cpu().numpy()
            else:
                actor_action = action_var.data.numpy()
                log_actor_action = log_action_tanh.data.numpy()

        hybrid_action = deepcopy(actor_action)
        # first check if ther is at least one actor that chose to offload
        proposed = np.count_nonzero(actor_action[:, 0] > 0)
        proposed_indices = np.where(actor_action[:, 0] > 0)[0]
        sumofproposed = np.sum(original_state[proposed_indices, 3])
        # print(proposed, proposed_indices, sumofproposed)
        if ENV_MODE == "H2":
            constraint = K_CHANNEL
        elif ENV_MODE == "TOBM":
            constraint = N_UNITS
        else:
            print("Unknown env_mode ", ENV_MODE)
            exit()
        if proposed > 0:  # find their Q-values
            if proposed > constraint or sumofproposed > S_E:  # if the number of agents proposed to offload is greater than the number of available channels
                action_Qs = np.zeros((self.n_agents))
                action_Qs.fill(-np.inf)
                for agentid in range(self.n_agents):
                    if actor_action[agentid, 0] > 0:
                        #print("336",whole_states_var.shape, whole_actions_var.shape, states_var[0,agentid,:].view(1, -1).shape, actor_action_var[0, agentid, :].view(1, -1).shape)
                        action_Qs[agentid] = actor_action[agentid, 2]
                # sort in q values in decending and select using k as constraint
                sorted_indices = np.argsort(action_Qs)[::-1]
                # now select tasks
                countaccepted = 0
                sizeaccepted = 0
                for agentid in range(self.n_agents):
                    if actor_action[sorted_indices[agentid], 0] > 0 and countaccepted < constraint and sizeaccepted + original_state[sorted_indices[agentid], 3] < S_E:
                        critic_action[sorted_indices[agentid]] = 1
                        countaccepted += 1
                        sizeaccepted += original_state[sorted_indices[agentid], 3]
            else:  # if the proposed tasks are less than the cosntraints
                for agentid in range(self.n_agents):
                    if hybrid_action[agentid, 0] > 0:
                        critic_action[agentid] = 1
                    else:
                        critic_action[agentid] = 0
        hybrid_action[:, 0] = critic_action
        # ============find_log_action============
        if self.use_soft_1_tanh or self.use_soft_2_tanh or self.use_Actor_RNN_soft1 or self.use_Actor_RNN_soft2:
            actor_action[:, 0] = critic_action
            arg = np.arange(0, self.n_agents)
            index = np.array(critic_action, dtype=np.int64)
            log_actor_action[:, 0] = log_actor_action_discrete[arg, index]
        # ============find_log_action============

        if self.use_VAE and self.preheating:
            discrete_action = torch.LongTensor(actor_action[:, 0])
            discrete_action = self.VAE.embedding(discrete_action)
            z = to_tensor_var(actor_action[:, 1:], self.use_cuda)
            decode_continue, _ = self.VAE.decode(discrete_action, z, state_var)
            if self.use_cuda:
                hybrid_action[:, 1:] = decode_continue.data.cpu().numpy()
            else:
                hybrid_action[:, 1:] = decode_continue.data.numpy()

        b = 1
        a = -b
        for n in range(self.n_agents):
            hybrid_action[n][1] = self.getactionbound(a, b, hybrid_action[n][1], 1)
            hybrid_action[n][2] = self.getactionbound(a, b, hybrid_action[n][2], 2)
        return actor_action, log_actor_action, critic_action, hybrid_action




    def evaluate(self, EVAL_EPISODES):
        if ENV_MODE == "H2":
            constraint = K_CHANNEL
        elif ENV_MODE == "TOBM":
            constraint = N_UNITS
        else:
            print("Unknown env_mode ", ENV_MODE)
            exit()
        for i in range(EVAL_EPISODES):
            self.eval_env_state = self.env_eval.reset_mec(i)
            self.eval_step_rewards = []
            self.eval_step_loss = []
            self.server_step_constraint_exceeds = 0
            self.energy_step_constraint_exceeds = 0
            self.time_step_constraint_exceeds = 0
            self.EC_step = 0
            self.LT_step = 0
            self.EX_step = 0
            self.LX_step = 0

            done = False
            if self.use_Actor_RNN or self.use_Actor_RNN_soft1 or self.use_Actor_RNN_soft2:
                self.actors[0].rnn_hidden = None
            if self.use_critic_RNN:
                self.critics[0].rnn_hidden = None
            while not done:
                state = self.eval_env_state
                # print("state", state)
                actor_action, _, _, hybrid_action = self.choose_action(state, True)
                proposed = np.count_nonzero(actor_action[:, 0] >= 0)
                proposed_indices = np.where(actor_action[:, 0] >= 0)[0]
                sumofproposed = np.sum(state[proposed_indices, 3])
                next_state, reward, reward_n, original_reward, done, eneryconstraint_exceeds, timeconstraint_exceeds = self.env_eval.step_mec(hybrid_action, True)
                self.eval_step_rewards.append(np.mean(reward))
                loss = np.ones_like(reward_n) * np.sum(reward_n) / self.n_agents
                self.eval_step_loss.append(np.mean(loss))
                self.energy_step_constraint_exceeds += eneryconstraint_exceeds
                self.time_step_constraint_exceeds += timeconstraint_exceeds

                # ==================EC, LT, EX, LX==================
                self.EC_step += np.sum(original_reward[0])
                self.EX_step += np.sum(original_reward[1])
                self.LT_step += np.sum(original_reward[2])
                self.LX_step += np.sum(original_reward[3])
                # ==================EC, LT, EX, LX==================
                if proposed > constraint or sumofproposed > S_E:  # if constraint exceeded count it
                    self.server_step_constraint_exceeds += 1
                # print(actor_action)
                if done:
                    self.eval_episode_rewards.append(np.sum(np.array(self.eval_step_rewards)))
                    self.eval_episode_loss.append(np.sum(np.array(self.eval_step_loss)))
                    self.server_episode_constraint_exceeds.append(self.server_step_constraint_exceeds/len(self.eval_step_rewards))
                    self.energy_episode_constraint_exceeds.append(self.energy_step_constraint_exceeds/len(self.eval_step_rewards))
                    self.time_episode_constraint_exceeds.append(self.time_step_constraint_exceeds/len(self.eval_step_rewards))
                    self.EC_episode.append(self.EC_step / len(self.eval_step_rewards))
                    self.LT_episode.append(self.EX_step / len(self.eval_step_rewards))
                    self.EX_episode.append(self.LT_step / len(self.eval_step_rewards))
                    self.LX_episode.append(self.LX_step / len(self.eval_step_rewards))

                    self.eval_step_rewards = []
                    self.server_step_constraint_exceeds = 0
                    self.energy_step_constraint_exceeds = 0
                    self.time_step_constraint_exceeds = 0
                    self.EC_step = 0
                    self.LT_step = 0
                    self.EX_step = 0
                    self.LX_step = 0

                    if self.done_penalty is not None:
                        reward = self.done_penalty
                else:
                    self.eval_env_state = next_state
            if i == EVAL_EPISODES-1 and done:
                # print(self.eval_episode_rewards)
                mean_reward = np.mean(np.array(self.eval_episode_rewards))
                mean_loss = np.mean(np.array(self.eval_episode_loss))
                mean_constraint = np.mean(np.array(self.server_episode_constraint_exceeds))
                mean_energyconstraint = np.mean(np.array(self.energy_episode_constraint_exceeds))
                mean_timeconstraint = np.mean(np.array(self.time_episode_constraint_exceeds))
                mean_EC = np.mean(np.array(self.EC_episode))
                mean_EX = np.mean(np.array(self.LT_episode))
                mean_LT = np.mean(np.array(self.EX_episode))
                mean_LX = np.mean(np.array(self.LX_episode))

                self.eval_episode_rewards = []
                self.eval_episode_loss = []
                self.server_episode_constraint_exceeds = []
                self.energy_episode_constraint_exceeds = []
                self.time_episode_constraint_exceeds = []
                self.EC_episode = []
                self.LT_episode = []
                self.EX_episode = []
                self.LX_episode = []

                self.mean_rewards.append(mean_reward)  # to be plotted by the main function
                self.mean_loss.append(mean_loss)
                self.episodes.append(self.n_episodes+1)
                self.results.append(mean_reward)
                self.loss.append(mean_loss)
                self.serverconstraints.append(mean_constraint)
                self.energyconstraints.append(mean_energyconstraint)
                self.timeconstraints.append(mean_timeconstraint)
                self.result_EC.append(mean_EC)
                self.result_EX.append(mean_EX)
                self.result_LT.append(mean_LT)
                self.result_LX.append(mean_LX)

                arrayresults = np.array(self.results)
                arrayloss = np.array(self.loss)
                arrayserver = np.array(self.serverconstraints)
                arrayenergy = np.array(self.energyconstraints)
                arraytime = np.array(self.timeconstraints)

                result_EC = np.array(self.result_EC)
                result_EX = np.array(self.result_EX)
                result_LT = np.array(self.result_LT)
                result_LX = np.array(self.result_LX)

                savetxt('./CSV/results/MAPPO_'+str(self.InfdexofResult)+'.csv', arrayresults)
                savetxt('./CSV/loss/MAPPO_'+str(self.InfdexofResult)+'.csv', arrayloss)
                savetxt('./CSV/Server_constraints/MAPPO_'+str(self.InfdexofResult)+'.csv', arrayserver)
                savetxt('./CSV/Energy_constraints/MAPPO_'+str(self.InfdexofResult)+'.csv', arrayenergy)
                savetxt('./CSV/Time_constraints/MAPPO_' + str(self.InfdexofResult)+'.csv', arraytime)

                savetxt('./CSV/result_EC/MAPPO_'+str(self.InfdexofResult)+'.csv', result_EC)
                savetxt('./CSV/result_EX/MAPPO_'+str(self.InfdexofResult)+'.csv', result_EX)
                savetxt('./CSV/result_LT/MAPPO_'+str(self.InfdexofResult)+'.csv', result_LT)
                savetxt('./CSV/result_LX/MAPPO_' + str(self.InfdexofResult)+'.csv', result_LX)

                # if self.env.record_min_max:
                #     loss_address = "./CSV/min_max/loss_min_max.pickle"
                #     file = open(loss_address, 'wb')
                #     pickle.dump(self.env.loss_min_max, file)
                # print("Episode:", self.n_episodes, "Episodic Energy:  Min mean Max : ", np.min(arrayenergy), mean_energyconstraint, np.max(arrayenergy))
    def evaluateAtTraining(self, EVAL_EPISODES):
        # print(self.eval_episode_rewards)
        mean_reward = np.mean(np.array(self.Training_episode_rewards))
        self.Training_episode_rewards = []
        # self.mean_rewards.append(mean_reward) # to be plotted by the main function
        self.Training_episodes.append(self.n_episodes+1)
        self.Training_results.append(mean_reward)
        arrayresults = np.array(self.Training_results)
        savetxt('./CSV/AtTraining/MAPPO_' + self.InfdexofResult+'.csv', arrayresults)
        # print("Episode:", self.n_episodes, "Episodic Reward:  Min mean Max : ", np.min(arrayresults), mean_reward, np.max(arrayresults))

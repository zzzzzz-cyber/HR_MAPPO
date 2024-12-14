import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np
import math
import pickle
from copy import deepcopy
from numpy import savetxt
from numpy import loadtxt
from utils import to_tensor_var
from Model_mu_std import *
from prioritized_memory import Memory
from mec_env import ENV_MODE, K_CHANNEL, S_E, N_UNITS, MAX_STEPS, LAMBDA_E, LAMBDA_T


MSE = nn.MSELoss(reduction='none')
PATH = "./model/MATD3/"

class POTD3(object):
    def __init__(self, InfdexofResult, env, env_eval, n_agents, state_dim, action_dim, action_lower_bound, action_higher_bound,
                 memory_capacity=10000, target_tau=0.2, reward_gamma=0.9, reward_scale=1., done_penalty=None,
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

        #  ========actor========
        # 使用RNN网络
        self.use_VAE = False
        self.use_tree = False

        self.env_state = env.reset_mec()
        self.n_episodes = 0
        self.n_episodes_batch = 0
        self.roll_out_n_steps = 1
        self.lamda = 0.95
        self.eps = 0.2
        self.epochs = 10

        self.memory = Memory(memory_capacity)

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

        self.preheating = False
        self.Max_preheating = 200
        self.preheating_number = 0

        critic_state_dim = self.n_agents * self.state_dim
        critic_action_dim = self.n_agents * self.action_dim

        #  ====================actor====================
        self.actors = [ActorNetwork_TD3(self.state_dim, self.action_dim, self.actor_output_activation)] * 1
        self.target_actors = deepcopy(self.actors)
        #  ====================actor====================

        #  ====================critic====================
        self.critics1 = [Q_CriticNetwork(critic_state_dim, critic_action_dim)] * 1
        self.critics2 = [Q_CriticNetwork(critic_state_dim, critic_action_dim)] * 1
        self.target_critics1 = deepcopy(self.critics1)
        self.target_critics2 = deepcopy(self.critics2)
        #  ====================critic====================


        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics1_optimizer = [Adam(b.parameters(), lr=self.critic_lr) for b in self.critics1]
            self.critics2_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics2]
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics1_optimizer = [RMSprop(b.parameters(), lr=self.critic_lr) for b in self.critics1]
            self.critics2_optimizer = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics2]

        if self.use_cuda:
            self.actors[0].cuda()
            self.target_actors[0].cuda()
            self.critics1[0].cuda()
            self.target_critics1[0].cuda()
            self.critics2[0].cuda()
            self.target_critics2[0].cuda()

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
            if self.n_episodes > self.batch_size:  # 在交换环境一定次数后开始评估
                self.evaluate(NUMBER_OF_EVAL_EPISODES)
                self.evaluateAtTraining(NUMBER_OF_EVAL_EPISODES)
            self.Data_collection()
            self.n_episodes += 1
            print(self.n_episodes)
            if self.n_episodes >= EPISODES_BEFORE_TRAIN:  # 在交换环境一定次数后开始训练
                self.train()
                #pass
        self.save_models(PATH)
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)
    def append_rand_sample(self, min_max_state, actor_action, critic_action, global_reward, min_max_next_state, done):
        self.memory.rand_add2(min_max_state, actor_action, critic_action, global_reward, min_max_next_state, done)

    def Data_collection(self):
        done = False
        while not done:  # 每10次作为一个episodes
            state = self.env_state
            actor_action, critic_action, hybrid_action = self.choose_action(state, False)
            min_max_state = self.env.preprocessing(state)
            next_state, reward, reward_n, original_reward, done, _, _ = self.env.step_mec(hybrid_action, False)
            original_next_state = next_state.copy()
            min_max_next_state = self.env.preprocessing(original_next_state)

            self.Training_step_rewards.append(np.mean(reward))
            if done:
                self.Training_episode_rewards.append(np.sum(np.array(self.Training_step_rewards)))
                self.Training_step_rewards = []
                if self.done_penalty is not None:
                    reward = self.done_penalty
            else:
                self.env_state = next_state
            # =====================global=====================
            global_reward = np.ones_like(reward_n) * np.sum(reward_n) / self.n_agents
            # =====================global=====================

            self.append_rand_sample(min_max_state, actor_action, critic_action, global_reward, min_max_next_state, done)

    def train(self):
        buffer_data = self.memory.rand_sample(self.batch_size)
        state, actor_action, critic_action, reward, next_state, done = zip(*buffer_data)
        states = np.vstack(state)
        actor_actions = np.vstack(actor_action)
        critic_actions = np.vstack(critic_action)
        rewards = np.vstack(reward)
        next_states = np.vstack(next_state)
        dones = np.vstack(done)

        dones = dones.astype(int)
        states_var = to_tensor_var(states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        whole_states_var = states_var.view(self.batch_size, -1)
        actor_actions_var = to_tensor_var(actor_actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)

        whole_actor_actions_var = actor_actions_var.view(self.batch_size, -1)
        rewards_var = to_tensor_var(rewards, self.use_cuda).view(-1, self.n_agents, 1)
        next_states_var = to_tensor_var(next_states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        whole_next_states_var = next_states_var.view(self.batch_size, -1)
        dones_var = to_tensor_var(dones, self.use_cuda).view(-1, 1)

        # Critic Update
        next_action_var1, next_action_var2 = self.target_actors[0](next_states_var)
        next_action_var = torch.cat([next_action_var1, next_action_var2], -1)
        # action_noise = torch.tensor(np.random.normal(loc=0.0, scale=0.2), dtype=torch.float)
        # action_noise = torch.clamp(action_noise, -0.5, 0.5)
        # next_action_var = torch.clamp(next_action_var, -1, 1)
        whole_next_action_var = next_action_var.view(self.batch_size, -1)

        q_value1 = self.target_critics1[0](whole_next_states_var, whole_next_action_var)
        q_value2 = self.target_critics2[0](whole_next_states_var, whole_next_action_var)
        next_value = torch.min(q_value1, q_value2)
        target_value = self.reward_scale * rewards_var[:, 0, :] + self.reward_gamma * next_value * (1 - dones_var)

        current_value1 = self.critics1[0](whole_states_var, whole_actor_actions_var)
        current_value2 = self.critics2[0](whole_states_var, whole_actor_actions_var)
        critic_1_loss = torch.mean(MSE(current_value1, target_value.detach()))
        critic_2_loss = torch.mean(MSE(current_value2, target_value.detach()))
        self.critics1_optimizer[0].zero_grad()
        critic_1_loss.backward()
        self.critics1_optimizer[0].step()
        self.critics2_optimizer[0].zero_grad()
        critic_2_loss.backward()
        self.critics2_optimizer[0].step()

        new_action_var1, new_action_var2 = self.actors[0](states_var)
        new_action_var = torch.cat([new_action_var1, new_action_var2], -1)
        whole_new_action_var = new_action_var.view(self.batch_size, -1)
        new_value1 = self.critics1[0](whole_states_var, whole_new_action_var)
        new_value2 = self.critics2[0](whole_states_var, whole_new_action_var)

        actor_loss = - torch.mean(torch.min(new_value1, new_value2))

        self.actors_optimizer[0].zero_grad()
        actor_loss.backward()
        self.actors_optimizer[0].step()

        # if self.n_episodes % 3 == 0:
        # self._soft_update_target(self.target_actors[0], self.actors[0])
        self.target_actors[0].load_state_dict(self.actors[0].state_dict())

        self._soft_update_target(self.target_critics1[0], self.critics1[0])
        self._soft_update_target(self.target_critics2[0], self.critics2[0])

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
        actor_action = np.zeros((self.n_agents, self.action_dim))
        log_actor_action = np.zeros((self.n_agents, self.action_dim))
        log_actor_action_discrete = np.zeros((self.n_agents, 2))
        critic_action = np.zeros((self.n_agents))
        hybrid_action = np.zeros((self.n_agents, self.action_dim))

        action_var1, action_var2 = self.actors[0](state_var)
        action_var = torch.cat([action_var1, action_var2], -1)
        if self.use_cuda:
            actor_action = action_var.data.cpu().numpy()
        else:
            actor_action = action_var.data.numpy()

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
        actor_action[:, 0] = critic_action

        b = 1
        a = -b
        for n in range(self.n_agents):
            hybrid_action[n][1] = self.getactionbound(a, b, hybrid_action[n][1], 1)
            hybrid_action[n][2] = self.getactionbound(a, b, hybrid_action[n][2], 2)
        return actor_action, critic_action, hybrid_action

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
            while not done:
                state = self.eval_env_state
                # print("state", state)
                actor_action, _,  hybrid_action = self.choose_action(state, True)
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
                    # the self.eval_step_rewards is used to deduce the step size
                    # print("eval reward and constraint", np.sum(np.array(self.eval_step_rewards)), self.server_step_constraint_exceeds)
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

                savetxt('./CSV/results/POTD3_'+str(self.InfdexofResult)+'.csv', arrayresults)
                savetxt('./CSV/loss/POTD3_'+str(self.InfdexofResult)+'.csv', arrayloss)
                savetxt('./CSV/Server_constraints/POTD3_'+str(self.InfdexofResult)+'.csv', arrayserver)
                savetxt('./CSV/Energy_constraints/POTD3_'+str(self.InfdexofResult)+'.csv', arrayenergy)
                savetxt('./CSV/Time_constraints/POTD3_' + str(self.InfdexofResult)+'.csv', arraytime)

                savetxt('./CSV/result_EC/POTD3_'+str(self.InfdexofResult)+'.csv', result_EC)
                savetxt('./CSV/result_EX/POTD3_'+str(self.InfdexofResult)+'.csv', result_EX)
                savetxt('./CSV/result_LT/POTD3_'+str(self.InfdexofResult)+'.csv', result_LT)
                savetxt('./CSV/result_LX/POTD3_' + str(self.InfdexofResult)+'.csv', result_LX)

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
        savetxt('./CSV/AtTraining/POTD3_' + self.InfdexofResult+'.csv', arrayresults)
        # print("Episode:", self.n_episodes, "Episodic Reward:  Min mean Max : ", np.min(arrayresults), mean_reward, np.max(arrayresults))
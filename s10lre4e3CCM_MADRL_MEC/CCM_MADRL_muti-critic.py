import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np
import random
from copy import deepcopy
from numpy import savetxt
from numpy import loadtxt
from utils import to_tensor_var
from Model import ActorNetwork, CriticNetwork
from prioritized_memory import Memory
from mec_env import ENV_MODE, K_CHANNEL, S_E, N_UNITS
class CCM_MADDPG(object):
    def __init__(self, InfdexofResult, env, env_eval, n_agents, state_dim, action_dim, action_lower_bound, action_higher_bound,
                 memory_capacity=10000, target_tau=1, reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_output_activation=torch.tanh, actor_lr=1e-4, critic_lr=1e-3,
                 optimizer_type="adam", max_grad_norm=None, batch_size=64, episodes_before_train=64,
                 epsilon_start=1, epsilon_end=0.01, epsilon_decay=None, use_cuda=False, use_tree=True):
        self.n_agents = n_agents
        self.env = env
        self.env_eval = env_eval
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lower_bound = action_lower_bound
        self.action_higher_bound = action_higher_bound

        self.env_state = env.reset_mec()
        self.n_episodes = 0
        self.roll_out_n_steps = 1

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.memory = Memory(memory_capacity)
        self.use_tree = use_tree
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
            print("epsilon_decay is NOne")
            exit()
        else:
            self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.target_tau = target_tau

        self.actors = [ActorNetwork(self.state_dim, self.action_dim, self.actor_output_activation)] * self.n_agents
        critic_state_dim = self.n_agents * self.state_dim
        critic_action_dim = self.n_agents * self.action_dim
        self.critics = [CriticNetwork(critic_state_dim, critic_action_dim, self.state_dim, self.action_dim)] * self.n_agents
        # to ensure target network and learning network has the same weights
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        if optimizer_type == "adam":
            self.actors_optimizer = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actors_optimizer = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critics_optimizer = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]

        if self.use_cuda:
            for i in range(self.n_agents):
                self.actors[i].cuda()
                self.critics[i].cuda()
                self.actors_target[i].cuda()
                self.critics_target[i].cuda()

        self.eval_episode_rewards = []
        self.server_episode_constraint_exceeds = []
        self.energy_episode_constraint_exceeds = []
        self.time_episode_constraint_exceeds = []
        self.eval_step_rewards = []
        self.mean_rewards = []

        self.episodes = []
        self.Training_episodes = []

        self.Training_episode_rewards = []
        self.Training_step_rewards = []

        self.InfdexofResult = InfdexofResult
        #self.save_models('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
        self.results = []
        self.Training_results = []
        self.serverconstraints = []
        self.energyconstraints = []
        self.timeconstraints = []
    def interact(self, MAX_EPISODES, EPISODES_BEFORE_TRAIN, NUMBER_OF_EVAL_EPISODES):
        while self.n_episodes < MAX_EPISODES:
            self.env_state = self.env.reset_mec()
            if self.n_episodes >= EPISODES_BEFORE_TRAIN:  # 在交换环境一定次数后开始评估
                self.evaluate(NUMBER_OF_EVAL_EPISODES)
                self.evaluateAtTraining(NUMBER_OF_EVAL_EPISODES)
            self.agent_rewards = [[] for n in range(self.n_agents)]
            done = False
            while not done:  # 每10次作为一个episodes
                state = self.env_state
                actor_action, critic_action, hybrid_action = self.choose_action(state, False)
                next_state, reward, done, _, _ = self.env.step_mec(hybrid_action)
                self.Training_step_rewards.append(np.mean(reward))
                if done:
                    self.Training_episode_rewards.append(np.sum(np.array(self.Training_step_rewards)))
                    self.Training_step_rewards = []
                    if self.done_penalty is not None:
                        reward = self.done_penalty
                    self.n_episodes += 1
                else:
                    self.env_state = next_state

                if self.use_tree:
                    self.append_tree_sample(state, actor_action, critic_action, reward, next_state,  done)
                else:
                    self.append_rand_sample(state, actor_action, critic_action, reward, next_state,  done)
            print(self.n_episodes)
            if self.n_episodes >= EPISODES_BEFORE_TRAIN:  # 在交换环境一定次数后开始训练
                self.train()
                #pass
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def append_rand_sample(self, states, actor_actions, critic_actions, rewards, next_states, dones):
        self.memory.rand_add(states, actor_actions, critic_actions, rewards, next_states, dones)

    def append_tree_sample(self, states, actor_actions, critic_actions, rewards, next_states, dones):
        error = 0
        target_q = 0
        current_q = 0
        states_var = to_tensor_var(states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actor_actions_var = to_tensor_var(actor_actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        critic_actions_var = to_tensor_var(critic_actions, self.use_cuda).view(-1, self.n_agents, 1)
        rewards_var = to_tensor_var(rewards, self.use_cuda).view(-1, self.n_agents, 1)
        next_states_var = to_tensor_var(next_states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        whole_states_var = states_var.view(-1, self.n_agents*self.state_dim)
        whole_actor_actions_var = actor_actions_var.view(-1, self.n_agents*self.action_dim)
        whole_next_states_var = next_states_var.view(-1, self.n_agents*self.state_dim)
        #dones_var = to_tensor_var(dones, self.use_cuda).view(-1, 1)
        nextactor_actions = []
        # Calculate next target actions for each agent
        for agent_id in range(self.n_agents):
            next_action_var = self.actors_target[agent_id](next_states_var[:, agent_id, :])
            if self.use_cuda:
                nextactor_actions.append(next_action_var.data.cpu())
            else:
                nextactor_actions.append(next_action_var.data)
        # Concatenate the next target actions into a single tensor
        nextactor_actions_var = torch.cat(nextactor_actions, dim=0)
        nextactor_actions_var = nextactor_actions_var.view(-1, actor_actions_var.size(1), actor_actions_var.size(2))
        whole_nextactor_actions_var = nextactor_actions_var.view(-1, self.n_agents*self.action_dim)
        # target prediction
        nextperQs = []
        currperQs = []
        for nexta in range(self.n_agents):  # to find the maxQ
            if nextactor_actions_var[0, nexta, 0] >= 0:
                nextperQs.append(self.critics_target[nexta](whole_next_states_var[0], whole_nextactor_actions_var[0], next_states_var[0, nexta, :], nextactor_actions_var[0, nexta, :]).detach())
        # if len(nextperQs) == 0:
        #     tar_perQ = self.critics_target[0](whole_next_states_var[0], whole_nextactor_actions_var[0], torch.zeros(self.state_dim), torch.zeros(self.action_dim)).detach()
        # else:
        #     tar_perQ = max(nextperQs)
            if nextactor_actions_var[0, nexta, 0] < 0:
                nextperQs.append(self.critics_target[nexta](whole_next_states_var[0], whole_nextactor_actions_var[0], torch.zeros(self.state_dim), torch.zeros(self.action_dim)).detach())
        nextperQs = torch.stack(nextperQs, dim=0)
        tar = self.reward_scale * rewards_var[0, 0, :] + self.reward_gamma * nextperQs * (1. - dones)
        # cselected = 0
        for a in range(self.n_agents):
            if critic_actions_var[0, a, 0] == 1:
                # current prediction
                curr_perQ = self.critics[a](whole_states_var[0], whole_actor_actions_var[0], states_var[0, a, :], actor_actions_var[0, a, :]).detach()
                error += (tar[a] - curr_perQ)**2
                # cselected += 1
            if critic_actions_var[0, a, 0] == 0:  # if all tasks were allocated locally, the feedback should be sent using the commbined local decision and a fake perAgent that learns the best Q value for that situation
                curr_perQ = self.critics[a](whole_states_var[0], whole_actor_actions_var[0], torch.zeros(self.state_dim), torch.zeros(self.action_dim)).detach()
                error += (tar[a] - curr_perQ)**2

        self.memory.addorupdate(error, (states, actor_actions, critic_actions, rewards, next_states, dones))
    # train on a sample batch
    def train(self):
        # do not train until exploration is enough
        if self.n_episodes <= self.episodes_before_train:
            return

        if self.use_tree:
            tryfetch = 0
            while tryfetch < 3:
                mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
                #print("idxs, is_weights", len(idxs), len(is_weights))
                mini_batch = np.array(mini_batch, dtype=object).transpose()
                if any(not isinstance(arr, np.ndarray) for arr in mini_batch[0]) or any(not isinstance(arr, np.ndarray) for arr in mini_batch[2]):
                    if tryfetch < 3:
                        tryfetch += 1
                    else:
                        print("mini_batch = ", mini_batch)
                        exit()
                else:
                    break
            errors = np.zeros(self.batch_size)
            states = np.vstack(mini_batch[0])
            actor_actions = np.vstack(mini_batch[1])
            critic_actions = np.vstack(mini_batch[2])
            rewards = np.vstack(mini_batch[3])
            next_states = np.vstack(mini_batch[4])
            dones = mini_batch[5]

        else:
            buffer_data = self.memory.rand_sample(self.batch_size)
            errors = np.zeros(self.batch_size)
            states, actor_actions, critic_actions, rewards, next_states, dones = zip(*buffer_data)
            states = np.vstack(states)
            actor_actions = np.vstack(actor_actions)
            critic_actions = np.vstack(critic_actions)
            rewards = np.vstack(rewards)
            next_states = np.vstack(next_states)
            dones = np.vstack(dones)



        # bool to binary
        dones = dones.astype(int)
        states_var = to_tensor_var(states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actor_actions_var = to_tensor_var(actor_actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        critic_actions_var = to_tensor_var(critic_actions, self.use_cuda).view(-1, self.n_agents, 1)
        rewards_var = to_tensor_var(rewards, self.use_cuda).view(-1, self.n_agents, 1)
        next_states_var = to_tensor_var(next_states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        dones_var = to_tensor_var(dones, self.use_cuda).view(-1, 1)

        whole_states_var = states_var.view(-1, self.n_agents*self.state_dim)
        whole_actor_actions_var = actor_actions_var.view(-1, self.n_agents*self.action_dim)
        whole_next_states_var = next_states_var.view(-1, self.n_agents*self.state_dim)

        next_actor_actions = []
        # Calculate next target actions for each agent
        for agent_id in range(self.n_agents):
            next_action_var = self.actors_target[agent_id](next_states_var[:, agent_id, :])
            if self.use_cuda:
                next_actor_actions.append(next_action_var)
            else:
                next_actor_actions.append(next_action_var)
        # Concatenate the next target actions into a single tensor
        next_actor_actions_var = torch.cat(next_actor_actions, dim=1)
        next_actor_actions_var = next_actor_actions_var.view(-1, actor_actions_var.size(1), actor_actions_var.size(2))
        whole_next_actor_actions_var = next_actor_actions_var.view(-1, self.n_agents*self.action_dim)


        #critic
        for agent_critic_id in range(self.n_agents):
            #target prediction
            tar_Qs = []
            cur_Qs = []
            for b in range(self.batch_size):
                next_Q = 0
                curr_Q = 0
                if next_actor_actions_var[b, agent_critic_id, 0] >= 0:
                    next_Q = self.critics_target[agent_critic_id](whole_next_states_var[b], whole_next_actor_actions_var[b], next_states_var[b, agent_critic_id, :], next_actor_actions_var[b, agent_critic_id, :])
                else:
                    next_Q = self.critics_target[agent_critic_id](whole_next_states_var[b], whole_next_actor_actions_var[b], torch.zeros(self.state_dim), torch.zeros(self.action_dim))

                if critic_actions_var[b, agent_critic_id, 0] == 1:
                    curr_Q = self.critics[agent_critic_id](whole_states_var[b], whole_actor_actions_var[b], states_var[b, agent_critic_id, :], actor_actions_var[b, agent_critic_id, :])
                else:
                    curr_Q = self.critics[agent_critic_id](whole_states_var[b], whole_actor_actions_var[b], torch.zeros(self.state_dim), torch.zeros(self.action_dim))

                tar = self.reward_scale * rewards_var[b, 0, :] + self.reward_gamma * next_Q * (1. - dones_var[b])
                tar_Qs.append(tar.detach())
                cur_Qs.append(curr_Q)
                errors[b] += (tar - curr_Q)**2

            current_q = torch.stack(cur_Qs, dim=0)
            target_q = torch.stack(tar_Qs, dim=0)
            critic_loss = nn.MSELoss()(current_q, target_q)

            self.critics_optimizer[agent_critic_id].zero_grad()
            critic_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.critics[agent_critic_id].parameters(), self.max_grad_norm)

            self.critics_optimizer[agent_critic_id].step()
            self._soft_update_target(self.critics_target[agent_critic_id], self.critics[agent_critic_id])# update target


        #different actors
        for agent_id_actor in range(self.n_agents):
            new_actor_actions = []
            # Calculate next target actions for each agent
            for agent_id_new in range(self.n_agents):
                new_actor_action_var = self.actors[agent_id_new](states_var[:, agent_id_new, :])
                if self.use_cuda:
                    new_actor_actions.append(new_actor_action_var)#new_actor_actions.append(newactor_action_var.data.cpu())
                else:
                    new_actor_actions.append(new_actor_action_var)#new_actor_actions.append(newactor_action_var.data)
            # Concatenate the new actions into a single tensor
            new_actor_actions_var = torch.cat(new_actor_actions, dim=1)
            new_actor_actions_var = new_actor_actions_var.view(-1, actor_actions_var.size(1), actor_actions_var.size(2))
            whole_new_actor_actions_var = new_actor_actions_var.view(-1, self.n_agents*self.action_dim)
            actor_loss = []
            for b in range(self.batch_size):
                if critic_actions_var[b, agent_id_actor, 0] == 1:  # if it was selected
                    perQ = self.critics[agent_id_actor](whole_states_var[b], whole_new_actor_actions_var[b], states_var[b, agent_id_actor, :], new_actor_actions_var[b, agent_id_actor, :])
                    actor_loss.append(perQ)

                if critic_actions_var[b, agent_id_actor, 0] == 0:
                    perQ = self.critics[agent_id_actor](whole_states_var[b], whole_new_actor_actions_var[b], torch.zeros(self.state_dim), torch.zeros(self.action_dim))
                    actor_loss.append(perQ)

            actor_loss = torch.stack(actor_loss, dim=0)
            actor_loss = - actor_loss.mean()
            actor_loss.requires_grad_(True)
            self.actors_optimizer[agent_id_actor].zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id_actor].parameters(), self.max_grad_norm)
            self.actors_optimizer[agent_id_actor].step()
            self._soft_update_target(self.actors_target[agent_id_actor], self.actors[agent_id_actor])  # update target network
        if self.use_tree:
            for i in range(self.batch_size):
                idx = idxs[i]
                #print("errors",idx,errors)
                self.memory.update(idx, errors[i])
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
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'actors_target': [actor_target.state_dict() for actor_target in self.actors_target],
            'critics': [critic.state_dict() for critic in self.critics],
            'critics_target': [critic_target.state_dict() for critic_target in self.critics_target],
            # Add other model parameters as needed
        }
        torch.save(checkpoint, path)

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
    def choose_action(self, state, evaluation):
        '''
        checkpoint = torch.load('./checkpoint/models_checkpoint'+str(self.InfdexofResult)+'.pth')
        for agent_id in range(self.n_agents):
            self.actors[agent_id].load_state_dict(checkpoint['actors'][agent_id])
            self.actors_target[agent_id].load_state_dict(checkpoint['actors_target'][agent_id])
            if agent_id == 0:
                self.critics[agent_id].load_state_dict(checkpoint['critics'][agent_id])
                self.critics_target[agent_id].load_state_dict(checkpoint['critics_target'][agent_id])
        '''
        #print("state",state.shape)
        state_var = to_tensor_var([state], self.use_cuda)
        # get actor_action
        actor_action = np.zeros((self.n_agents, self.action_dim))  # actual output of  actor. will be used as is in training. It will be scaled and the task offloading rounded when using hybrid action
        critic_action = np.zeros((self.n_agents))  # used to decide offloade or local. It will be used as indexer of selected tasks in the training as well
        hybrid_action = np.zeros((self.n_agents, self.action_dim))  # to be used only by the environment. its values cobinations of actor and critic actions rounded and scaled as needed by environmebt
        for agent_id in range(self.n_agents):
            action_var = self.actors[agent_id](state_var[:,agent_id,:])
            if self.use_cuda:
                actor_action[agent_id] = action_var.data.cpu().numpy()[0]
            else:
                actor_action[agent_id] = action_var.data.numpy()[0]

        if not evaluation:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                      np.exp(-1. * self.n_episodes / self.epsilon_decay)
            #print("epsilon = ",epsilon)
            # add noise
            noise = np.random.randn(self.n_agents, self.action_dim) * epsilon
            actor_action += noise

            for n in range(self.n_agents):
                for i in range(3):
                    if actor_action[n, i] < -1:
                        actor_action[n, i] = -1
                    if actor_action[n, i] > 1:
                        actor_action[n, i] = 1

        # get critic_action and final (hybrid) actions
        hybrid_action = deepcopy(actor_action)
        # first check if ther is at least one actor that chose to offload
        proposed = np.count_nonzero(actor_action[:, 0] >= 0)
        proposed_indices = np.where(actor_action[:, 0] >= 0)[0]
        sumofproposed = np.sum(state[proposed_indices, 3])
        # print(proposed, proposed_indices, sumofproposed)
        if ENV_MODE == "H2":
            constraint = K_CHANNEL
        elif ENV_MODE == "TOBM":
            constraint = N_UNITS
        else :
            print("Unknown env_mode ", ENV_MODE)
            exit()
        if proposed > 0:  # find their Q-values
            if proposed > constraint or sumofproposed > S_E:  # if the number of agents proposed to offload is greater than the number of available channels
                if not evaluation and (np.random.rand() <= epsilon):  # explore
                    agent_list = np.arange(self.n_agents).tolist()
                    random.shuffle(agent_list)
                    randomorder = random.sample(agent_list, constraint)
                    sizeaccepted = np.sum(state[randomorder, 3])
                    while sizeaccepted > S_E:
                        element_to_delete = random.choice(randomorder)
                        randomorder.remove(element_to_delete)
                        sizeaccepted = np.sum(state[randomorder, 3])
                    critic_action[randomorder] = 1
                else:
                    critic_action_Qs = np.zeros((self.n_agents))
                    critic_action_Qs.fill(-np.inf)
                    # distinguishedandjoined
                    states_var = to_tensor_var(state, self.use_cuda).view(-1, self.n_agents, self.state_dim)
                    whole_states_var = states_var.view(-1, self.n_agents*self.state_dim)
                    actor_action_var = to_tensor_var(actor_action, self.use_cuda).view(-1, self.n_agents, self.action_dim)
                    whole_actions_var = actor_action_var.view(-1, self.n_agents*self.action_dim)
                    for agentid in range(self.n_agents):
                        if actor_action[agentid, 0] >= 0:
                            #print("336",whole_states_var.shape, whole_actions_var.shape, states_var[0,agentid,:].view(1, -1).shape, actor_action_var[0, agentid, :].view(1, -1).shape)
                            critic_action_Qs[agentid] = self.critics[0](whole_states_var.squeeze(), whole_actions_var.squeeze(), states_var[0, agentid, :], actor_action_var[0, agentid, :]).detach()
                    # sort in q values in decending and select using k as constraint
                    sorted_indices = np.argsort(critic_action_Qs)[::-1]
                    # now select tasks
                    countaccepted = 0
                    sizeaccepted = 0
                    for agentid in range(self.n_agents):
                        if actor_action[sorted_indices[agentid], 0] >= 0 and countaccepted < constraint and sizeaccepted + state[sorted_indices[agentid], 3] < S_E:
                            critic_action[sorted_indices[agentid]] = 1
                            countaccepted += 1
                            sizeaccepted += state[sorted_indices[agentid], 3]
            else:  # if the proposed tasks are less than the cosntraints
                for agentid in range(self.n_agents):
                    if hybrid_action[agentid, 0] < 0:
                        critic_action[agentid] = 0
                    else:
                        critic_action[agentid] = 1
        hybrid_action[:, 0] = critic_action
        # get bounded to action_bound
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
        else :
            print("Unknown env_mode ", ENV_MODE)
            exit()
        for i in range(EVAL_EPISODES):
            self.eval_env_state = self.env_eval.reset_mec(i)
            self.eval_step_rewards = []
            self.server_step_constraint_exceeds = 0
            self.energy_step_constraint_exceeds = 0
            self.time_step_constraint_exceeds = 0
            done = False
            while not done:
                state = self.eval_env_state
                # print("state", state)
                actor_action, critic_action, hybrid_action = self.choose_action(state, True)
                proposed = np.count_nonzero(actor_action[:, 0] >= 0)
                proposed_indices = np.where(actor_action[:, 0] >= 0)[0]
                sumofproposed = np.sum(state[proposed_indices, 3])
                next_state, reward, done, eneryconstraint_exceeds, timeconstraint_exceeds = self.env_eval.step_mec(hybrid_action)
                self.eval_step_rewards.append(np.mean(reward))
                self.energy_step_constraint_exceeds += eneryconstraint_exceeds
                self.time_step_constraint_exceeds += timeconstraint_exceeds
                if proposed > constraint or sumofproposed > S_E:  # if constraint exceeded count it
                    self.server_step_constraint_exceeds += 1
                # print(actor_action)
                if done:
                    self.eval_episode_rewards.append(np.sum(np.array(self.eval_step_rewards)))
                    self.server_episode_constraint_exceeds.append(self.server_step_constraint_exceeds/len(self.eval_step_rewards))
                    # the self.eval_step_rewards is used to deduce the step size
                    # print("eval reward and constraint", np.sum(np.array(self.eval_step_rewards)), self.server_step_constraint_exceeds)
                    self.energy_episode_constraint_exceeds.append(self.energy_step_constraint_exceeds/len(self.eval_step_rewards))
                    self.time_episode_constraint_exceeds.append(self.time_step_constraint_exceeds/len(self.eval_step_rewards))
                    self.eval_step_rewards = []
                    self.server_step_constraint_exceeds = 0
                    self.energy_step_constraint_exceeds = 0
                    self.time_step_constraint_exceeds = 0
                    if self.done_penalty is not None:
                        reward = self.done_penalty
                else:
                    self.eval_env_state = next_state
            if i == EVAL_EPISODES-1 and done:
                # print(self.eval_episode_rewards)
                mean_reward = np.mean(np.array(self.eval_episode_rewards))
                mean_constraint = np.mean(np.array(self.server_episode_constraint_exceeds))
                mean_energyconstraint = np.mean(np.array(self.energy_episode_constraint_exceeds))
                mean_timeconstraint = np.mean(np.array(self.time_episode_constraint_exceeds))
                self.eval_episode_rewards = []
                self.server_episode_constraint_exceeds = []
                self.energy_episode_constraint_exceeds = []
                self.time_episode_constraint_exceeds = []
                self.mean_rewards.append(mean_reward)  # to be plotted by the main function
                self.episodes.append(self.n_episodes+1)
                self.results.append(mean_reward)
                self.serverconstraints.append(mean_constraint)
                self.energyconstraints.append(mean_energyconstraint)
                self.timeconstraints.append(mean_timeconstraint)
                arrayresults = np.array(self.results)
                arrayserver = np.array(self.serverconstraints)
                arrayenergy = np.array(self.energyconstraints)
                arraytime = np.array(self.timeconstraints)
                savetxt('./CSV/results/CCM_MADRL'+str(self.InfdexofResult)+'.csv', arrayresults)
                savetxt('./CSV/Server_constraints/CCM_MADRL'+str(self.InfdexofResult)+'.csv', arrayserver)
                savetxt('./CSV/Energy_constraints/CCM_MADRL'+str(self.InfdexofResult)+'.csv', arrayenergy)
                savetxt('./CSV/Time_constraints/CCM_MADRL'+ str(self.InfdexofResult)+'.csv', arraytime)
                # print("Episode:", self.n_episodes, "Episodic Energy:  Min mean Max : ", np.min(arrayenergy), mean_energyconstraint, np.max(arrayenergy))
    def evaluateAtTraining(self, EVAL_EPISODES):
        # print(self.eval_episode_rewards)
        mean_reward = np.mean(np.array(self.Training_episode_rewards))
        self.Training_episode_rewards = []
        # self.mean_rewards.append(mean_reward) # to be plotted by the main function
        self.Training_episodes.append(self.n_episodes+1)
        self.Training_results.append(mean_reward)
        arrayresults = np.array(self.Training_results)
        savetxt('./CSV/AtTraining/CCM_MADRL' + self.InfdexofResult+'.csv', arrayresults)
        # print("Episode:", self.n_episodes, "Episodic Reward:  Min mean Max : ", np.min(arrayresults), mean_reward, np.max(arrayresults))
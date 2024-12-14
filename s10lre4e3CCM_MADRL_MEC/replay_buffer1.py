import numpy as np
import torch
from utils import to_tensor_var


class ReplayBuffer:
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.loc_state_dim = args.state_size
        self.action_dim = args.action_size
        self.glo_state_dim = args.n_agents * args.state_size
        self.glo_action_dim = args.n_agents * args.action_size
        self.episode_limit = args.steps
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        self.buffer = {'loc_current_obs_n': np.empty([self.batch_size, self.n_agents, self.loc_state_dim]),
                       'glo_current_obs_1': np.empty([self.batch_size, self.glo_state_dim]),

                       'loc_reward_n': np.empty([self.batch_size, self.n_agents, 1]),
                       'loc_origin_reward_4xn': np.empty([self.batch_size, 4, self.n_agents]),
                       'glo_reward_n': np.empty([self.batch_size, self.n_agents, 1]),

                       'loc_current_act_n': np.empty([self.batch_size, self.n_agents, self.action_dim]),
                       'loc_current_critic_n': np.empty([self.batch_size, self.n_agents]),
                       'loc_current_log_act_n': np.empty([self.batch_size, self.n_agents, self.action_dim]),
                       'glo_current_act_1': np.empty([self.batch_size, self.glo_action_dim]),
                       'loc_next_act_n': np.empty([self.batch_size, self.n_agents, self.action_dim]),

                       'loc_obs_next_n': np.empty([self.batch_size, self.n_agents, self.loc_state_dim]),
                       'glo_obs_next_1': np.empty([self.batch_size, self.glo_state_dim]),
                       'loc_next_critic_n': np.empty([self.batch_size, self.n_agents]),

                       'glo_done_1': np.empty([self.batch_size, 1])
                       }
        self.episode_num = 0

    def store_transition(self, loc_current_obs_n, glo_current_obs_1,
                         loc_reward_n, loc_origin_reward_4xn, glo_reward_n,
                         loc_current_act_n, loc_current_critic_n, loc_current_log_act_n, glo_current_act_1, loc_next_act_n,
                         loc_obs_next_n, glo_obs_next_1, loc_next_critic_n,
                         glo_done_1):
        self.buffer['loc_current_obs_n'][self.episode_num] = loc_current_obs_n
        self.buffer['glo_current_obs_1'][self.episode_num] = glo_current_obs_1

        self.buffer['loc_reward_n'][self.episode_num] = loc_reward_n
        self.buffer['loc_origin_reward_4xn'][self.episode_num] = loc_origin_reward_4xn
        self.buffer['glo_reward_n'][self.episode_num] = glo_reward_n

        self.buffer['loc_current_act_n'][self.episode_num] = loc_current_act_n
        self.buffer['loc_current_critic_n'][self.episode_num] = loc_current_critic_n
        self.buffer['loc_current_log_act_n'][self.episode_num] = loc_current_log_act_n
        self.buffer['glo_current_act_1'][self.episode_num] = glo_current_act_1
        self.buffer['loc_next_act_n'][self.episode_num] = loc_next_act_n

        self.buffer['loc_obs_next_n'][self.episode_num] = loc_obs_next_n
        self.buffer['glo_obs_next_1'][self.episode_num] = glo_obs_next_1
        self.buffer['loc_next_critic_n'][self.episode_num] = loc_next_critic_n

        self.buffer['glo_done_1'][self.episode_num] = glo_done_1




    def get_training_data(self, use_cuda):
        batch = {}
        for key in self.buffer.keys():
            batch[key] = to_tensor_var(self.buffer[key], use_cuda)
        return batch

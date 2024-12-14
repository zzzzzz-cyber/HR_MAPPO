import numpy as np
import torch
from utils import to_tensor_var


class VAEReplayBuffer:
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.loc_state_dim = args.state_size
        self.action_dim = args.action_size
        self.glo_state_dim = args.n_agents * args.state_size
        self.glo_action_dim = args.n_agents * args.action_size
        self.batch_size = args.batch_size
        self.episode_limit = args.steps
        self.episode_num = 0
        self.buffer = None
        self.VAE_reset_buffer()
        # create a buffer (dictionary)

    def VAE_reset_buffer(self):
        self.buffer = {'VAE_mu': np.empty([self.batch_size, self.episode_limit, self.n_agents, 6]),
                       'VAE_std': np.empty([self.batch_size, self.episode_limit, self.n_agents, 6]),

                       'recons_action': np.empty([self.batch_size, self.episode_limit, self.n_agents, self.action_dim]),
                       'original_action': np.empty([self.batch_size, self.episode_limit, self.n_agents, self.action_dim]),

                       'prediction_residual': np.empty([self.batch_size, self.episode_limit, self.n_agents, self.loc_state_dim]),
                       'true_residual': np.empty([self.batch_size, self.episode_limit, self.n_agents, self.loc_state_dim])}
        self.episode_num = 0

    def VAE_store_transition(self, episode_step, vae_mu, vae_std,
                             recons_action, original_action,
                             prediction_residual, true_residual):
        self.buffer['VAE_mu'][self.episode_num][episode_step] = vae_mu
        self.buffer['VAE_std'][self.episode_num][episode_step] = vae_std

        self.buffer['recons_action'][self.episode_num][episode_step] = recons_action
        self.buffer['original_action'][self.episode_num][episode_step] = original_action

        self.buffer['prediction_residual'][self.episode_num][episode_step] = prediction_residual
        self.buffer['true_residual'][self.episode_num][episode_step] = true_residual

    def VAE_get_training_data(self, use_cuda):
        batch = {}
        for key in self.buffer.keys():
            batch[key] = to_tensor_var(self.buffer[key], use_cuda)
        return batch

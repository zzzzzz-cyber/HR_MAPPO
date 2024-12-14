import torch as th
import torch.nn as nn
class ActorNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 32)

        self.discrete = nn.Linear(32, 1)
        self.local_policy_mu = nn.Linear(32, 1)
        self.local_policy_std = nn.Linear(32, 1)

        self.offload_policy_mu = nn.Linear(32, 1)
        self.offload_policy_std = nn.Linear(32, 1)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        if self.output_activation == nn.functional.softmax:
            mu = self.output_activation(self.fc3(out), dim=-1)
            std = th.relu(self.log_std(out))
        else:
            mu = self.output_activation(self.fc3(out))
            std = nn.functional.softplus(self.log_std(out))
        return mu, std


class CriticNetwork(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state):
        # Input = th.cat([state, action], -1)
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class ActorNetwork_RNN(nn.Module):
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(ActorNetwork_RNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.rnn = nn.GRUCell(128, 32)
        self.rnn_hidden = None
        self.fc3 = nn.Linear(32, output_size)
        self.log_std = nn.Linear(32, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        out = th.relu(self.fc1(state))
        self.rnn_hidden = self.rnn(out, self.rnn_hidden)

        if self.output_activation == nn.functional.softmax:
            mu = self.output_activation(self.fc3(self.rnn_hidden), dim=-1)
            std = th.relu(self.log_std(self.rnn_hidden))
        else:
            mu = self.output_activation(self.fc3(self.rnn_hidden))
            std = nn.functional.softplus(self.log_std(self.rnn_hidden))
        return mu, std

class CriticNetwork_RNN(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork_RNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.rnn = nn.GRUCell(256, 128)
        self.rnn_hidden = None
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state):
        # Input = th.cat([state, action], -1)
        out = th.relu(self.fc1(state))
        self.rnn_hidden = self.rnn(out, self.rnn_hidden)
        out = self.fc3(self.rnn_hidden)
        return out

class decentralized_CriticNetwork(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, output_size=1, init_w=3e-3):
        super(decentralized_CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state):
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        out = self.fc3(out)
        return out



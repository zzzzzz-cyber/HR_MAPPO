import torch as th
import torch.nn as nn
class NormLayer(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # 层归一化包含两个可以学习的参数
        self.eps = eps

    def forward(self, x):
        norm = (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps)
        return norm

class ActorNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = NormLayer(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = NormLayer(128)
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        if self.output_activation == nn.functional.softmax:
            out = self.output_activation(self.fc3(out), dim=-1)
        else:
            out = self.output_activation(self.fc3(out))
        return out

class CriticNetwork(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, pestate, peraction, output_size=1, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + pestate + peraction, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.ln1 = NormLayer(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = NormLayer(128)
        self.fc3 = nn.Linear(128, output_size)
        
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, action, pstate, paction):
        out = th.cat([state, action, pstate, paction], 0)
        out = th.relu(self.fc1(out))
        out = th.relu(self.fc2(out))
        out = self.fc3(out)
        return out



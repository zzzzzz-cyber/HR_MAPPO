import torch
import torch as th
import torch.nn as nn

MSE = nn.MSELoss(reduction='none')
class ActorNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 32)

        # self.discrete = nn.Linear(32, 1)
        self.fc3 = nn.Linear(32, output_size)
        self.log_std = nn.Linear(32, output_size)

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

class Q_CriticNetwork(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(Q_CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, action):
        Input = th.cat([state, action], -1)
        out = th.relu(self.fc1(Input))
        out = th.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class CriticNetwork_vae(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork_vae, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, action):
        Input = th.cat([state, action], -1)
        out = th.relu(self.ln1(self.fc1(Input)))
        out = th.relu(self.ln2(self.fc2(out)))
        out = self.fc3(out)
        return out


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
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class CriticNetwork_noise(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork_noise, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, noise):
        state = torch.cat([state, noise], -1)
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class CriticNetwork_overall_noise(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork_overall_noise, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.ln1 = NormLayer(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = NormLayer(128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, noise):
        state = torch.cat([state, noise], -1)
        out = th.relu(self.ln1(self.fc1(state)))
        out = th.relu(self.ln2(self.fc2(out)))
        out = self.fc3(out)
        return out

class CriticNetwork_single(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork_single, self).__init__()
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


class ActorNetwork_soft(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(ActorNetwork_soft, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 32)

        self.discrete = nn.Linear(32, 2)

        self.fc3 = nn.Linear(32, 2)
        self.log_std = nn.Linear(32, 2)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        ac = nn.functional.softmax(self.discrete(out), -1)

        mu = self.output_activation(self.fc3(out))
        std = nn.functional.softplus(self.log_std(out))

        return ac, mu, std

class ActorNetwork_soft_RNN1(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(ActorNetwork_soft_RNN1, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)

        self.rnn = nn.GRUCell(128, 32)
        self.rnn_hidden = None

        self.discrete = nn.Linear(32, 2)

        self.fc3 = nn.Linear(32, 2)
        self.log_std = nn.Linear(32, 2)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation

    def __call__(self, state):
        out = th.relu(self.fc1(state))
        self.rnn_hidden = self.rnn(out, self.rnn_hidden)

        ac = nn.functional.softmax(self.discrete(self.rnn_hidden), -1)
        mu = self.output_activation(self.fc3(self.rnn_hidden))
        std = nn.functional.softplus(self.log_std(self.rnn_hidden))
        return ac, mu, std

class D3QN_advantage(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, continue_action_dim, output_activation, init_w=1e-3):
        super(D3QN_advantage, self).__init__()
        self.fc1 = nn.Linear(state_dim + continue_action_dim, 128)
        self.fc2 = nn.Linear(128, 32)

        self.discrete = nn.Linear(32, 2)
        self.output_activation = output_activation

        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)

        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, continue_action):
        out = th.cat([state, continue_action], -1)
        out = th.relu(self.fc1(out))
        out = th.relu(self.fc2(out))
        ac = self.output_activation(self.discrete(out))
        return ac

class D3QN_value(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, continue_action_dim, output_activation, init_w=1e-3):
        super(D3QN_value, self).__init__()
        self.fc1 = nn.Linear(state_dim + continue_action_dim, 256)
        self.ln1 = NormLayer(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = NormLayer(128)
        self.discrete = nn.Linear(128, 2)
        self.output_activation = output_activation

        # self.fc1.weight.data.uniform_(-init_w, init_w)
        # self.fc1.bias.data.uniform_(-init_w, init_w)
        # #
        # self.fc2.weight.data.uniform_(-init_w, init_w)
        # self.fc2.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, continue_action):
        out = th.cat([state, continue_action], -1)
        out = self.ln1(self.fc1(out))
        out = self.ln2(self.fc2(out))
        ac = self.discrete(out)
        return ac

class ParmNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(ParmNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state):
        out = th.relu(self.ln1(self.fc1(state)))
        out = th.relu(self.ln2(self.fc2(out)))
        out = self.fc3(out)
        return out


class DDPG_Actor(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(DDPG_Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.ln1 = NormLayer(128)

        self.lstm = nn.LSTMCell(128, 32)
        self.lstm_hidden = None
        self.lstm_memorize = None

        self.ln2 = NormLayer(32)
        self.action1 = nn.Linear(32, 1)
        self.action2 = nn.Linear(32, 2)

        self.action1.weight.data.uniform_(-init_w, init_w)
        self.action1.bias.data.uniform_(-init_w, init_w)

        self.action2.weight.data.uniform_(-init_w, init_w)
        self.action2.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        out = th.relu(self.ln1(self.fc1(state)))
        self.lstm_hidden, self.lstm_memorize = self.lstm(out, (self.lstm_hidden, self.lstm_memorize))
        self.lstm_hidden = th.relu(self.ln2(self.lstm_hidden))
        ac1 = self.output_activation(self.action1(self.lstm_hidden))
        ac2 = self.output_activation(self.action2(self.lstm_hidden))

        return ac1, ac2

class DDPG_Actor2(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(DDPG_Actor2, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.action = nn.Linear(128, 2)

        self.action.weight.data.uniform_(-init_w, init_w)
        self.action.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        out = self.output_activation(self.action(out))

        return out

class DDPG_critic(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(DDPG_critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher

        self.ln1 = NormLayer(256)
        self.lstm = nn.LSTMCell(256, 128)
        self.lstm_hidden = None
        self.lstm_memorize = None
        self.ln2 = NormLayer(128)
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, action):
        Input = th.cat([state, action], -1)
        out = th.relu(self.ln1(self.fc1(Input)))
        self.lstm_hidden, self.lstm_memorize = self.lstm(out, (self.lstm_hidden, self.lstm_memorize))
        self.lstm_hidden = th.relu(self.ln2(self.lstm_hidden))
        out = self.fc3(self.lstm_hidden)
        return out

class ActorNetwork_soft_RNN2(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(ActorNetwork_soft_RNN2, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.rnn = nn.GRUCell(128, 32)
        self.rnn_hidden = None
        self.discrete = nn.Linear(32, 2)

        self.action2_mu = nn.Linear(32, 1)
        self.action2_std = nn.Linear(32, 1)

        self.action3_mu = nn.Linear(32, 1)
        self.action3_std = nn.Linear(32, 1)

        self.action2_mu.weight.data.uniform_(-init_w, init_w)
        self.action2_mu.bias.data.uniform_(-init_w, init_w)

        self.action3_mu.weight.data.uniform_(-init_w, init_w)
        self.action3_mu.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        out = th.relu(self.fc1(state))
        self.rnn_hidden = self.rnn(out, self.rnn_hidden)
        ac1 = nn.functional.softmax(self.discrete(self.rnn_hidden), -1)

        mu2 = self.output_activation(self.action2_mu(self.rnn_hidden))
        std2 = nn.functional.softplus(self.action2_std(self.rnn_hidden))

        mu3 = self.output_activation(self.action3_mu(self.rnn_hidden))
        std3 = nn.functional.softplus(self.action3_std(self.rnn_hidden))

        return ac1, mu2, std2, mu3, std3

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

class FeedForward(nn.Module):
    def __init__(self, d_model, latent_dim, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.mean1 = nn.Linear(d_model, latent_dim)
        self.log_std1 = nn.Linear(d_model, latent_dim)

        self.mean2 = nn.Linear(d_model, latent_dim)
        self.log_std2 = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        x = self.dropout(nn.functional.relu(self.linear_1(x)))
        mu1 = torch.tanh(self.mean1(x))
        log_std1 = nn.functional.softplus(self.log_std1(x))

        mu2 = torch.tanh(self.mean2(x))
        log_std2 = nn.functional.softplus(self.log_std2(x))
        return mu1, log_std1, mu2, log_std2

class VAE(nn.Module):
    def __init__(self, discrete_action_dim, continue_action_dim, env_continue_action_dim, state_dim):
        super().__init__()
        self.embedding = nn.Embedding(2, discrete_action_dim)
        self.p_action = None

        # 编码器
        self.encoding = nn.Linear(env_continue_action_dim, 256)
        self.ln_encoding = NormLayer(256)

        self.condition1 = nn.Linear(discrete_action_dim + state_dim, 256)
        self.ln_condition1 = NormLayer(256)

        self.fc1 = nn.Linear(256, 256)
        # μ和σ
        self.mean = nn.Linear(256, continue_action_dim)
        self.log_std = nn.Linear(256, continue_action_dim)

        self.discrete_action_dim = discrete_action_dim
        # 重构
        self.latent = nn.Linear(continue_action_dim + continue_action_dim, 256)
        self.ln_latent = NormLayer(256)

        self.condition2 = nn.Linear(discrete_action_dim + state_dim, 256)

        self.fc2 = nn.Linear(256, 256)
        self.re_continue = nn.Linear(256, env_continue_action_dim)
        self.fc3 = nn.Linear(256, 256)
        self.ln3 = NormLayer(256)
        self.ff = FeedForward(256, continue_action_dim)

    def encoder(self, embedding_dis, continue_action, state):
        x = torch.cat([embedding_dis, state], -1)
        condition = self.condition1(x)
        encoding = self.encoding(continue_action)
        x = self.ln_encoding(condition * encoding)
        mu1, std1, mu2, std2 = self.ff(x)
        return mu1, std1, mu2, std2

    def decoder(self, embedding, z, state):
        x = torch.cat([embedding, state], -1)
        condition = self.condition2(x)
        latent = self.latent(z)
        x = condition * latent
        x = self.ln_latent(x)
        out = torch.tanh(self.re_continue(x))
        return out

    def reparameterize(self, mu1, log_std1, mu2, log_std2):
        std1 = torch.exp(0.5 * log_std1)
        eps1 = torch.randn_like(std1)

        std2 = torch.exp(0.5 * log_std2)
        eps2 = torch.randn_like(std2)
        return eps1 * std1 + mu1, eps2 * std2 + mu2

    def forward(self, discrete_action, continue_action, state):
        embedding = self.embedding(discrete_action)
        mu1, std1, mu2, std2 = self.encoder(embedding, continue_action, state)
        z1, z2 = self.reparameterize(mu1, std1, mu2, std2)
        z = torch.cat([z1, z2], -1)
        decode_action = self.decoder(embedding, z, state)
        return mu1, std1, mu2, std2, decode_action

class VAE2(nn.Module):
    def __init__(self, discrete_action_dim, continue_action_dim, env_continue_action_dim, state_dim):
        super().__init__()
        self.embedding = nn.Embedding(2, discrete_action_dim)
        self.p_action = None

        # 编码器
        self.encoding = nn.Linear(env_continue_action_dim, 256)
        self.ln_encoding = nn.LayerNorm(256)

        self.condition1 = nn.Linear(discrete_action_dim + state_dim, 256)
        self.ln_condition1 = nn.LayerNorm(256)

        self.fc1 = nn.Linear(256, 256)
        # μ和σ
        self.mean = nn.Linear(256, continue_action_dim)
        self.log_std = nn.Linear(256, continue_action_dim)

        self.discrete_action_dim = discrete_action_dim
        # 重构
        self.latent = nn.Linear(continue_action_dim, 256)
        self.ln_latent = nn.LayerNorm(256)

        self.condition2 = nn.Linear(discrete_action_dim + state_dim, 256)
        self.ln_condition2 = nn.LayerNorm(256)


        self.fc2 = nn.Linear(256, 256)
        self.re_continue = nn.Linear(256, env_continue_action_dim)

        self.fc3 = nn.Linear(256, 256)
        self.ln3 = nn.LayerNorm(256)

        self.prediction = nn.Linear(256, state_dim)


class ActorNetwork_vae(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, discrete_action_dim, continue_action_dim, output_activation, init_w=1e-3):
        super(ActorNetwork_vae, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        self.discrete = nn.Linear(128, discrete_action_dim)

        self.fc3 = nn.Linear(128, continue_action_dim)
        # self.fc3.weight.data.uniform_(-init_w, init_w)
        # self.fc3.bias.data.uniform_(-init_w, init_w)

        # self.discrete.weight.data.uniform_(-init_w, init_w)
        # self.discrete.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
        # self.relu = nn.LeakyReLU(negative_slope=5e-2)
    def __call__(self, state):
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        ac1 = self.output_activation(self.discrete(out))
        ac2 = self.output_activation(self.fc3(out))

        return ac1, ac2

class ActorNetwork_ppovae(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, discrete_action_dim, continue_action_dim, output_activation, init_w=1e-3):
        super(ActorNetwork_ppovae, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        self.discrete_action = nn.Linear(128, discrete_action_dim)
        self.discrete_action_std = nn.Linear(128, discrete_action_dim)

        self.continue_action = nn.Linear(128, continue_action_dim)
        self.continue_action_std = nn.Linear(128, continue_action_dim)

        self.discrete_action.weight.data.uniform_(-init_w, init_w)
        self.discrete_action.bias.data.uniform_(-init_w, init_w)

        self.continue_action.weight.data.uniform_(-init_w, init_w)
        self.continue_action.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation

    def __call__(self, state):
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        mu1 = self.output_activation(self.discrete_action(out))
        std1 = nn.functional.softplus(self.discrete_action_std(out))

        mu2 = self.output_activation(self.continue_action(out))
        std2 = nn.functional.softplus(self.continue_action_std(out))

        return mu1, std1, mu2, std2

class ActorNetwork_ppovae2(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, discrete_action_dim, continue_action_dim, output_activation, init_w=1e-3):
        super(ActorNetwork_ppovae2, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)

        self.discrete_action = nn.Linear(128, 2)

        self.continue_action = nn.Linear(128, continue_action_dim)
        self.continue_action_std = nn.Linear(128, continue_action_dim)

        self.discrete_action.weight.data.uniform_(-init_w, init_w)
        self.discrete_action.bias.data.uniform_(-init_w, init_w)

        self.continue_action.weight.data.uniform_(-init_w, init_w)
        self.continue_action.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation

    def __call__(self, state):
        out = th.relu(self.fc1(state))
        out = th.relu(self.fc2(out))
        ac = nn.functional.softmax(self.discrete_action(out), -1)

        mu2 = self.output_activation(self.continue_action(out))
        std2 = nn.functional.softplus(self.continue_action_std(out))

        return ac, mu2, std2

class ActorNetwork_ppovae3_M(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, discrete_action_dim, continue_action_dim, output_activation, init_w=1e-3):
        super(ActorNetwork_ppovae3_M, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        self.discrete_action = nn.Linear(128, 2)

        self.continue_action1 = nn.Linear(128, continue_action_dim)
        self.continue_action_std1 = nn.Linear(128, continue_action_dim)

        self.continue_action2 = nn.Linear(128, continue_action_dim)
        self.continue_action_std2 = nn.Linear(128, continue_action_dim)

        self.discrete_action.weight.data.uniform_(-init_w, init_w)
        self.discrete_action.bias.data.uniform_(-init_w, init_w)

        # activation function for the output
        self.output_activation = output_activation

    def __call__(self, state):
        out = th.relu(self.ln1(self.fc1(state)))
        out = th.relu(self.ln2(self.fc2(out)))
        ac = nn.functional.softmax(self.discrete_action(out), -1)

        mu2 = self.output_activation(self.continue_action1(out))
        std2 = nn.functional.softplus(self.continue_action_std1(out))

        mu3 = self.output_activation(self.continue_action2(out))
        std3 = nn.functional.softplus(self.continue_action_std2(out))

        return ac, mu2, std2, mu3, std3

class CriticNetwork_tdvae(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork_tdvae, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        # state_dim + discrete_action_dim + continue_action_dim = for the combined,
        # equivalent of it for the per agent, and 1 for distinguisher
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, action):
        Input = th.cat([state, action], -1)
        out = th.relu(self.fc1(Input))
        out = th.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ActorNetwork_soft_tanh(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(ActorNetwork_soft_tanh, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 32)
        self.ln2 = nn.LayerNorm(32)
        self.action1 = nn.Linear(32, 2)

        self.action2_mu = nn.Linear(32, 1)
        self.action2_std = nn.Linear(32, 1)

        self.action3_mu = nn.Linear(32, 1)
        self.action3_std = nn.Linear(32, 1)

        self.action1.weight.data.uniform_(-init_w, init_w)
        self.action1.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        out = th.relu(self.fc1(state))

        out = th.relu(self.fc2(out))
        ac1 = nn.functional.softmax(self.action1(out), -1)

        mu2 = self.output_activation(self.action2_mu(out))
        std2 = nn.functional.softplus(self.action2_std(out))

        mu3 = self.output_activation(self.action3_mu(out))
        std3 = nn.functional.softplus(self.action3_std(out))

        return ac1, mu2, std2, mu3, std3

class ActorNetwork_TD3(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(ActorNetwork_TD3, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.ln1 = NormLayer(128)
        self.fc2 = nn.Linear(128, 32)
        self.ln2 = NormLayer(32)
        self.action1 = nn.Linear(32, 1)
        self.action2 = nn.Linear(32, 2)

        self.action1.weight.data.uniform_(-init_w, init_w)
        self.action1.bias.data.uniform_(-init_w, init_w)

        self.action2.weight.data.uniform_(-init_w, init_w)
        self.action2.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        out = self.ln1(self.fc1(state))
        out = self.ln2(self.fc2(out))
        ac1 = self.output_activation(self.action1(out))
        ac2 = self.output_activation(self.action2(out))
        return ac1, ac2

class ActorNetwork_RNN(nn.Module):
    def __init__(self, state_dim, output_size, output_activation, init_w=1e-3):
        super(ActorNetwork_RNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)

        self.fc2 = nn.Linear(128, 32)

        self.rnn = nn.GRUCell(128, 32)
        self.rnn_hidden = None
        self.fc3 = nn.Linear(32, output_size)
        self.log_std = nn.Linear(32, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        # activation function for the output
        self.output_activation = output_activation
    def __call__(self, state):
        x1 = th.relu(self.fc1(state))

        x2 = th.relu(self.fc2(x1))
        self.rnn_hidden = self.rnn(x1, self.rnn_hidden)

        x3 = x2 + self.rnn_hidden

        if self.output_activation == nn.functional.softmax:
            mu = self.output_activation(self.fc3(x3), dim=-1)
            std = th.relu(self.log_std(x3))
        else:
            mu = self.output_activation(self.fc3(x3))
            std = nn.functional.softplus(self.log_std(x3))
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

class CriticNetwork_local_state(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, pestate, peraction, output_size=1, init_w=3e-3):
        super(CriticNetwork_local_state, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + pestate + peraction, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, action, pstate, paction):
        out = th.cat([state, action, pstate, paction], -1)
        out = th.relu(self.fc1(out))
        out = th.relu(self.fc2(out))
        out = self.fc3(out)
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



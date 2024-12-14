import torch
import torch.nn as nn
import torch.nn.functional as F
import math

MSE = nn.MSELoss(reduction='none')


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, feature_dim, dk_dim, dropout=0.1):
        super().__init__()
        self.d_model = feature_dim
        self.d_k = dk_dim
        self.h = heads
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(feature_dim, feature_dim)

    def attention(self, q, k, v, d_k, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = self.attention(q, k, v, self.d_k, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class NormLayer(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # 层归一化包含两个可以学习的参数
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
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
        x = self.dropout(F.relu(self.linear_1(x)))
        mu1 = torch.tanh(self.mean1(x))
        log_std1 = nn.functional.softplus(self.log_std1(x))

        mu2 = torch.tanh(self.mean2(x))
        log_std2 = nn.functional.softplus(self.log_std2(x))
        return mu1, log_std1, mu2, log_std2


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dk_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.attention = MultiHeadAttention(heads, d_model, dk_dim, dropout=dropout)
        self.norm_2 = NormLayer(d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        x1 = self.norm_1(x)
        x2 = x + self.attention(x1, x1, x1)
        x3 = self.norm_2(x2)
        x4 = x2 + self.fc(x3)
        return x4


class Encoder(nn.Module):
    def __init__(self, discrete_action_dim, continue_action_dim, env_continue_action_dim, state_dim, d_model, dk, N, heads, dropout):
        super().__init__()
        self.N = N
        self.encoding = nn.Linear(env_continue_action_dim, d_model)
        self.condition = nn.Linear(discrete_action_dim + state_dim, d_model)
        self.ln = NormLayer(d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, dk, heads, dropout) for _ in range(N)])
        self.ff = FeedForward(d_model, continue_action_dim)

    def forward(self, embedding_dis, continue_action, state):
        x = torch.cat([embedding_dis, state], -1)
        condition = self.condition(x)
        encoding = self.encoding(continue_action)
        x = self.ln(condition * encoding)
        for layer in self.layers:
            x = layer(x)
        mu1, std1, mu2, std2 = self.ff(x)
        return mu1, std1, mu2, std2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, dk_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dk_dim, dropout=dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model)

    def forward(self, hidden_data):
        x1 = self.norm_1(hidden_data)
        x2 = hidden_data + self.dropout_1(self.attn_1(x1, x1, x1))

        x3 = self.norm_2(x2)
        x4 = x2 + self.dropout_2(self.fc1(x3))
        return x4


class Decoder(nn.Module):
    def __init__(self, discrete_action_dim, continue_action_dim, env_continue_action_dim, state_dim, d_model, dk, N, heads, dropout):
        super().__init__()
        self.N = N
        self.condition = nn.Linear(discrete_action_dim + state_dim, d_model)
        self.latent = nn.Linear(continue_action_dim + continue_action_dim, d_model)
        self.norm_1 = NormLayer(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, dk, heads, dropout) for _ in range(N)])
        self.norm_2 = NormLayer(d_model)
        self.out = nn.Linear(d_model, env_continue_action_dim)

    def forward(self, embedding, z, state):
        x = torch.cat([embedding, state], -1)
        condition = self.condition(x)
        latent = self.latent(z)
        x = condition * latent
        x = self.norm_1(x)
        for layer in self.layers:
            x = layer(x)
        out = torch.tanh(self.out(x))
        return out


class Transformer(nn.Module):
    def __init__(self, discrete_action_dim, continue_action_dim, env_continue_action_dim, state_dim, d_model, dk, N, heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(2, discrete_action_dim)
        self.encoder = Encoder(discrete_action_dim, continue_action_dim, env_continue_action_dim, state_dim, d_model, dk, N, heads, dropout)
        self.decoder = Decoder(discrete_action_dim, continue_action_dim, env_continue_action_dim, state_dim, d_model, dk, N, heads, dropout)

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
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
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
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
        ac = nn.functional.softmax(self.discrete_action(out), -1)

        mu2 = self.output_activation(self.continue_action(out))
        std2 = nn.functional.softplus(self.continue_action_std(out))

        return ac, mu2, std2

class ActorNetwork_ppovae3(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, discrete_action_dim, continue_action_dim, output_activation, init_w=1e-3):
        super(ActorNetwork_ppovae3, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = NormLayer(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = NormLayer(128)

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
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
        ac = nn.functional.softmax(self.discrete_action(out), -1)

        mu2 = self.output_activation(self.continue_action1(out))
        std2 = nn.functional.softplus(self.continue_action_std1(out))

        mu3 = self.output_activation(self.continue_action2(out))
        std3 = nn.functional.softplus(self.continue_action_std2(out))

        return ac, mu2, std2, mu3, std3

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
        out = torch.relu(self.fc1(state))
        self.rnn_hidden = self.rnn(out, self.rnn_hidden)
        out = self.fc3(self.rnn_hidden)
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
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
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
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class CriticNetwork_overall_noise_T(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork_overall_noise_T, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state, noise):
        state = torch.cat([state, noise], -1)
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class CriticNetwork_single_T(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork_single_T, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256) # state_dim + action_dim = for the combined, equivalent of it for the per agent, and 1 for distinguisher
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def __call__(self, state):
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out


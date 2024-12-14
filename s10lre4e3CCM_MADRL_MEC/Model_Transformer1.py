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
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model, latent_dim, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.mean = nn.Linear(d_model, latent_dim)
        self.log_std = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        mu = torch.tanh(self.mean(x))
        log_std = nn.functional.softplus(self.log_std(x))
        return mu, log_std


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
        self.embedding = nn.Embedding(2, discrete_action_dim)
        self.encoding = nn.Linear(env_continue_action_dim, d_model)
        self.condition = nn.Linear(discrete_action_dim + state_dim, d_model)
        self.ln = NormLayer(d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, dk, heads, dropout) for _ in range(N)])
        self.ff = FeedForward(d_model, continue_action_dim)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.xavier_normal_(self.embedding.weight)
    #     nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, discrete_action, continue_action, state):
        embedding_dis = self.embedding(discrete_action)
        x = torch.cat([embedding_dis, state], -1)
        condition = self.condition(x)
        encoding = self.encoding(continue_action)
        x = self.ln(condition * encoding)
        for layer in self.layers:
            x = layer(x)
        mu, std = self.ff(x)
        return mu, std


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
        self.embedding = nn.Embedding(2, discrete_action_dim)
        self.condition = nn.Linear(discrete_action_dim + state_dim, d_model)
        self.latent = nn.Linear(continue_action_dim, d_model)
        self.norm_1 = NormLayer(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, dk, heads, dropout) for _ in range(N)])
        self.norm_2 = NormLayer(d_model)
        self.out = nn.Linear(d_model, env_continue_action_dim)

    def forward(self, discreate_action, z, state):
        embedding = self.embedding(discreate_action)
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
        self.encoder = Encoder(discrete_action_dim, continue_action_dim, env_continue_action_dim, state_dim, d_model, dk, N, heads, dropout)
        self.decoder = Decoder(discrete_action_dim, continue_action_dim, env_continue_action_dim, state_dim, d_model, dk, N, heads, dropout)

    def reparameterize(self, mu, log_std):
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, discrete_action, continue_action, state):
        mu, std = self.encoder(discrete_action, continue_action, state)
        z = self.reparameterize(mu, std)
        decode_action = self.decoder(discrete_action, z, state)
        return mu, std, decode_action

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


class CriticNetwork_overall_noise(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, output_size=1, init_w=3e-3):
        super(CriticNetwork_overall_noise, self).__init__()
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
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out


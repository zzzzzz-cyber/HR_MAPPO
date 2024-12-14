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

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
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

    def forward(self, x, mask):
        x1 = self.norm_1(x)
        x2 = x + self.attention(x1, x1, x1, mask)

        x3 = self.norm_2(x2)
        x4 = x2 + self.fc(x3)

        return x4

class Encoder(nn.Module):
    def __init__(self, discrete_action_dim, env_continue_action_dim, state_dim, d_model, dk, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embedding = nn.Embedding(2, discrete_action_dim)
        self.condition = nn.Linear(discrete_action_dim + state_dim, d_model)
        self.ln_condition = NormLayer(d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, dk, heads, dropout) for _ in range(N)])
        self.ff = FeedForward(d_model, env_continue_action_dim)

        self.fc1 = nn.Linear(state_dim, d_model)
        self.fc2 = nn.Linear(d_model, 2)

    def discrete_action(self, state):
        out = torch.relu(self.fc1(state))
        ac = nn.functional.softmax(self.fc2(out), -1)
        return ac

    def continue_action(self, discrete_action, state, mask):
        x = torch.cat([discrete_action, state], -1)
        x = self.ln_condition(self.condition(x))
        for layer in self.layers:
            x = layer(x, mask)
        mu, std = self.ff(x)
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


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, z, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(z)
        x = z + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.norm = nn.LayerNorm(d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

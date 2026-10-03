"""Slimmable one-layer LSTM language model (paper Sec. III-A).

A width ratio r activates the first floor(r * 256) hidden units: the LSTM gate rows,
the recurrent columns and the output-layer columns are sliced, so a client with a
narrower subnet really computes less.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeterogeneousSubnetLSTM(nn.Module):
    def __init__(self, vocab_size, target_vocab_size, embedding_dim, hidden_dim, num_layers=1,
                 subnet_ratio=0.5, min_active_units=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.subnet_ratio = float(subnet_ratio)
        self.active_units = max(min_active_units, min(int(hidden_dim * self.subnet_ratio), hidden_dim))

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, target_vocab_size)

        mask = torch.zeros(num_layers, hidden_dim)
        mask[:, : self.active_units] = 1.0
        self.register_buffer("subnet_mask", mask)

    def forward(self, x, use_subnet=True):
        if not use_subnet:  # full supernet (used for the global test sets)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
            out, _ = self.lstm(self.embedding(x), (h0, h0.clone()))
            return self.fc(out[:, -1, :])
        return forward_subnet_with_indices(self, x, list(range(self.active_units)), self.hidden_dim)


def forward_subnet_with_indices(model, x, active_idx_list, hidden_dim):
    """LSTM forward pass using only the hidden units in `active_idx_list`
    (a prefix for HASA / HeteroFL, a random or rolling set for the other baselines)."""
    device = x.device
    active_idx = torch.tensor(active_idx_list, device=device, dtype=torch.long)
    a = len(active_idx)
    batch_size = x.size(0)
    embedded = model.embedding(x)

    gate_indices = torch.cat([active_idx, active_idx + hidden_dim,
                              active_idx + 2 * hidden_dim, active_idx + 3 * hidden_dim])
    weight_ih = model.lstm.weight_ih_l0[gate_indices, :]
    weight_hh = model.lstm.weight_hh_l0[gate_indices][:, active_idx]
    bias_ih = model.lstm.bias_ih_l0[gate_indices]
    bias_hh = model.lstm.bias_hh_l0[gate_indices]

    h_t = torch.zeros(batch_size, a, device=device)
    c_t = torch.zeros(batch_size, a, device=device)
    for t in range(embedded.size(1)):
        gates = embedded[:, t, :] @ weight_ih.T + bias_ih + h_t @ weight_hh.T + bias_hh
        i_t = torch.sigmoid(gates[:, :a])
        f_t = torch.sigmoid(gates[:, a:2 * a])
        g_t = torch.tanh(gates[:, 2 * a:3 * a])
        o_t = torch.sigmoid(gates[:, 3 * a:])
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
    return F.linear(h_t, model.fc.weight[:, active_idx], model.fc.bias)

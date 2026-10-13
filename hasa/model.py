"""Slimmable one-layer LSTM language model (paper Sec. III-A).

A width ratio r activates the first floor(r * 256) hidden units: the LSTM gate rows,
the recurrent columns and the output-layer columns are sliced, so a client with a
narrower subnet really computes less.
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


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
    """LSTM forward pass that uses only the hidden units in `active_idx_list`
    (a prefix for HASA / HeteroFL, a random or rolling set for the other baselines).

    The gate rows, the recurrent block and the output columns of those units are cut out
    of the shared weights, and PyTorch's LSTM kernel runs on the slices: the subnet computes
    less, and its gradients flow back into the shared weights."""
    idx = torch.as_tensor(active_idx_list, device=x.device, dtype=torch.long)
    gates = torch.cat([idx + k * hidden_dim for k in range(4)])   # input, forget, cell, output gates
    lstm = model.lstm
    weights = {"weight_ih_l0": lstm.weight_ih_l0[gates],
               "weight_hh_l0": lstm.weight_hh_l0[gates][:, idx],
               "bias_ih_l0": lstm.bias_ih_l0[gates],
               "bias_hh_l0": lstm.bias_hh_l0[gates]}
    h0 = torch.zeros(1, x.size(0), len(idx), device=x.device)
    out, _ = functional_call(_lstm_shell(lstm.input_size, len(idx)), weights, (model.embedding(x), (h0, h0)))
    return F.linear(out[:, -1], model.fc.weight[:, idx], model.fc.bias)


@functools.lru_cache(maxsize=None)
def _lstm_shell(input_size, hidden_size):
    """An LSTM module with no weights of its own (built on the meta device, so creating it
    uses no random numbers); functional_call supplies the sliced weights."""
    return nn.LSTM(input_size, hidden_size, batch_first=True, device="meta")

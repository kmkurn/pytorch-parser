# Copyright 2019 Kemal Kurniawan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Neural syntactic parsers in PyTorch."""

__version__ = '0.0.0'

from typing import Mapping

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import LongTensor, Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscRNNG(nn.Module):
    REDUCE = 0
    SHIFT = 1

    def __init__(
            self,
            word_embedding: nn.Embedding,
            nt_embedding: nn.Embedding,
            action_embedding: nn.Embedding,
            action2nt: Mapping[int, int],
            *,
            stack_size: int = 128,
            n_layers: int = 2,
            hidden_size: int = 128,
            word_dropout: float = 0.5,
            nt_dropout: float = 0.2,
            action_dropout: float = 0.3,
            dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.action2nt = action2nt
        self.word_embedder = nn.Sequential(
            word_embedding,
            nn.Dropout2d(word_dropout),  # drop some words entirely
        )
        self.nt_embedder = nn.Sequential(
            Rearrange('n -> () n'),
            nt_embedding,
            nn.Dropout2d(nt_dropout),  # drop some NTs entirely
            Rearrange('d n dim -> (d n) dim'),  # d==1
        )
        self.action_embedder = nn.Sequential(
            action_embedding,
            nn.Dropout2d(action_dropout),  # drop some actions entirely
        )
        self.buffer2stack_proj = nn.Linear(word_embedding.embedding_dim, stack_size)
        self.nt2stack_proj = nn.Linear(nt_embedding.embedding_dim, stack_size)
        self.subtree_encoders = nn.ModuleList([
            nn.LSTM(stack_size, stack_size, num_layers=n_layers, dropout=dropout),
            nn.LSTM(stack_size, stack_size, num_layers=n_layers, dropout=dropout)
        ])
        self.subtree_mlp = nn.Sequential(
            nn.Linear(2 * stack_size, stack_size),
            nn.ReLU(),
            nn.Linear(stack_size, stack_size),
        )
        self.buffer_guard = nn.Parameter(torch.empty(hidden_size))
        self.buffer_encoder = nn.LSTM(
            word_embedding.embedding_dim, hidden_size, num_layers=n_layers, dropout=dropout)
        self.history_guard = nn.Parameter(torch.empty(hidden_size))
        self.history_encoder = nn.LSTM(
            action_embedding.embedding_dim, hidden_size, num_layers=n_layers, dropout=dropout)
        self.stack_guard = nn.Parameter(torch.empty(hidden_size))
        self.stack_encoder = nn.LSTM(
            stack_size, hidden_size, num_layers=n_layers, dropout=dropout)
        self.action_mlp = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_embedding.num_embeddings),
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in [self.buffer_guard, self.history_guard, self.stack_guard]:
            nn.init.uniform_(p, -0.01, 0.01)

    def forward(self, words: LongTensor, actions: LongTensor) -> Tensor:
        winputs = rearrange(self.word_embedder(words), 'bsz slen wdim -> slen bsz wdim')
        ainputs = rearrange(self.action_embedder(actions), 'bsz alen adim -> alen bsz adim')

        # Init word buffer
        buff = winputs.flip([0])  # reverse sequence
        # shape: (slen, bsz, hdim)
        buff_encoded, _ = self.buffer_encoder(buff)  # precompute encoding
        buff_len = buff.size(0)

        # Init action history
        # shape: (alen, bsz, hdim)
        hist_encoded, _ = self.history_encoder(ainputs)  # precompute encoding
        hist_len = 0

        # Init stack
        stack = []
        stack_open_nt = []

        loss, bsz = 0., buff.size(1)
        for a in rearrange(actions, 'bsz alen -> alen bsz'):
            # Get word buffer state
            if buff_len <= 0:
                dim = self.buffer_guard.size(0)
                buff_state = self.buffer_guard.unsqueeze(0).expand(bsz, dim)
            else:
                buff_state = buff_encoded[buff_len - 1]

            # Get action history state
            if hist_len <= 0:
                dim = self.history_guard.size(0)
                hist_state = self.history_guard.unsqueeze(0).expand(bsz, dim)
            else:
                hist_state = hist_encoded[hist_len - 1]

            # Get stack state
            if not stack:
                dim = self.stack_guard.size(0)
                stack_state = self.stack_guard.unsqueeze(0).expand(bsz, dim)
            else:
                # shape: (len, bsz, sdim)
                inputs = torch.stack(stack)
                # shape: (len, bsz, hdim)
                outputs, _ = self.stack_encoder(inputs)
                # shape: (bsz, hdim)
                stack_state = outputs[-1]

            # Compute action scores
            # shape: (bsz, 3*hdim)
            parser_state = torch.cat([buff_state, hist_state, stack_state], dim=-1)
            # shape: (bsz, 3*hdim)
            parser_state = self.dropout(parser_state)
            # shape: (bsz, n_actions)
            scores = self.action_mlp(parser_state)

            loss += F.cross_entropy(scores, a)

            if a[0].item() == self.REDUCE:
                assert a.eq(self.REDUCE).all(), 'actions must be REDUCE'
                children = []
                while stack_open_nt and not stack_open_nt[-1]:
                    children.append(stack.pop())
                    stack_open_nt.pop()
                assert stack_open_nt, 'cannot REDUCE because no open nonterm'
                parent = stack.pop()
                stack_open_nt.pop()

                # Encode subtree
                inputs_fwd, inputs_bwd = [parent], [parent]
                inputs_fwd.extend(children)
                inputs_bwd.extend(reversed(children))
                # shape: (len, bsz, sdim)
                inputs_fwd = torch.stack(inputs_fwd)
                # shape: (len, bsz, sdim)
                inputs_bwd = torch.stack(inputs_bwd)
                outputs = []
                for inputs, enc in zip([inputs_fwd, inputs_bwd], self.subtree_encoders):
                    out, _ = enc(inputs)
                    outputs.append(out[-1])
                # shape: (bsz, 2*sdim)
                outputs = torch.cat(outputs, dim=-1)
                # shape: (bsz, sdim)
                outputs = self.subtree_mlp(outputs)

                stack.append(outputs)
                stack_open_nt.append(False)
            elif a[0].item() == self.SHIFT:
                assert a.eq(self.SHIFT).all(), 'actions must be SHIFT'
                # shape: (bsz, wdim)
                inputs = buff[buff_len - 1]
                # shape: (bsz, sdim)
                outputs = self.buffer2stack_proj(inputs)
                stack.append(outputs)
                stack_open_nt.append(False)
                buff_len -= 1
            else:
                # shape: (bsz,)
                inputs = torch.empty_like(a)
                for i in range(bsz):
                    inputs[i] = self.action2nt[a[i].item()]
                # shape: (bsz, ntdim)
                outputs = self.nt_embedder(inputs)
                # shape: (bsz, sdim)
                outputs = self.nt2stack_proj(outputs)
                stack.append(outputs)
                stack_open_nt.append(True)

            hist_len += 1

        return loss

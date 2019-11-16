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

from typing import List

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
        self.word_embedder = nn.Sequential(
            word_embedding,
            Rearrange('bsz slen wdim -> slen bsz wdim'),
            nn.Dropout2d(word_dropout),
        )
        self.nt_embedder = nn.Sequential(
            nt_embedding,
            Rearrange('bsz ntlen ntdim -> ntlen bsz ntdim'),
            nn.Dropout2d(nt_dropout),
        )
        self.action_embedder = nn.Sequential(
            action_embedding,
            Rearrange('bsz alen adim -> alen bsz adim'),
            nn.Dropout2d(action_dropout),
        )
        self.buffer2stack_proj = nn.Linear(word_embedding.embedding_dim, stack_size)
        self.nt2stack_proj = nn.Linear(nt_embedding.embedding_dim, stack_size)
        self.subtree_encoders = nn.ModuleList([
            nn.LSTM(stack_size, stack_size, num_layers=n_layers, dropout=dropout),
            nn.LSTM(stack_size, stack_size, num_layers=n_layers, dropout=dropout)
        ])
        self.subtree_proj = nn.Sequential(
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
        self.action_proj = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_embedding.num_embeddings),
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in [self.buffer_guard, self.history_guard, self.stack_guard]:
            nn.init.uniform_(p, -0.01, 0.01)

    def forward(self, words: LongTensor, nonterms: LongTensor, actions: LongTensor) -> Tensor:
        self._init_buff(self.word_embedder(words), self.nt_embedder(nonterms))
        self._init_hist(self.action_embedder(actions))
        self._init_stack()

        loss = 0.
        for a in rearrange(actions, 'bsz alen -> alen bsz'):
            loss += F.cross_entropy(self._get_act_scores(), a)
            if a[0].item() == self.REDUCE:
                assert a.eq(self.REDUCE).all(), 'actions must be REDUCE'
                self._reduce()
            elif a[0].item() == self.SHIFT:
                assert a.eq(self.SHIFT).all(), 'actions must be SHIFT'
                self._shift()
            else:
                self._push_nt()
            self._hist_len += 1

        return loss

    def _init_buff(self, winputs: Tensor, ntinputs: Tensor) -> None:
        # shape: (slen, bsz, wdim)
        self._buff = winputs.flip([0])  # reverse sequence
        self._buff_encoded, _ = self.buffer_encoder(self._buff)  # precompute encoding
        self._buff_len = self._buff.size(0)

        # shape: (ntlen, bsz, ntdim)
        self._ntbuff = ntinputs.flip([0])  # reverse sequence
        self._ntbuff_len = self._ntbuff.size(0)

    def _init_hist(self, inputs: Tensor) -> None:
        # shape: (alen, bsz, adim)
        self._hist_encoded, _ = self.history_encoder(inputs)  # precompute encoding
        self._hist_len = 0

    def _init_stack(self) -> None:
        self._stack = []
        self._stack_open_nt = []

    def _get_act_scores(self) -> Tensor:
        inputs = [self._encode_buff(), self._encode_hist(), self._encode_stack()]
        inputs = [self.dropout(x) for x in inputs]
        return self.action_proj(torch.cat(inputs, dim=-1))

    def _reduce(self) -> None:
        children = []
        while self._stack_open_nt and not self._stack_open_nt[-1]:
            children.append(self._stack.pop())
            self._stack_open_nt.pop()
        assert self._stack_open_nt, 'cannot REDUCE because no open nonterm'
        parent = self._stack.pop()
        self._stack_open_nt.pop()
        self._stack.append(self._encode_subtree(parent, children))
        self._stack_open_nt.append(False)

    def _shift(self) -> None:
        # shape: (bsz, wdim)
        inputs = self._buff[self._buff_len - 1]
        # shape: (bsz, sdim)
        outputs = self.buffer2stack_proj(inputs)

        self._stack.append(outputs)
        self._stack_open_nt.append(False)
        self._buff_len -= 1

    def _push_nt(self) -> None:
        # shape: (bsz, ntdim)
        inputs = self._ntbuff[self._ntbuff_len - 1]
        # shape: (bsz, sdim)
        outputs = self.nt2stack_proj(inputs)

        self._stack.append(outputs)
        self._stack_open_nt.append(True)
        self._ntbuff_len -= 1

    def _encode_buff(self) -> Tensor:
        if self._buff_len <= 0:
            bsz, dim = self._buff.size(1), self.buffer_guard.size(0)
            return self.buffer_guard.unsqueeze(0).expand(bsz, dim)
        # shape: (bsz, hdim)
        return self._buff_encoded[self._buff_len - 1]

    def _encode_hist(self) -> Tensor:
        if self._hist_len <= 0:
            bsz, dim = self._hist_encoded.size(1), self.history_guard.size(0)
            return self.history_guard.unsqueeze(0).expand(bsz, dim)
        # shape: (bsz, hdim)
        return self._hist_encoded[self._hist_len - 1]

    def _encode_stack(self) -> Tensor:
        if not self._stack:
            bsz, dim = self._buff.size(1), self.stack_guard.size(0)
            return self.stack_guard.unsqueeze(0).expand(bsz, dim)

        # shape: (len, bsz, sdim)
        inputs = torch.stack(self._stack)
        # shape: (len, bsz, hdim)
        outputs, _ = self.stack_encoder(inputs)
        # shape: (bsz, hdim)
        return outputs[-1]

    def _encode_subtree(self, parent: Tensor, children: List[Tensor]) -> Tensor:
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
        return self.subtree_proj(outputs)

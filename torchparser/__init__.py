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
from torch import LongTensor, Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscRNNG(nn.Module):
    REDUCE = 0
    SHIFT = 1
    STACK_DIM = 128
    HIDDEN_DIM = 128

    def __init__(
            self,
            word_embedder: nn.Embedding,
            nt_embedder: nn.Embedding,
            action_embedder: nn.Embedding,
    ) -> None:
        super().__init__()
        self.word_embedder = word_embedder
        self.nt_embedder = nt_embedder
        self.action_embedder = action_embedder
        self.buffer2stack_proj = nn.Linear(word_embedder.embedding_dim, self.STACK_DIM)
        self.nt2stack_proj = nn.Linear(nt_embedder.embedding_dim, self.STACK_DIM)
        self.subtree_encoders = nn.ModuleList([
            nn.LSTM(self.STACK_DIM, self.STACK_DIM, num_layers=2),
            nn.LSTM(self.STACK_DIM, self.STACK_DIM, num_layers=2)
        ])
        self.subtree_proj = nn.Sequential(
            nn.Linear(2 * self.STACK_DIM, self.STACK_DIM),
            nn.ReLU(),
            nn.Linear(self.STACK_DIM, self.STACK_DIM),
        )
        self.buffer_guard = nn.Parameter(torch.empty(self.HIDDEN_DIM))
        self.buffer_encoder = nn.LSTM(
            word_embedder.embedding_dim, self.HIDDEN_DIM, num_layers=2)
        self.history_guard = nn.Parameter(torch.empty(self.HIDDEN_DIM))
        self.history_encoder = nn.LSTM(
            action_embedder.embedding_dim, self.HIDDEN_DIM, num_layers=2)
        self.stack_guard = nn.Parameter(torch.empty(self.HIDDEN_DIM))
        self.stack_encoder = nn.LSTM(self.STACK_DIM, self.HIDDEN_DIM, num_layers=2)
        self.action_proj = nn.Sequential(
            nn.Linear(3 * self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, action_embedder.num_embeddings),
        )
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
        self._buff = rearrange(winputs, 'bsz slen wdim -> slen bsz wdim')
        tmp = self._buff.flip([0])  # reverse sequence
        self._buff_encoded, _ = self.buffer_encoder(tmp)  # precompute encoding
        self._buff_len = self._buff.size(0)

        self._ntbuff = rearrange(ntinputs, 'bsz ntlen ntdim -> ntlen bsz ntdim')
        self._ntbuff_len = self._ntbuff.size(0)

    def _init_hist(self, inputs: Tensor) -> None:
        self._hist = rearrange(inputs, 'bsz alen adim -> alen bsz adim')
        self._hist_encoded, _ = self.history_encoder(self._hist)  # precompute encoding
        self._hist_len = 0

    def _init_stack(self) -> None:
        self._stack = []
        self._stack_open_nt = []

    def _get_act_scores(self) -> Tensor:
        x = torch.cat([self._encode_buff(), self._encode_hist(), self._encode_stack()], dim=-1)
        return self.action_proj(x)

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
        inputs = self._buff[-self._buff_len]
        # shape: (bsz, sdim)
        outputs = self.buffer2stack_proj(inputs)

        self._stack.append(outputs)
        self._stack_open_nt.append(False)
        self._buff_len -= 1

    def _push_nt(self) -> None:
        # shape: (bsz, ntdim)
        inputs = self._ntbuff[-self._ntbuff_len]
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
            bsz, dim = self._hist.size(1), self.history_guard.size(0)
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

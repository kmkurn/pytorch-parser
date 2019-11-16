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

from typing import List, Mapping

from einops import rearrange
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
            max_open_nt: int = 100,
    ) -> None:
        super().__init__()
        self.action2nt = action2nt
        self.word_embedding = word_embedding
        self.word_dropout = nn.Dropout2d(word_dropout)
        self.nt_embedding = nt_embedding
        self.nt_dropout = nn.Dropout2d(nt_dropout)
        self.action_embedding = action_embedding
        self.action_dropout = nn.Dropout2d(action_dropout)
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
        self.max_open_nt = max_open_nt
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in [self.buffer_guard, self.history_guard, self.stack_guard]:
            nn.init.uniform_(p, -0.01, 0.01)

    @property
    def n_actions(self) -> int:
        return self.action_embedding.num_embeddings

    def forward(self, words: LongTensor, actions: LongTensor) -> Tensor:
        winputs = self.word_dropout(self.word_embedding(words))
        winputs = rearrange(winputs, 'bsz slen wdim -> slen bsz wdim')
        ainputs = self.action_dropout(self.action_embedding(actions))
        ainputs = rearrange(ainputs, 'bsz alen adim -> alen bsz adim')

        # Init word buffer
        buff = winputs.flip([0])  # reverse sequence
        buff_encoded, _ = self.buffer_encoder(buff)  # precompute encoding
        buff_len = buff.size(0)

        # Init action history
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
                inputs = rearrange(stack, 'len bsz sdim -> len bsz sdim')
                outputs, _ = self.stack_encoder(inputs)
                stack_state = outputs[-1]

            # Compute action scores
            parser_state = rearrange([buff_state, hist_state, stack_state],
                                     'n bsz hdim -> bsz (n hdim)')
            parser_state = self.dropout(parser_state)
            scores = self.action_mlp(parser_state)

            loss += F.cross_entropy(scores, a)

            if a[0].item() == self.REDUCE:
                assert a.eq(self.REDUCE).all(), 'actions must be REDUCE'
                children = []
                while stack_open_nt and not stack_open_nt[-1]:
                    assert stack
                    children.append(stack.pop())
                    stack_open_nt.pop()
                assert stack_open_nt, 'cannot REDUCE because no open nonterm'
                parent = stack.pop()
                stack_open_nt.pop()

                # Encode subtree
                inputs_fwd, inputs_bwd = [parent], [parent]
                inputs_fwd.extend(children)
                inputs_bwd.extend(reversed(children))
                inputs_fwd = rearrange(inputs_fwd, 'len bsz sdim -> len bsz sdim')
                inputs_bwd = rearrange(inputs_bwd, 'len bsz sdim -> len bsz sdim')
                outputs = []
                for inputs, enc in zip([inputs_fwd, inputs_bwd], self.subtree_encoders):
                    out, _ = enc(inputs)
                    outputs.append(out[-1])
                outputs = rearrange(outputs, 'n bsz sdim -> bsz (n sdim)')
                outputs = self.subtree_mlp(outputs)

                stack.append(outputs)
                stack_open_nt.append(False)

            elif a[0].item() == self.SHIFT:
                assert a.eq(self.SHIFT).all(), 'actions must be SHIFT'
                inputs = buff[buff_len - 1]
                outputs = self.buffer2stack_proj(inputs)
                stack.append(outputs)
                stack_open_nt.append(False)
                buff_len -= 1

            else:
                inputs = torch.empty_like(a)
                for i in range(bsz):
                    inputs[i] = self.action2nt[a[i].item()]
                inputs = self.nt_dropout(self.nt_embedding(inputs).unsqueeze(0)).squeeze(0)
                outputs = self.nt2stack_proj(inputs)
                stack.append(outputs)
                stack_open_nt.append(True)

            hist_len += 1

        return loss

    def decode(self, words: LongTensor) -> List[List[int]]:
        winputs = self.word_dropout(self.word_embedding(words))
        winputs = rearrange(winputs, 'bsz slen wdim -> slen bsz wdim')
        buffs = winputs.flip([0])  # reverse sequence
        buffs_encoded, _ = self.buffer_encoder(buffs)  # precompute encoding

        def lift(rnn):
            def lifted(inputs):
                inputs = inputs.unsqueeze(1)
                outputs, (hn, cn) = rnn(inputs)
                outputs = outputs.squeeze(1)
                hn = hn.squeeze(1)
                cn = cn.squeeze(1)
                return outputs, (hn, cn)

            return lifted

        bsz, pred_actions = winputs.size(1), []
        for b in range(bsz):
            # Init word buffer
            buff = buffs[:, b, :]
            buff_encoded = buffs_encoded[:, b, :]
            buff_len = buff.size(0)

            # Init action history
            hist = []

            # Init stack
            stack = []
            stack_open_nt = []
            n_open = 0

            pred_a = []
            while len(stack) != 1 or buff_len > 0:
                # Get word buffer state
                if buff_len <= 0:
                    buff_state = self.buffer_guard
                else:
                    buff_state = buff_encoded[buff_len - 1]

                # Get action history state
                if not hist:
                    hist_state = self.history_guard
                else:
                    inputs = rearrange(hist, 'len adim -> len adim')
                    outputs, _ = lift(self.history_encoder)(inputs)
                    hist_state = outputs[-1]

                # Get stack state
                if not stack:
                    stack_state = self.stack_guard
                else:
                    inputs = rearrange(stack, 'len sdim -> len sdim')
                    outputs, _ = lift(self.stack_encoder)(inputs)
                    stack_state = outputs[-1]

                # Compute action scores
                parser_state = rearrange([buff_state, hist_state, stack_state],
                                         'n hdim -> (n hdim)')
                parser_state = self.dropout(parser_state)
                scores = self.action_mlp(parser_state)

                # Constrain invalid actions
                acts = torch.arange(self.n_actions).to(scores.device)
                nt_mask = (acts != self.SHIFT) & (acts != self.REDUCE)
                if buff_len <= 0 or n_open >= self.max_open_nt:
                    scores.masked_fill_(nt_mask, float('-inf'))
                if buff_len <= 0 or n_open < 1:
                    scores[self.SHIFT] = float('-inf')
                if stack_open_nt and stack_open_nt[-1]:
                    scores[self.REDUCE] = float('-inf')
                if n_open < 2 and buff_len > 0:
                    scores[self.REDUCE] = float('-inf')

                # Get best action with max score
                a = scores.argmax()
                pred_a.append(a.item())

                if a.item() == self.REDUCE:
                    children = []
                    while stack_open_nt and not stack_open_nt[-1]:
                        assert stack
                        children.append(stack.pop())
                        stack_open_nt.pop()
                    assert stack_open_nt, 'cannot REDUCE because no open nonterm'
                    parent = stack.pop()
                    stack_open_nt.pop()
                    n_open -= 1

                    # Encode subtree
                    inputs_fwd, inputs_bwd = [parent], [parent]
                    inputs_fwd.extend(children)
                    inputs_bwd.extend(reversed(children))
                    inputs_fwd = rearrange(inputs_fwd, 'len sdim -> len sdim')
                    inputs_bwd = rearrange(inputs_bwd, 'len sdim -> len sdim')
                    outputs = []
                    for inputs, enc in zip([inputs_fwd, inputs_bwd], self.subtree_encoders):
                        out, _ = lift(enc)(inputs)
                        outputs.append(out[-1])
                    outputs = rearrange(outputs, 'n sdim -> (n sdim)')
                    outputs = self.subtree_mlp(outputs)

                    stack.append(outputs)
                    stack_open_nt.append(False)

                elif a.item() == self.SHIFT:
                    inputs = buff[buff_len - 1]
                    outputs = self.buffer2stack_proj(inputs)
                    stack.append(outputs)
                    stack_open_nt.append(False)
                    buff_len -= 1

                else:
                    inputs = a.new_tensor(self.action2nt[a.item()])
                    inputs = self.nt_dropout(self.nt_embedding(inputs).view(1, 1, -1)).view(-1)
                    outputs = self.nt2stack_proj(inputs)
                    stack.append(outputs)
                    stack_open_nt.append(True)
                    n_open += 1

                hist.append(self.action_embedding(a))

            pred_actions.append(pred_a)

        return pred_actions

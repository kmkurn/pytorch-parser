import torch
import torch.nn as nn

from torchparser import DiscRNNG


class TestForward:
    def test_ok(self):

        n_words, n_nonterms, n_actions = 10, 11, 12
        word_size, act_size, nt_size = 13, 14, 15
        word_embedding = nn.Embedding(n_words, word_size)
        nt_embedding = nn.Embedding(n_nonterms, nt_size)
        action_embedding = nn.Embedding(n_actions, act_size)
        # vocab(NT): S=0 NP=1 VP=2 ADJP=3
        # vocab(action): R=0 SH=1 NT(S)=2 NT(NP)=3 NT(VP)=4 NT(ADJP)=5
        action2nt = {2: 0, 3: 1, 4: 2, 5: 3}
        parser = DiscRNNG(word_embedding, nt_embedding, action_embedding, action2nt)
        # (S (NP john) (VP loves (NP mary)))
        # (S (NP john) (VP is (ADJP cool)))
        # NT(S) NT(NP) SH R NT(VP) SH NT(NP) SH R R R
        # NT(S) NT(NP) SH R NT(VP) SH NT(ADJP) SH R R R
        words = torch.randint(n_words, (2, 3))
        actions = torch.tensor([[2, 3, 1, 0, 4, 1, 3, 1, 0, 0, 0],
                                [2, 3, 1, 0, 4, 1, 5, 1, 0, 0, 0]])

        loss = parser(words, actions)

        assert torch.is_tensor(loss)
        loss.backward()
        for p in parser.parameters():
            assert not p.requires_grad or p.grad is not None

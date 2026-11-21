"""
model.py
--------
The two-branch 1D CNN (simplified AstroNet, Shallue & Vanderburg 2018).

    global view (2001) -> [conv-relu-pool] x4 --\
                                                 concat -> dense(64) -> relu -> dropout -> dense(1) = logit
    local view  (201)  -> [conv-relu-pool] x2 --/

Every block is: Conv1d(kernel 5, 'same' padding) -> ReLU -> MaxPool(2).
Shapes are written next to each line so you can follow the tensor through the network.
"""

import torch
import torch.nn as nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),   # length unchanged
        nn.ReLU(),
        nn.MaxPool1d(2),                                     # length halves
    )


class AstroNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        # global branch: 2001 -> 1000 -> 500 -> 250 -> 125 points
        self.global_branch = nn.Sequential(
            block(1, 16), block(16, 16), block(16, 32), block(32, 32))
        # local branch: 201 -> 100 -> 50 points
        self.local_branch = nn.Sequential(
            block(1, 16), block(16, 32))

        n_features = 32 * 125 + 32 * 50          # = 4000 + 1600 = 5600
        self.head = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),                     # one number: the logit
        )

    def forward(self, g, l):
        # g: (batch, 2001), l: (batch, 201)  ->  add a channel axis for Conv1d
        g = self.global_branch(g.unsqueeze(1)).flatten(1)    # (batch, 4000)
        l = self.local_branch(l.unsqueeze(1)).flatten(1)     # (batch, 1600)
        z = torch.cat([g, l], dim=1)                         # (batch, 5600)
        return self.head(z).squeeze(1)                       # (batch,) logits


if __name__ == "__main__":
    m = AstroNet()
    n = sum(p.numel() for p in m.parameters())
    print(m)
    print(f"trainable parameters: {n:,}")
    print(m(torch.zeros(3, 2001), torch.zeros(3, 201)).shape)

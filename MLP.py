from torch import nn


class ColorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.Sigmoid(),
            nn.Linear(256, 64), nn.Sigmoid(),
            nn.Linear(64, 3), nn.Sigmoid())

    def forward(self, x): return self.net(x)
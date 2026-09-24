import torch.nn as nn

from katago_train.encode import policy_size


class Net(nn.Module):
    def __init__(self, size=9, channels=32):
        super().__init__()
        self.size = size
        self.conv = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.Flatten(),
            nn.Linear(2 * size * size, policy_size(size)),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Flatten(),
            nn.Linear(size * size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.conv(x)
        return self.policy_head(h), self.value_head(h)
import torch
import torch.nn as nn
import torch.nn.functional as F


class cifar_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        # 3×32×32 → 32×32×32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)

        # 32×32×32 → 64×16×16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)

        # 64×16×16 → 128×8×8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(16, 128)

        # classifier for binary output
        self.fc = nn.Linear(128 * 8 * 8, 1)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)    # 32×32 → 16×16

        x = F.relu(self.gn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)    # 16×16 → 8×8

        x = F.relu(self.gn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # flatten
        x = torch.sigmoid(self.fc(x))
        return x

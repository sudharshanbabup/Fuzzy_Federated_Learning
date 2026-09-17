"""Compact convolutional models used by the federated clients."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """2-block CNN for 28x28 single-channel inputs (Fashion-MNIST / MNIST)."""

    def __init__(self, num_classes: int = 10, in_ch: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CIFARCNN(nn.Module):
    """3-block CNN with GroupNorm and LeakyReLU for 32x32 RGB inputs.

    GroupNorm is used instead of BatchNorm because BatchNorm running statistics
    are themselves an attack surface and are not well defined under non-IID
    federated averaging.  LeakyReLU is used instead of ReLU because a single
    amplified Byzantine update early in training can otherwise drive the network
    into a constant-logit state with zero gradient, from which no aggregation
    rule can recover; the resulting chance-level accuracy would be an artefact of
    the activation rather than a property of the defence.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2),
            )

        self.b1 = block(3, 32)
        self.b2 = block(32, 64)
        self.b3 = block(64, 128)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 128),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.b3(self.b2(self.b1(x))))


def build_model(dataset: str) -> nn.Module:
    if dataset in ("fmnist", "mnist"):
        return SmallCNN(10, 1)
    if dataset == "cifar10":
        return CIFARCNN(10)
    raise ValueError(f"unknown dataset {dataset}")


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

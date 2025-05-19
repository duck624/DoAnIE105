# File: ~/FedMIA/models/alexnet.py
'''
AlexNet for MNIST and CIFAR100. Adjusted for grayscale (MNIST) and RGB (CIFAR100).
Without BN, the start learning rate should be 0.01.
Adapted from YANG, Wei
'''
import torch
import torch.nn as nn

__all__ = ['alexnet', 'AlexNet']

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, droprate=0):
        super(AlexNet, self).__init__()
        in_channels = 1 if num_classes == 10 else 3
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        fc_input_size = 256 * 3 * 3 if num_classes == 10 else 256 * 4 * 4
        if droprate > 0.:
            self.fc = nn.Sequential(
                nn.Dropout(droprate),
                nn.Linear(fc_input_size, num_classes)
            )
        else:
            self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Adapted for MNIST and CIFAR100.
    """
    model = AlexNet(**kwargs)
    return model

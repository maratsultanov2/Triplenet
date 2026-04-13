"""
Triplenet — первая Multi-Head архитектура, предшественник TAT-7.
Сохранена как исторический артефакт.
"""

import torch
import torch.nn as nn

class Triplenet(nn.Module):
    """Минимальная версия Triplenet (3 головы)."""
    
    def __init__(self):
        super().__init__()
        self.heads = 3
        self.path1 = nn.Sequential(nn.Linear(784, 128), nn.ReLU())
        self.path2 = nn.Sequential(nn.Linear(784, 128), nn.Tanh())
        self.path3 = nn.Sequential(nn.Linear(784, 128), nn.LeakyReLU())
        self.classifier = nn.Linear(128 * 3, 10)
    
    def forward(self, x):
        y1 = self.path1(x)
        y2 = self.path2(x)
        y3 = self.path3(x)
        combined = torch.cat([y1, y2, y3], dim=1)
        return self.classifier(combined)

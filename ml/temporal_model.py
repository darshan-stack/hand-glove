"""Small GRU model for dynamic gesture sequences.

Input shape: [batch, time, channels]. Designed to be small enough to test on
CPU before Raspberry Pi deployment.
"""
import torch
from torch import nn


class GestureGRU(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=1, num_classes=8, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1])


def probabilities(model, x):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x), dim=-1)

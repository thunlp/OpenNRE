import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, input_size=50, hidden_size=256, dropout=0, kernel_size=3, padding=1, activation_function=F.relu):
        """
        Args:
            input_size: dimention of input embedding
            kernel_size: kernel_size for CNN
            padding: padding for CNN
            hidden_size: hidden size
        """
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.act = activation_function
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            input features: (B, L, I_EMBED)
        Return:
            output features: (B, H_EMBED)
        """
        # Check size of tensors
        x = x.transpose(1, 2) # (B, I_EMBED, L)
        x = self.conv(x) # (B, H_EMBED, L)
        x = self.act(x) # (B, H_EMBED, L)
        x = self.dropout(x) # (B, H_EMBED, L)
        x = x.transpose(1, 2) # (B, L, H_EMBED)
        return x

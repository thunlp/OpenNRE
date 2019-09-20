import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaxPool(nn.Module):

    def __init__(self, kernel_size, segment_num=None):
        """
        Args:
            input_size: dimention of input embedding
            kernel_size: kernel_size for CNN
            padding: padding for CNN
        hidden_size: hidden size
        """
        super().__init__()
        self.segment_num = segment_num
        if self.segment_num != None:
            self.mask_embedding = nn.Embedding(segment_num + 1, segment_num)
            self.mask_embedding.weight.data.copy_(torch.FloatTensor(np.concatenate([np.zeros((1, segment_num)), np.identity(segment_num)], axis=0)))
            self.mask_embedding.weight.requires_grad = False
            self._minus = -100
        self.pool = nn.MaxPool1d(kernel_size)

    def forward(self, x, mask=None):
        """
        Args:
            input features: (B, L, I_EMBED)
        Return:
            output features: (B, H_EMBED)
        """
        # Check size of tensors
        if mask is None or self.segment_num is None or self.segment_num == 1:
            x = x.transpose(1, 2) # (B, L, I_EMBED) -> (B, I_EMBED, L)
            x = self.pool(x).squeeze(-1) # (B, I_EMBED, 1) -> (B, I_EMBED)
            return x
        else:
            B, L, I_EMBED = x.size()[:3]
            # mask = 1 - self.mask_embedding(mask).transpose(1, 2).unsqueeze(2) # (B, L) -> (B, L, S) -> (B, S, L) -> (B, S, 1, L)
            # x = x.transpose(1, 2).unsqueeze(1) # (B, L, I_EMBED) -> (B, I_EMBED, L) -> (B, 1, I_EMBED, L)
            # x = (x + self._minus * mask).contiguous().view([-1, I_EMBED, L]) # (B, S, I_EMBED, L) -> (B * S, I_EMBED, L)
            # x = self.pool(x).squeeze(-1) # (B * S, I_EMBED, 1) -> (B * S, I_EMBED)
            # x = x.view([B, -1])  # (B, S * I_EMBED)
            # return x
            mask = 1 - self.mask_embedding(mask).transpose(1, 2)
            x = x.transpose(1, 2)
            pool1 = self.pool(x + self._minus * mask[:, 0:1, :])
            pool2 = self.pool(x + self._minus * mask[:, 1:2, :])
            pool3 = self.pool(x + self._minus * mask[:, 2:3, :])

            x = torch.cat([pool1, pool2, pool3], 1)
            # x = x.squeeze(-1)
            return  x

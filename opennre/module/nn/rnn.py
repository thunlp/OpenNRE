import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, input_size=50, hidden_size=256, dropout=0, bidirectional=False, num_layers=1, activation_function="tanh"):
        """
        Args:
            input_size: dimention of input embedding
            hidden_size: hidden size
            dropout: dropout layer on the outputs of each RNN layer except the last layer
            bidirectional: if it is a bidirectional RNN
            num_layers: number of recurrent layers
            activation_function: the activation function of RNN, tanh/relu
        """
        super().__init__()
        if bidirectional:
            hidden_size /= 2
        self.rnn = nn.RNN(input_size, 
                          hidden_size, 
                          num_layers, 
                          nonlinearity=activation_function, 
                          dropout=dropout, 
                          bidirectional=bidirectional)

    def forward(self, x):
        """
        Args:
            input features: (B, L, I_EMBED)           
        Return:
            output features: (B, L, H_EMBED)
        """
        # Check size of tensors
        x = x.transpose(0, 1) # (L, B, I_EMBED)
        x, h = self.rnn(x) # (L, B, H_EMBED)
        x = x.transpose(0, 1) # (B, L, I_EMBED)
        return x

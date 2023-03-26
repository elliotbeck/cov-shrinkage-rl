# libraries
from torch import nn, flatten
from torch.nn import init
import torch.nn.functional as F

# define critic network


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=2, batch_first=True)
        self.fc_value = nn.Linear(hidden_dim, output_dim)
        # init.xavier_normal_(self.lstm.weight)
        # init.xavier_normal_(self.fc_value.weight)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(x[:, -1, :])
        value = self.fc_value(x)
        return value

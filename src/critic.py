# libraries
from torch import nn
from torch.nn import init
import torch.nn.functional as F

# define critic network
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(Critic, self).__init__()
        self.fc_hidden = nn.Linear(input_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, output_dim)
        init.xavier_normal_(self.fc_hidden.weight)
        init.xavier_normal_(self.fc_value.weight)

    def forward(self, x):
        hidden = F.leaky_relu(self.fc_hidden(x))
        value = self.fc_value(hidden)
        return value
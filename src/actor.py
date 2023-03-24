# import libraries
from torch import nn, flatten
from torch.nn import init
import torch.nn.functional as F


# set up the actor
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim)
        # init.xavier_normal_(self.lstm.weight)
        #  init.xavier_normal_(self.fc_mu.weight)
        #  init.xavier_normal_(self.fc_sigma.weight)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(x[:, -1, :])
        # x = F.dropout(x, training=self.training)
        mu = F.softplus(self.fc_mu(x)) + 1e-5
        sigma = F.softplus(self.fc_sigma(x)) + 1e-5
        return mu, sigma

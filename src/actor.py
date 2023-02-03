# import libraries
from torch import nn, flatten
from torch.nn import init
import torch.nn.functional as F


# set up the actor
class Actor(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=1):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=3, 
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=3, 
                               out_channels=3, 
                               kernel_size=5)
        self.conv2_drop = nn.Dropout2d() # TODO: check if this is correct, training flag?
        self.fc_hidden = nn.LazyLinear(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim)
        # init.xavier_normal_(self.fc_hidden.weight)
        init.xavier_normal_(self.fc_mu.weight)
        init.xavier_normal_(self.fc_sigma.weight)
        init.xavier_normal_(self.conv1.weight)
        init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = flatten(x.view(x.shape[0],-1))
        x = F.relu(self.fc_hidden(x))
        # x = F.dropout(x, training=self.training)
        mu = F.softplus(self.fc_mu(x)) + 1e-5
        sigma = F.softplus(self.fc_sigma(x)) + 1e-5
        return mu, sigma

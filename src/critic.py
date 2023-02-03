# libraries
from torch import nn, flatten
from torch.nn import init
import torch.nn.functional as F

# define critic network
class Critic(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=1):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=5, 
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=5, 
                               out_channels=10, 
                               kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc_hidden = nn.LazyLinear(hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, output_dim)
        # init.xavier_normal_(self.fc_hidden.weight)
        init.xavier_normal_(self.fc_value.weight)
        init.xavier_normal_(self.conv1.weight)
        init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = flatten(x.view(x.shape[0],-1))
        x = F.relu(self.fc_hidden(x))
        value = self.fc_value(x)
        return value

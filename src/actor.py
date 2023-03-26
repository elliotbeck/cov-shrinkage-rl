# import libraries
from torch import nn, flatten
from torch.nn import init
import torch.nn.functional as F


# set up the actor
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2)
        # self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(x[:, -1, :])
        # x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

# libraries
from torch import nn

# define critic network
class Critic(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(Critic, self).__init__()
        # TODO: change the network architecture to conv net input
        self.net = nn.Sequential(nn.Linear(input_shape, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size,hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, 1))
    
    def forward(self,x):
        x = self.net(x)
        return x
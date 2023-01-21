# import libraries
import torch.nn as nn

# define actor network      
class Actor(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size):
        super(Actor, self).__init__()
        # TODO: change the network architecture to conv net input
        self.net = nn.Sequential(nn.Linear(input_shape, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size,hidden_size),
                                 nn.ReLU(),
                                 )
        self.mean = nn.Sequential(nn.Linear(hidden_size, output_shape),
                                  nn.Tanh())                    # tanh squashed output to the range of -1..1
        self.variance =nn.Sequential(nn.Linear(hidden_size, output_shape),
                                     nn.Softplus())             # log(1 + e^x) has the shape of a smoothed ReLU
    
    def forward(self,x):
        x = self.net(x)
        return self.mean(x), self.variance(x) 
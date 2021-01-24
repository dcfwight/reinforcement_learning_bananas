# import the Pytorch submodules required to create neural network
import torch.nn as nn
import torch.nn.functional as F

# create a deep network

class QNetwork(nn.Module):
    """Actor (Policy Model)"""
    
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64, fc3_units=16, dropout=0.5):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, state):
        """Map from state to action values"""
        x = self.fc1(state)
        x = F.relu(x)
#         x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
#         x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
#         x = self.dropout(x)
        x = self.fc4(x)
        # do we have softmax here?
        # do we need dropout?
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv

class SimpleMLP(nn.Module):
    # input_dim = dim of node features
    def __init__(self, input_dim, hidden_dim=32):
        super(SimpleMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_hat = self.linear3(x)
        y_hat = y_hat.squeeze(-1)
        return y_hat
    
class SimpleLSTM(nn.Module):
    def __init__(self,input_dim, hidden_dim=32):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True,num_layers=1)   # D_in: input.dim of 1 LSTM, H: dim of Hidden state vector
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self,x):
        # hidden = long-term memory, cell = short-term memory
        output,(hidden,cell) = self.lstm(x)
        x = F.relu(self.linear1(output[:,:,:]))    # take last time steps of the output
        y_hat = self.linear2(x)
        y_hat = y_hat.squeeze(-1)
        return y_hat
    
class SimpleLSTM_encoder(nn.Module):
    def __init__(self,input_dim, hidden_dim=32):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True,num_layers=1)   # D_in: input.dim of 1 LSTM, H: dim of Hidden state vector
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 128)

    def forward(self,x):
        # hidden = long-term memory, cell = short-term memory
        output,(hidden,cell) = self.lstm(x)
        x = F.relu(self.linear1(output[:,-1,:]))    # take last time steps of the output
        y_hat = self.linear2(x)
        y_hat = y_hat.squeeze(-1)
        return y_hat

class SimpleGCN(nn.Module):
    # input_dim = dim of node features
    # output_dim = number of class which is ON/OFF
    def __init__(self, input_dim, hidden_dim, edge_index, output_dim=1):
        super(SimpleGCN, self).__init__()
        self.edge_index = edge_index
        # GCN layers
        self.gcn_conv1 = GCNConv(input_dim,hidden_dim)
        # Linear layers
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.gcn_conv1(x,self.edge_index))
        x = F.relu(self.linear1(x))    # take last time steps of the output
        y_hat = self.linear2(x)
        y_hat = y_hat.squeeze(-1)
        return y_hat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import GCNConv

class SimpleModel01(nn.Module):
    # input_dim = dim of node features
    # output_dim = number of class which is ON/OFF
    def __init__(self, adj_mat, input_dim, hidden_dim=0, output_dim=2):
        super(SimpleModel01, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        self.adj_mat = adj_mat
        self.edge_index = self.adj_mat.to_sparse().indices().to(device)
        #print(self.edge_index)
        
        # GCN layers
        self.gcn_conv1 = GCNConv(input_dim,output_dim)

    def forward(self, x):
        x = self.gcn_conv1(x,self.edge_index)
        y_hat = F.log_softmax(x)
        return y_hat

class SimpleModel02(nn.Module):
    # input_dim = dim of node features
    # output_dim = number of class which is ON/OFF
    def __init__(self, adj_mat, input_dim, hidden_dim, output_dim=2):
        super(SimpleModel02, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        self.adj_mat = adj_mat
        self.edge_index = self.adj_mat.to_sparse().indices().to(device)
        
        # GCN layers
        self.gcn_conv1 = GCNConv(input_dim,hidden_dim)
        
        # Linear layers
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.gcn_conv1(x,self.edge_index)
        x = F.relu(x)
        y_hat = F.log_softmax(self.linear(x))
        return y_hat

class SimpleModel03(nn.Module):
    # input_dim = dim of node features
    # output_dim = number of class which is ON/OFF
    def __init__(self, adj_mat, input_dim, hidden_dim, output_dim=2):
        super(SimpleModel03, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        self.adj_mat = adj_mat
        self.edge_index = self.adj_mat.to_sparse().indices().to(device)
        
        # GCN layers
        self.gcn_conv1 = GCNConv(input_dim,hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim,hidden_dim)
        
        # Linear layers
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.gcn_conv1(x,self.edge_index)
        x = F.relu(x)
        x = self.gcn_conv2(x,self.edge_index)
        x = F.relu(x)
        y_hat = F.log_softmax(self.linear(x))
        return y_hat
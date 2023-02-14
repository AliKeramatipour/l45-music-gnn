import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import slice_temporal_data, create_tonnetz_adjacency_matrix
from models import SimpleModel01, SimpleModel02, SimpleModel03

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

epochs = 100
lr = 0.001

def update_stats(training_stats, epoch_stats):
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key,val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats

def train_gnn(X, y, mask, model, optimiser):
    model.train()
    optimiser.zero_grad()
    y_hat = model(X)[mask]
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimiser.step()
    return loss.data

def evaluate_gnn(X, y, mask, model):
    model.eval()
    y_hat = model(X)[mask]
    y_hat = y_hat.data.max(1)[1]
    num_correct = y_hat.eq(y.data).sum()
    accuracy = 100.0 * (num_correct/len(y))
    return accuracy
    
# Training loop
def train_eval_loop_gnn(model, train_x, train_y, train_mask, valid_x, valid_y, valid_mask):
    optimiser = optim.Adam(model.parameters(), lr=lr)
    training_stats = None
    # Training loop
    for epoch in range(epochs):
        train_loss = train_gnn(train_x, train_y, train_mask, model, optimiser)
        train_acc = evaluate_gnn(train_x, train_y, train_mask, model)
        valid_acc = evaluate_gnn(valid_x, valid_y, valid_mask, model)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f} validation accuracy: {valid_acc:.3f}")
        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch':epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    return training_stats

if __name__ == '__main__':
    # High parameters for data structure
    music_len = 100
    time_steps = 5
    number_nodes = 12

    # Generate Data
    # Toy music example
    toy_music = (torch.rand(music_len,number_nodes)<0.5).to(torch.float)
    data = slice_temporal_data(toy_music,window_size=5)
    print("Input: ",data[0][0])
    print("Output: ",data[0][1])
    # Toy adjacency matrix for testing
    toy_adj_mat = (torch.rand(number_nodes,number_nodes)<0.5).to(torch.float)
    print("Adjancy Matrix: ",toy_adj_mat)
    print("Edge_index: ",toy_adj_mat.to_sparse())

    data_X = torch.stack([d[0] for d in data])
    y = torch.stack([d[1] for d in data])
    #print(y.shape)
    data_Y = np.zeros((y.shape[0],y.shape[1],2))
    for i,nodes in enumerate(y):
        for j,node in enumerate(nodes):
            data_Y[i][j][int(node)] = 1.0
    data_Y = torch.tensor(data_Y)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.33, random_state=42)
    print(X_train.shape)
    print(y_train.shape)

    
    # Training and Evaluation
    #model = ...

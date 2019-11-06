import torch 
import torch.nn.functional as F 
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
import pdb

from torch_geometric.data import Data
edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3], [1, 2, 0, 2, 0, 1, 3, 2]], dtype=torch.long) 
inputX = torch.tensor([[-1], [0], [1], [1]], dtype=torch.float)
print (type (inputX))
labelY = torch.tensor([0, 0, 1, 1], dtype=torch.long)
train_mask = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
test_mask = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
val_mask = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
print (type (labelY))
dataset = Data(edge_index=edge_index, test_mask=test_mask, train_mask=train_mask, val_mask=val_mask, x=inputX, y=labelY)
print (dataset)
dataset.num_classes=2

from torch_geometric.datasets import Planetoid 
dataset = Planetoid(root='/tmp/Cora', name='Cora')

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        return aggr_out


class Net(torch.nn.Module): 
    def __init__(self): 
        super(Net, self).__init__() 
        self.conv1 = EdgeConv(dataset.num_node_features, 16) 
        self.conv2 = EdgeConv(16, dataset.num_classes) 
    
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index
#        pdb.set_trace()
        x = self.conv1(x, edge_index) 
        x = F.relu(x) 
#        x = F.dropout(x, training=self.training) 
        x = self.conv2(x, edge_index) 
        return F.log_softmax(x, dim=1)

device = torch.device('cpu') 
model = Net().to(device) 
data = dataset[0].to(device) 
print (data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) 
model.train() 
for epoch in range(200): 
    optimizer.zero_grad() 
#    pdb.set_trace()
    out = model(data)
#    print (model.conv1.weight)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) 
    loss.backward() 
    print (loss)
    optimizer.step()
    

model.eval() 
_, pred = model(data).max(dim=1) 
correct = float (pred[data.test_mask].eq(
                     data.y[data.test_mask]).sum().item()) 
acc = correct / data.test_mask.sum().item() 
print('Accuracy: {:.4f}'.format(acc)) 

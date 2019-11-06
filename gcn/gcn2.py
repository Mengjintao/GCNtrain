import sys
import inspect
import torch 
import time
import torch.nn.functional as F 
#from torch_geometric.nn import GCNConv 
import pdb

from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.utils import scatter_
#from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset

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
#dataset = Planetoid(root='/tmp/Cora', name='Cora')
dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
#dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')

class GCNConv(torch.nn.Module):
#class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(GCNConv, self).__init__()
#        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached   = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        size = [None, None]
        dim = 0
        i   = 1
        size[0] = x.size(dim)
        size[1] = size[0]
        
        tmp = torch.index_select(x, 0, edge_index[0])
        message_args = []
        message_args.append(tmp)
        message_args.append(norm)
        out = self.message(*message_args)
        out = scatter_('add', out, edge_index[i], dim_size=size[i])
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class Net(torch.nn.Module): 
    def __init__(self): 
        super(Net, self).__init__() 
        self.conv1 = GCNConv(dataset.num_node_features, 128) 
        self.conv2 = GCNConv(128, dataset.num_classes) 
    
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

gcntime = 0
losstime = 0
lossbacktime = 0
opttime  = 0


for epoch in range(200): 
    optimizer.zero_grad() 
#    pdb.set_trace()
    t0 = time.time()
    out = model(data)
    t1 = time.time()
    gcntime += t1-t0
    print (model.conv1.weight)
    t0 = time.time()
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) 
    t1 = time.time()
    losstime += t1-t0
    t0 = time.time()
    loss.backward() 
    t1 = time.time()
    lossbacktime += t1-t0
    print (loss)
    t0 = time.time()
    optimizer.step()
    t1 = time.time()
    opttime += t1-t0
   
print gcntime
print losstime
print lossbacktime
print opttime
model.eval() 
_, pred = model(data).max(dim=1) 
correct = float (pred[data.test_mask].eq(
                     data.y[data.test_mask]).sum().item()) 
acc = correct / data.test_mask.sum().item() 
print('Accuracy: {:.4f}'.format(acc)) 

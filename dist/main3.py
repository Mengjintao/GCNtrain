from __future__ import division, print_function
from random import Random
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from random import randint
from time import sleep
import torch
from math import ceil
import torch.nn.functional as F
from torch import distributed, nn
from torch.utils import data
from torchvision import datasets, transforms
from torch.autograd import Variable
import time

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(batch_size):
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = distributed.get_world_size()
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(distributed.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size= batch_size, shuffle=True)
    return train_set

class MNISTDataLoader(data.DataLoader):
    def __init__(self, root, batch_size, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        dataset = datasets.MNIST(root, train=train, transform=transform, download=True)
        print ("MNIST = %d" %len(dataset))

        sampler = None
        if train and distributed_is_initialized():
            sampler = data.DistributedSampler(dataset)

        super(MNISTDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
        )

def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False

class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    def update(self, output, target):
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)


class Trainer(object):

    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def fit(self, epochs, batch_size):

        tottime = 0
        for epoch in range(1, epochs + 1):
            begin = time.time()
            train_loss, train_acc = self.train(batch_size = batch_size)
            end = time.time()
            tottime += (end - begin)
            test_loss, test_acc = self.evaluate()
            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
            )
        print (tottime)

    def train(self, batch_size):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        num_batches = ceil( len(self.train_loader.dataset) / float(batch_size) )
        epoch_loss = 0
        for data, target in self.train_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            loss = F.cross_entropy(output, target)
            epoch_loss += loss.data
            self.optimizer.zero_grad()
            loss.backward()
           
            local_size  = float(distributed.get_world_size())
            local_ranks = list(range(int(local_size)))
            for param in self.model.parameters():
                distributed.all_reduce(tensor=param.grad.data, op=distributed.ReduceOp.SUM)
                param.grad.data /= local_size
            self.optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, target)
        print('Rank ', distributed.get_rank(), ', epoch ', ': ', epoch_loss / num_batches)

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = F.cross_entropy(output, target)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, target)

        return test_loss, test_acc

class Net(nn.Module):
    """ Network architecture. """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def run2(args):
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    train_set = partition_dataset(batch_size=args.batch_size)

    model = Net()

    if distributed_is_initialized():
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        model = nn.DataParallel(model)
        model.to(device)

#    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.5)
    
    num_batches = ceil( len(train_set.dataset) / float(args.batch_size) )
    print ("hello world")
    print (num_batches)
    print (len(train_set.dataset))

    test_loader  = MNISTDataLoader(args.root, args.batch_size, train=False)
    trainer = Trainer(model, optimizer, train_set, test_loader, device)
    trainer.fit(args.epochs, batch_size=args.batch_size)

    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.data
            loss.backward()
            
            local_size = float(distributed.get_world_size())
            local_ranks = list(range(int(local_size)))
            for param in model.parameters():
                distributed.all_reduce(tensor=param.grad.data, op=distributed.ReduceOp.SUM)
                param.grad.data /= local_size
            optimizer.step()
        print('Rank ', distributed.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        distributed.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
        )

    run2(args)

if __name__ == '__main__':
    main()

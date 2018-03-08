#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 18:00:56 2018

@author: twweng

The basic script of using pytorch to train a 2-layer MLP. 

"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
# F is a library of functions, e.g. the loss function, the relu, etc.
import torch.nn.functional as F 
import torch.optim as optim
# transforms can transform the input data (for each image)
from torchvision import datasets, transforms
from torch.autograd import Variable


## define nn model (the cnn model)
## we can actually use two nn.Sequential to define the convolution layer and then FC layer (the linear layer with ReLU)
## e.g. the AlexNet in torchvision: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#        self.conv2_drop = nn.Dropout2d()
#        self.fc1 = nn.Linear(320, 50)
#        self.fc2 = nn.Linear(50, 10)

#    def forward(self, x):
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#        x = x.view(-1, 320)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        x = self.fc2(x)
#        return F.log_softmax(x, dim=1)


## define training function
def train(epoch):
    model.train()
    print('size of train_loader = {}, N = {}'.format(len(train_loader), N))
    for batch_idx, (data, target) in enumerate(train_loader):
        #print('batch_idx = {}'.format(batch_idx))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        # Because our current model is linear, so we need to flatten the data to (batch_size, 784). For the "Net" model, they use convolutional layer first, so their input is (batch_size, 1,28,28)
        # Alternatively, we can add a lambda function in the transforms of the DataLoader (e.g. add such a line: transforms.Lambda(lambda x: x.view(-1)) to reshape each image) for our model)
        if batch_idx == len(train_loader)-1: # the last batch
            #print('data size = {}, target = {}'.format(data.shape, target.shape)) 
            # the last batch of data will not be batch_size if total number of data is not a multiple of batch size, so let data.view to decide first 
            data = data.view(-1,784) # 784 for mnist
        else:
            data = data.view(N,-1)
        
        # Forward pass: output is the predicted output, target is the true label
        output = model(data)
        # compute the loss
        loss = loss_fn(output,target)
        ## loss = F.nll_loss(output, target)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

## define test function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
    
        data, target = Variable(data, volatile=True), Variable(target)
        
        data = data.view(1000,-1)
        output = model(data)
        
        if args.loss == 'nll_loss':
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        elif args.loss == 'cross_entropy':
            test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print("length", len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    print('\nTest loss = {}'.format(test_loss))
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    print('Accuracy: {}/{}'.format(correct, len(test_loader.dataset)))
    print('percentage ({:.0f}%)\n'.format(100 * int(correct) / len(test_loader.dataset)))



## main function
if __name__ == "__main__":
    
    
    # parse the training settings: 
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--loss',choices=['nll_loss','cross_entropy'], default='cross_entropy')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # seed, argument setting
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
     
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    ## download dataset 
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = args.batch_size, 784, 100, 10
    
    if args.loss == 'nll_loss':
        print('using nll_loss!')
        loss_fn = nn.NLLLoss()
        model = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                nn.Linear(H, D_out),
                nn.LogSoftmax(dim=1)
                )
    elif args.loss == 'cross_entropy':
        print('using cross_entropy loss')
        loss_fn = nn.CrossEntropyLoss()
        model = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(),
                nn.Linear(H, D_out),
                nn.Softmax(dim=1)
                )

    if args.cuda:
        model.cuda() 
    
    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Variables it should update.
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
           
    for epoch in range(1, args.epochs+1):
        train(epoch)
        test()


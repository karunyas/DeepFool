#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 18:00:56 2018

@author: twweng

rewrite the training code using nn to define module

"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable



## define nn model
class Net(nn.Module):
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
        return F.log_softmax(x, dim=1)


## define training function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        # for current use, previous didn't have this line
        data = data.view(100,-1)
        output = model(data)
        ## loss = F.nll_loss(output, target)
        
        loss = loss_fn(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            #print('Target = {}'.format(target))
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
        #test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        
        if args.loss == 'nll_loss':
            test_loss += F.nll_loss(output, target, size_average=False).data[0]
        elif args.loss == 'cross_entropy':
            test_loss += F.cross_entropy(output, target, size_average=False).data[0]
        
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest loss = {}'.format(test_loss))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
    N, D_in, H, D_out = 100, 784, 100, 10
    
    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
#    x = Variable(torch.randn(N, D_in))
#    y = Variable(torch.randn(N, D_out), requires_grad=False)
    
    # Use the nn package to define our model and loss function.
   
#   model = nn.Sequential(
#              nn.Linear(D_in, H),
#              nn.ReLU(),
#              nn.Linear(H, D_out),
#              nn.Softmax(dim=1)
#            )
    #loss_fn = torch.nn.MSELoss(size_average=False)
   
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




   # loss_fn = nn.NLLLoss()

   # loss_fn = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda() 
    
    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Variables it should update.
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
        
#    for t in range(500):
#        # Forward pass: compute predicted y by passing x to the model.
#        y_pred = model(x)
#    
#        # Compute and print loss.
#        loss = loss_fn(y_pred, y)
#        print(t, loss.data[0])
#      
#        # Before the backward pass, use the optimizer object to zero all of the
#        # gradients for the variables it will update (which are the learnable weights
#        # of the model)
#        optimizer.zero_grad()
#    
#        # Backward pass: compute gradient of the loss with respect to model parameters
#        loss.backward()
#        
#        # Calling the step function on an Optimizer makes an update to its parameters
#        optimizer.step()
          
#    for epoch in range(1, args.epochs + 1):
#        train(epoch)
#        test()
   
    for epoch in range(1, args.epochs+1):
        train(epoch)
        test()
    # train(2)
    

# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size
        """ Neural Net Layers w/ 128 hidden units"""
        self.conv1 = torch.nn.Conv2d(in_size, 8, 5)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(8, 16, 5)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.hidden1 = torch.nn.Linear(16*5*5, 128)
        self.output = torch.nn.Linear(128, out_size)
        self.optimizer = torch.optim.SGD(self.parameters(), lrate, momentum=0.98555527, weight_decay=0.03)

    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = x.view(-1, self.in_size, 32, 32)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.pool(torch.nn.functional.relu(x))
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.pool(torch.nn.functional.relu(x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = torch.nn.functional.relu(self.hidden1(x))
        x = self.output(x)
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad()
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()



def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N

    model's performance could be sensitive to the choice of learning_rate. We recommend trying different values in case
    your first choice does not seem to work well.
    """
    # Standardzing the training data and development data
    train_mean = train_set.mean(dim=1, keepdim=True)
    train_std = train_set.std(dim=1, keepdim=True)
    train_set = (train_set - train_mean)/train_std   
    
    dev_mean = dev_set.mean(dim=1, keepdim=True)
    dev_std = dev_set.std(dim=1, keepdim=True)
    dev_set = (dev_set - dev_mean)/dev_std
    
    # Training
    loss = []
    net = NeuralNet(0.0009, torch.nn.CrossEntropyLoss(), 3, 2)    
    for itr in range(n_iter):
        start_idx = (itr*batch_size) % len(train_set)
        batch = train_set[start_idx: start_idx + batch_size]
        batch_labels = train_labels[start_idx: start_idx + batch_size]
        batch_loss = net.step(batch, batch_labels)
        loss.append(batch_loss)
    # Development
    dev_labels = []
    for image in dev_set:
        x = net(image)
        dev_labels.append(torch.argmax(x))
    return loss,dev_labels,net

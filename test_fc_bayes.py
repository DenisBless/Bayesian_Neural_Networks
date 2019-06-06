from __future__ import print_function

import math
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm
from utils_BNN import neg_ELBO, Logger
from model_BNN import FC_Net

use_cuda = torch.cuda.is_available()

learning_rate = 0.001
weight_decay = 0
batch_size = 16
num_epochs = 2


# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor()
    ])

# choose the training and test datasets
train_data = datasets.MNIST('data', train=True,
                              download=True, transform=transform)
test_data = datasets.MNIST('data', train=False,
                             download=True, transform=transform)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

net = FC_Net(10, 1)
logger = Logger(net)

if use_cuda:
    net.cuda()

neg_elbo = neg_ELBO(loss=nn.CrossEntropyLoss())
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

bayesian_train_acc = []
bayesian_val_acc = []
var_list = []
var_list = logger.get_variance(var_list)

mu_list = []
mu_list = logger.get_mean(mu_list)

var_grads = []
mean_grads = []

for epoch in range(1, num_epochs + 1):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    m_train = math.ceil(len(train_data) / batch_size)
    m_test = math.ceil(len(test_data) / batch_size)

    net.train()

    total = 0
    correct = 0

    for batch_idx, (data, target) in zip(tqdm(range(m_train)), (train_loader)):
        # move tensors to GPU if CUDA is available
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        beta = 1 / 2000000
        #
        # if batch_idx > 0:
        #     var_list = logger.get_logvariance_gradients(var_list)
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output, kl = net.probforward(data)

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # calculate the batch loss
        loss = neg_elbo(output, target, kl, beta)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += (loss.item() * data.size(0))


    test_correct = 0
    test_total = 0

    # with torch.no_grad():
    #     for i, (test_input, test_target) in zip(tqdm(range(m_test)), test_loader):
    #         # for (test_data, test_target) in test_loader:
    #         test_input, test_target = test_input.cuda(), test_target.cuda()
    #         test_outputs, kl = net.probforward(test_input)
    #         _, test_predicted = torch.max(test_outputs.data, 1)
    #         test_total += test_target.size(0)
    #         test_correct += (test_predicted == test_target).sum().item()

    train_loss = train_loss / len(train_loader.dataset)

    bayesian_train_acc.append(correct / total)
    # bayesian_val_acc.append(test_correct / test_total)

    # var_list = logger.get_variance(var_list)
    # mu_list = logger.get_mean(mu_list)
    # var_grads = logger.get_variance_gradients(var_grads)
    # mean_grads = logger.get_mean_gradients(mean_grads)

    print('--------------------------------------------------------------')
    print('Epoch:', epoch)
    print('--------------------------------------------------------------')
    print('Trainig loss:', train_loss)
    print('--------------------------------------------------------------')
    print('Accuracy of the network on the train images: {} percent ({}/{})'.format(
        100 * correct / total, correct, total))
    print('--------------------------------------------------------------')

    # print('Accuracy of the network on the test images: {} percent ({}/{})'.format(
    #     100 * test_correct / test_total, test_correct, test_total))
    # print('--------------------------------------------------------------')
import torch.nn as nn
from bayesian_layer import FlattenLayer, Bayesian_conv2D, Bayesian_fullyconnected

class Small_conv_net(nn.Module):
    def __init__(self, outputs, inputs):
        super(Small_conv_net, self).__init__()

        self.conv1 = Bayesian_conv2D(inputs, 6, 5, stride=1)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Bayesian_conv2D(6, 16, 5, stride=1)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = Bayesian_fullyconnected(5 * 5 * 16, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.soft3 = nn.Softplus()

        self.fc2 = Bayesian_fullyconnected(120, outputs)

        layers = [self.conv1, self.conv1_bn, self.soft1, self.pool1, self.conv2, self.conv2_bn, self.soft2, self.pool2,
                  self.flatten, self.fc1, self.fc1_bn, self.soft3, self.fc2]

        self.layers = nn.ModuleList(layers)


        # self.conv1 = Bayesian_conv2D(inputs, 32, 5, stride=1, padding=2, bias=True)
        # self.conv1_bn = nn.BatchNorm2d(32)
        # self.soft1 = nn.Softplus()
        # self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        #
        # self.conv2 = Bayesian_conv2D(32, 64, 5, stride=1, padding=2, bias=True)
        # self.conv2_bn = nn.BatchNorm2d(64)
        # self.soft2 = nn.Softplus()
        # self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        #
        # self.conv3 = Bayesian_conv2D(64, 128, 5, stride=1, padding=1, bias=True)
        # self.conv3_bn = nn.BatchNorm2d(128)
        # self.soft3 = nn.Softplus()
        # self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        #
        # self.flatten = FlattenLayer(2 * 2 * 128,)
        # self.fc1 = Bayesian_fullyconnected(2 * 2 * 128, 100, bias=True)
        # self.fc1_bn = nn.BatchNorm1d(100)
        # self.soft5 = nn.Softplus()
        #
        # self.fc3 = Bayesian_fullyconnected(100, outputs, bias=True)
        #
        # self.layers = [self.conv1, self.conv1_bn, self.soft1, self.pool1, self.conv2,self.conv2_bn, self.soft2, self.pool2,
        #           self.conv3, self.conv3_bn, self.soft3, self.pool3, self.flatten, self.fc1, self.fc1_bn, self.soft5,
        #           self.fc3]



    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'conv_forward_bayes') and callable(layer.conv_forward_bayes):
                x, _kl, = layer.conv_forward_bayes(x)
                kl += _kl

            elif hasattr(layer, 'fc_forward_bayes') and callable(layer.fc_forward_bayes):
                x, _kl, = layer.fc_forward_bayes(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl

class FC_Net(nn.Module):
    def __init__(self, outputs, inputs):
        super(FC_Net, self).__init__()

        self.flatten = FlattenLayer(28*28)
        self.fc1 = Bayesian_fullyconnected(28*28, 120)
        self.soft3 = nn.Softplus()

        self.fc2 = Bayesian_fullyconnected(120, 84)
        self.soft4 = nn.Softplus()

        self.fc3 = Bayesian_fullyconnected(84, outputs)

        layers = [self.flatten, self.fc1, self.soft3, self.fc2, self.soft4, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'conv_forward_bayes') and callable(layer.conv_forward_bayes):
                x, _kl, = layer.conv_forward_bayes(x)
                kl += _kl

            elif hasattr(layer, 'fc_forward_bayes') and callable(layer.fc_forward_bayes):
                x, _kl, = layer.fc_forward_bayes(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl
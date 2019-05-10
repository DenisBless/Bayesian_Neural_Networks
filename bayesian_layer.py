import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from utils_BNN import Normal, ELBO

cuda = torch.cuda.is_available()

class _ConvNd(nn.Module):
    """
    Module for bayesian inference in a convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, prior_log_var_init=0.05, q_log_var_init=0.05):
        super(_ConvNd, self).__init__()

        # Params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Init variance of prior and variational distribution q
        self.prior_log_var_init = prior_log_var_init
        self.q_log_var_init = q_log_var_init

        # Placeholder variables
        self.conv_q_w_mean = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.conv_q_w_std = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.log_alpha = Parameter(torch.Tensor(1,1))  # TODO: size and differnce to q_std unclear

        # Init of weights
        self.weights = Normal(mu=self.conv_q_w_mean, logvar=self.conv_q_w_std)

        # Init of prior
        self.prior = Normal(mu=0, logvar=self.prior_log_var_init)

    def reset_params(self):
        """
        Fills the placeholder variables with values.
        """

        # TODO: Unknown rule of thumb
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        self.conv_q_w_mean.data.uniform(-stdv,stdv)
        self.log_alpha.data.uniform(-stdv, stdv)
        self.conv_q_w_std.data.fill(self.q_log_var_init)


class Bayesian_conv2D(_ConvNd):
    """
    Implements a 2D convolutional bayesian layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size) # TODO: unknown operator _pair
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(Bayesian_conv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups,)

    def conv_forward_bayes(self, input):

        # Sample from weights with conv_q_w_mean and conv_q_w_std
        weight = self.weight.sample()

        # Two consecutice convolutions for local reparameterization trick
        conv_q_w_mean = F.conv2d(input=input, weight=weight, stride=self.stride,
                            padding=self.padding)
        conv_q_w_std = torch.sqrt(1e-8 + F.conv2d().pow(2),
                             weight=torch.exp(self.log_alpha)*weight.pow(2),
                             stride=self.stride, padding=self.padding)

        if cuda:
            conv_q_w_mean,conv_q_w_std = conv_q_w_mean.cuda(), conv_q_w_std.cuda()
            output = conv_q_w_mean + conv_q_w_std * torch.cuda.FloatTensor(conv_q_w_mean.size()).normal_()
            output = output.cuda()

        # Calculate second term of the ELBO (Monte Carlo approx.): # TODO: Check correctness 1/n term is missing? or is it added later via beta?
        # The KL Divergence between the variational distribution q and the prior over w
        conv_q_w = Normal(mu=conv_q_w_mean, logvar=conv_q_w_std)
        weight_sample =  conv_q_w.sample()
        qw_pdf = conv_q_w.pdf(weight_sample)
        qw_log_pdf = conv_q_w.logpdf(weight_sample)
        kl_div = torch.sum(qw_log_pdf - self.prior.logpdf(weight_sample))

        return output, kl_div
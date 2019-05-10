import math
import torch



class Normal():
    # scalar version
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        self.shape = mu.size()

        super(Normal, self).__init__()

    def logpdf(self, x):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * self.logvar - (x - self.mu).pow(2) / (2 * torch.exp(self.logvar))

    def pdf(self, x):
        return torch.exp(self.logpdf(x))

    def sample(self):
        if self.mu.is_cuda:
            eps = torch.cuda.FloatTensor(self.shape).normal_()
        else:
            eps = torch.FloatTensor(self.shape).normal_()
        # local reparameterization trick
        return self.mu + torch.exp(0.5 * self.logvar) * eps

    def entropy(self):
        return 0.5 * math.log(2. * math.pi * math.e) + 0.5 * self.logvar


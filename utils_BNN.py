import math
import torch
from torch import nn
import torch.nn.functional as F


# Returns value of the evidence lower bound (ELBO)
class neg_ELBO(nn.Module):
    """
    the negative ELBO can be decomposed into the expected negative log likelihood ENLL and the KL-Divergence
    between the variational and the prior distribution
    """

    def __init__(self, loss=nn.CrossEntropyLoss()):
        super(neg_ELBO, self).__init__()
        self.loss = loss

    def forward(self, outputs, y, kl_div, beta):
        ENLL = self.loss(outputs,y)
        loss = ENLL + beta * kl_div
        return loss


def kl_divergence(q_mean, q_std, p_mean, p_std):
    return torch.sum(0.5 * (-torch.log(q_std**2/p_std**2) + (q_std**2 + (q_mean-p_mean)**2)/(p_std**2) - 1)) #


class Logger:
    """
    Logs parameters of the model
    """
    def __init__(self, model):
        self.model = model

    def get_variance(self, var_list):
        for c in self.model.children():
            try:
                var_list.append((c.sigma ** 2).reshape(-1, ).detach().cpu().numpy())
            except:
                pass

        return var_list

    def get_logvariance(self, var_list):
        for c in self.model.children():
            try:
                var_list.append((torch.exp(c.logvar)).reshape(-1, ).detach().cpu().numpy())
            except:
                pass

        return var_list

    def get_mean(self, mean_list):
        for c in self.model.children():
            try:
                mean_list.append((c.mean).reshape(-1, ).detach().cpu().numpy())
            except:
                pass

        return mean_list

    def get_mean_gradients(self, mean_grads):
        for params in list(self.model.named_parameters()):
            if 'mean' in params[0]:
                mean_grads.append(params[1].grad.reshape(-1, ).detach().cpu().numpy())

        return mean_grads

    def get_variance_gradients(self, var_grads):
        for params in list(self.model.named_parameters()):
            if 'sigma' or 'logvar' in params[0]:
                var_grads.append(params[1].grad.reshape(-1, ).detach().cpu().numpy())
                # var_grads.append(params[1].grad)

        return var_grads

    def get_logvariance_gradients(self, var_grads):
        for params in list(self.model.named_parameters()):
            if 'logvar' in params[0]:
                var_grads.append(params[1].grad.reshape(-1, ).detach().cpu().numpy())
                # var_grads.append(params[1].grad)

        return var_grads



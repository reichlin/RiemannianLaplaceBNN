import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.functional import hessian
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from models_utils import Network


class VIBNN(nn.Module):

    def __init__(self, network_specs, weight_decay, beta_kl, lr, loss_type, diag, n_test_samples, device, init_vals):
        super().__init__()

        self.loss_type = loss_type
        self.device = device
        self.n_test_samples = n_test_samples
        self.weight_decay = weight_decay
        self.beta_kl = beta_kl
        scale_mean = init_vals['mu']
        scale_std = init_vals['std']

        self.diag = diag

        self.f = Network(network_specs, loss_type, probabilistic=True)

        self.theta_mu = nn.Parameter(torch.randn(self.f.tot_params) * scale_mean, requires_grad=True)
        if diag:
            self.theta_rho = nn.Parameter(torch.randn(self.f.tot_params) * scale_std, requires_grad=True)
        else:
            self.theta_rho = nn.Parameter(torch.tril(torch.randn(self.f.tot_params, self.f.tot_params) * scale_std), requires_grad=True)

        self.f.init_params(self.f.get_unflat_params(self.theta_mu))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def get_sigma(self, rho):
        if self.diag:
            return F.softplus(rho)
        else:
            return torch.tril(rho) - torch.diag(torch.diag(rho)) + torch.diag(F.softplus(torch.diag(rho))) + 1.0 * torch.eye(rho.size(0), device=self.device)

    def sample_weights(self):
        if self.diag:
            theta_i = torch.randn(self.theta_mu.shape).to(self.device) * self.get_sigma(self.theta_rho) + self.theta_mu
        else:
            theta_i = torch.randn(self.theta_mu.shape).to(self.device) @ self.get_sigma(self.theta_rho) + self.theta_mu
        return self.f.get_unflat_params(theta_i)

    def forward(self, x):
        theta = self.sample_weights()
        return self.f(x, theta)

    def loss(self, y_pred, y):
        L_evidence = self.f.loss_neglikelihood(y_pred, y)
        L_prior = self.kl(self.theta_mu, self.get_sigma(self.theta_rho)).sum()

        Loss = L_evidence + self.beta_kl * L_prior

        losses = {'Evidence': L_evidence.detach().cpu().item(),
                  'KL': L_prior.detach().cpu().item()}

        return losses, Loss

    def kl(self, mu, sigma):
        sigma_w = self.weight_decay * torch.eye(mu.shape[0]).to(mu.device)

        if self.diag:
            sigma2 = torch.diag(sigma**2)
        else:
            sigma2 = sigma @ sigma.T

        p = MultivariateNormal(torch.zeros(mu.shape[0]).to(mu.device), sigma_w)
        q = MultivariateNormal(mu, sigma2)

        return torch.distributions.kl_divergence(q, p)

    def posterior(self, all_x, loader):
        x_train, y_train = loader.dataset.x_train, loader.dataset.y_train
        py = torch.stack([self(all_x) for _ in range(self.n_test_samples)], 0)
        y_mu = py.mean(0)
        y_std = py.std(0)
        return y_mu, y_std, py

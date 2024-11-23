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

from laplace import Laplace


class LABNN(nn.Module):

    def __init__(self, implementation_type, loss_category, network_specs, weight_decay, lr, loss_type, n_test_samples, hessian_type, probabilistic, device):
        super().__init__()

        self.implementation_type = implementation_type
        self.loss_category = loss_category
        self.loss_type = loss_type
        self.device = device
        self.n_test_samples = n_test_samples
        self.weight_decay = weight_decay

        self.hessian_type = hessian_type

        self.f = Network(network_specs, loss_type, probabilistic=probabilistic)
        self.theta = nn.ParameterList([nn.Parameter(torch.zeros(t_size), requires_grad=True) for t_size in self.f.get_theta_shape()])
        self.f.init_params(self.theta)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, theta=None):
        if theta is None:
            theta = self.theta
        try:
            return self.f(x, theta)
        except:
            return self.f(x, self.f.get_unflat_params(theta))

    def loss(self, y_pred, y):
        L_evidence = self.f.loss_neglikelihood(y_pred, y)
        losses = {'Evidence': L_evidence.detach().cpu().item()}
        return losses, L_evidence

    def loss_f(self, theta, x, y):
        try:
            return self.f.loss_neglikelihood(self.f(x, theta), y)
        except:
            return self.f.loss_neglikelihood(self.f(x, self.f.get_unflat_params(theta)), y)

    def get_hessian(self, theta, x_train, y_train):

        if self.hessian_type == 'full':
            H = hessian(self.loss_f, (theta, x_train, y_train))[0][0]
        elif self.hessian_type == 'diag':
            H = hessian(self.loss_f, (theta, x_train, y_train))[0][0]
            H = torch.diag(torch.diag(H))
        elif self.hessian_type == 'fisher':
            theta.requires_grad_(True)
            L = self.f.loss_neglikelihood(self.f(x_train, self.f.get_unflat_params(theta)), y_train)
            theta_grad = torch.autograd.grad(outputs=L, inputs=theta)[0]
            H = theta_grad.view(-1, 1) @ theta_grad.view(1, -1)

        return H

    def compute_best_marginal(self, all_w, theta, H, x, y, samples):
        all_marginals = []
        for w in all_w:
            try:
                var = torch.linalg.inv((H + w * torch.eye(H.shape[0]).to(self.device)))
                q = MultivariateNormal(theta, var)
                dist_theta = torch.stack([q.sample() for _ in range(samples)], 0)
                py = torch.stack([self(x, theta_sample) for theta_sample in dist_theta], 0)
                log_like = -self.f.loss_neglikelihood(py, torch.cat([torch.unsqueeze(y, 0)]*samples, 0))
                p = MultivariateNormal(torch.zeros(H.shape[0]).to(H.device), (1 / w) * torch.eye(H.shape[0]).to(H.device))
                p_theta = torch.stack([p.log_prob(theta_sample) for theta_sample in dist_theta], 0)
                all_marginals.append((log_like + torch.mean(p_theta)).detach().cpu().item())
            except:
                all_marginals.append(-np.inf)
        return all_w[np.argmax(np.array(all_marginals))]

    def find_hyper(self, theta, H, x, y, samples=100):

        all_w = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
        best_w = self.compute_best_marginal(all_w, theta, H, x, y, samples)

        all_w = np.linspace(best_w*0.1, best_w*10)
        best_w = self.compute_best_marginal(all_w, theta, H, x, y, samples)

        return best_w

    def posterior(self, all_x, loader):

        if self.implementation_type == 0:

            la = Laplace(self, self.loss_category,
                         subset_of_weights="all",
                         hessian_structure=self.hessian_type)
            la.fit(loader)
            la.optimize_prior_precision(
                method='marglik',
                pred_type='glm',
                link_approx='mc',
                val_loader=loader
            )

            py = torch.stack([self(all_x, self.f.get_unflat_params(wegihts)) for wegihts in la.sample(self.n_test_samples)], 0)

        else:
            x_train, y_train = loader.dataset.x_train, loader.dataset.y_train

            theta = self.f.get_flat_params(self.theta)

            H = self.get_hessian(theta, x_train, y_train.float())

            best_w = self.find_hyper(theta, H, x_train, y_train, 100)

            var = torch.linalg.inv((H + best_w * torch.eye(H.shape[0]).to(self.device)))

            p_theta = MultivariateNormal(theta, var)
            py = torch.stack([self(all_x, p_theta.sample()) for _ in range(self.n_test_samples)], 0)

        y_mu = py.mean(0)
        y_std = py.std(0)
        return y_mu, y_std, py


class RLABNN(nn.Module):

    def __init__(self, input_size, output_size, weight_decay, lr, loss_type, device):
        super().__init__()

    def forward(self, x):
        return


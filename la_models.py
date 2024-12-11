import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.functional import hessian, jvp, jacobian
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from models_utils import Network
from scipy.integrate import solve_ivp

from laplace import Laplace

from torchdiffeq import odeint


class LABNN(nn.Module):

    def __init__(self, implementation_type, loss_category, network_specs, weight_decay, lr, loss_type, n_test_samples, hessian_type, probabilistic, marginal_type, do_riemannian, device):
        super().__init__()

        self.marginal_type = marginal_type

        self.implementation_type = implementation_type
        self.loss_category = loss_category
        self.loss_type = loss_type
        self.device = device
        self.n_test_samples = n_test_samples
        self.weight_decay = weight_decay

        self.hessian_type = hessian_type

        self.do_riemannian = do_riemannian

        self.f = Network(network_specs, loss_type, probabilistic=probabilistic)
        self.theta = nn.ParameterList([nn.Parameter(torch.zeros(t_size), requires_grad=True) for t_size in self.f.get_theta_shape()])
        self.f.init_params(self.theta)

        self.I = torch.eye(self.f.tot_params).to(self.device)

        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)

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
        return self.f.loss_neglikelihood(self(x, theta=theta), y)

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
        elif self.hessian_type == 'gauss_newton':

            J = torch.squeeze(jacobian(self.forward, (x_train, theta))[1])

            if self.loss_category == 'classification':
                z = self.forward(x_train, theta)
                p = torch.softmax(z, -1)
                p_diag = torch.diag_embed(p, dim1=-2, dim2=-1)
                p_outer = torch.bmm(torch.unsqueeze(p, -1), torch.unsqueeze(p, 1))
                S = p_diag - p_outer
                H = torch.sum(torch.bmm(torch.bmm(S, J).transpose(2, 1), J), 0)
            else:
                H = torch.sum(torch.bmm(torch.unsqueeze(J, -1), torch.unsqueeze(J, 1)), 0)
        return H

    def compute_best_marginal(self, all_w, theta, H, x, y, samples):
        all_marginals = []
        for w in all_w:
            try:
                mu_zero = torch.zeros(H.shape[0]).to(H.device)
                L_like = self.f.loss_neglikelihood(self(x, theta), y)

                precision_posterior = H + w * self.I
                q = MultivariateNormal(theta, precision_matrix=precision_posterior)
                p = MultivariateNormal(mu_zero, precision_matrix=w * self.I)
                L_posterior = L_like - p.log_prob(theta)

                if self.marginal_type == 'determinant':
                    logdet_posterior = - torch.log(torch.linalg.det(precision_posterior))
                    log_marginal = - L_posterior + 0.5 * logdet_posterior
                else:
                    assert self.self.loss_category == 'regression', print("marginal expectation not implemented for classification")
                    dist_theta = torch.stack([q.sample() for _ in range(samples)], 0)
                    y_mc = torch.stack([self(x, theta_sample) for theta_sample in dist_theta], 0)
                    log_D = torch.sum(- (y_mc - torch.cat([torch.unsqueeze(y, 0)] * samples, 0)) ** 2 / 2, (1, 2))
                    log_p_mc = torch.stack([p.log_prob(theta_sample) for theta_sample in dist_theta], 0)
                    log_q_mc = torch.stack([q.log_prob(theta_sample) for theta_sample in dist_theta], 0)
                    log_marginal = torch.logsumexp(log_D + log_p_mc - log_q_mc, dim=0)
                if torch.isnan(log_marginal) or torch.isinf(log_marginal):
                    log_marginal = -np.inf
                else:
                    log_marginal = log_marginal.detach().cpu().item()
                all_marginals.append(log_marginal)
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

            la = Laplace(self,
                         self.loss_category,
                         subset_of_weights="all",
                         hessian_structure=self.hessian_type)
            la.fit(loader)
            la.optimize_prior_precision(method='marglik',
                                        pred_type='glm',
                                        link_approx='mc')

            py = torch.stack([self(all_x, self.f.get_unflat_params(weights)) for weights in la.sample(self.n_test_samples)], 0)

        else:
            self.x_train, self.y_train = loader.dataset.x_train, loader.dataset.y_train

            theta = self.f.get_flat_params(self.theta)

            H = self.get_hessian(theta, self.x_train, self.y_train.float())

            best_w = self.find_hyper(theta, H, self.x_train, self.y_train, 100)

            posterior_precision = H + best_w * self.I

            p_theta = MultivariateNormal(theta, precision_matrix=posterior_precision)
            if self.do_riemannian:
                self.n_theta = theta.shape[0]
                py = torch.stack([self(all_x, self.solve_expmap(theta, p_theta.sample())) for _ in range(self.n_test_samples)], 0)
            else:
                py = torch.stack([self(all_x, p_theta.sample()) for _ in range(self.n_test_samples)], 0)

        y_map = self(all_x, self.theta)

        y_mu = py.mean(0)
        y_std = py.std(0)
        return y_map, y_mu, y_std, py

    def solve_expmap(self, x, v):
        # exp_x(v) for point on manifold x & tangent vector v
        x0 = torch.cat((x, v), dim=-1)
        sol = solve_ivp(self.geodesic_ode_fun, [0, 1], x0.detach().cpu().numpy().flatten(), dense_output=True, atol = 1e-3, rtol= 1e-6)
        theta = torch.from_numpy(sol['y'][:self.n_theta, -1]).float().to(self.device)
        return theta

    def geodesic_ode_fun(self, t, x_np):
        dgamma_np = x_np[self.n_theta:]
        x_torch = torch.from_numpy(x_np).float().to(self.device).squeeze()
        gamma, dgamma = torch.split(x_torch, split_size_or_sections=int(self.n_theta), dim=-1)
        dL = self.grad_loss(gamma).detach().cpu().numpy()
        hvp = jvp(self.grad_loss, gamma, dgamma)[1].detach().cpu().numpy()
        ddgamma = -dL / (1 + np.dot(dL, dL)) * np.dot(dgamma_np, hvp)
        dx = np.concatenate((dgamma_np, ddgamma))
        return dx

    def grad_loss(self, theta):
        return jacobian(self.loss_f, (theta, self.x_train, self.y_train), create_graph=True)[0]


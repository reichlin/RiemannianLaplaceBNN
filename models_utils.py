import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.special import softmax
from sklearn import metrics
from netcal.metrics import ECE, MCE


class Network(nn.Module):
    def __init__(self, network_specs, loss_type, probabilistic=False):
        super(Network, self).__init__()

        self.loss_type = loss_type
        self.probabilistic = probabilistic

        self.theta_shapes = []
        for layer in network_specs['architecture']:
            if len(layer) == 2:
                self.theta_shapes.append([layer[1], layer[0]])
                self.theta_shapes.append([layer[1]])
            else:
                self.theta_shapes.append([layer[2], layer[0], layer[1], layer[1]])
                self.theta_shapes.append([layer[2]])
                pass
        self.activation = network_specs['activation']

        self.tot_params = sum([math.prod(param_shape) for param_shape in self.theta_shapes])

    def get_theta_shape(self):
        return self.theta_shapes

    def init_params(self, theta):
        for theta_i in theta:
            if theta_i.dim() > 1:
                torch.nn.init.xavier_uniform_(theta_i)

    def get_flat_params(self, theta):
        vec_theta = []
        for param in theta:
            vec_theta.append(param.data.view(-1))
        vec_theta = torch.cat(vec_theta)
        return vec_theta

    def get_unflat_params(self, theta):
        theta_params = []
        i = 0
        for param_shape in self.theta_shapes:
            n_params = math.prod(param_shape)
            theta_params.append(theta[i:i+n_params].view(param_shape))
            i += n_params
        return theta_params

    def forward(self, x, theta):

        n = len(theta)
        h = x
        for i in range(0, (n // 2) - 1):
            if theta[2*i].dim() > 2:
                h = self.activation(F.conv2d(h, theta[2*i], bias=theta[2*i+1]))
                h = F.avg_pool2d(h, 2)
            else:
                h = torch.flatten(h, 1) if h.dim() == 4 else h
                h = self.activation(F.linear(h, theta[2*i], bias=theta[2*i+1]))
        out = F.linear(h, theta[n-2], bias=theta[n-1])
        if not self.probabilistic:
            return out

        mu, sigma = torch.split(out, out.shape[-1] // 2, dim=-1)
        return torch.concatenate([mu, F.softplus(sigma) + 0.01], -1)

    # def loss_neglikelihood(self, y_pred, y):
    #     if self.loss_type == 'mse':
    #         mse = (y_pred - y)**2
    #         if y_pred.dim() > 2:
    #             mse = torch.sum(mse, dim=tuple(range(1, mse.dim())))
    #         return mse.mean()
    #     elif self.loss_type == 'NLL':
    #         N = Normal(*torch.split(y_pred, y_pred.shape[-1] // 2, dim=-1))
    #         nll = - N.log_prob(y)
    #         if y_pred.dim() > 2:
    #             nll = torch.sum(nll, dim=tuple(range(1, nll.dim())))
    #         return nll.mean()
    #     elif self.loss_type == 'CE':
    #
    #         shape = y.shape
    #         if len(shape) > 1:
    #             y = y.view(-1)
    #             y_pred = y_pred.view(-1, y_pred.shape[-1])
    #             ce = F.cross_entropy(y_pred, y.long(), reduction='none')
    #             ce = torch.reshape(ce, shape)
    #             ce = torch.sum(ce, dim=tuple(range(1, ce.dim())))
    #         else:
    #             ce = F.cross_entropy(y_pred, y.long(), reduction='none')
    #         return ce.mean()
    def loss_neglikelihood(self, y_pred, y):
        if self.loss_type == 'mse':
            L = (y_pred - y)**2 / 2
        elif self.loss_type == 'NLL':
            N = Normal(*torch.split(y_pred, y_pred.shape[-1] // 2, dim=-1))
            L = - N.log_prob(y)
        elif self.loss_type == 'CE':
            shape = y.shape
            if len(shape) > 1:
                y = y.view(-1)
                y_pred = y_pred.view(-1, y_pred.shape[-1])
            L = F.cross_entropy(y_pred, y.long(), reduction='none')
            #     ce = torch.reshape(ce, shape)
            #     ce = torch.sum(ce, dim=tuple(range(1, ce.dim())))
            # else:
            #     ce = F.cross_entropy(y_pred, y.long(), reduction='none')
        return L.sum()


def regression_metrics(model, loader):

    stats_metrics = {}

    x_test, y_test = loader.dataset.x_test, loader.dataset.y_test
    y_map, y_mu, y_std, py = model.posterior(x_test, loader)

    y = y_test.detach().cpu().numpy()[:,0]
    dist_y_pred = py.detach().cpu().numpy()[:, :, 0]

    mse = np.array([metrics.mean_squared_error(y, dist_y_pred[i]) for i in range(dist_y_pred.shape[0])])
    stats_metrics['MSE'] = [np.mean(mse), np.std(mse)]

    return stats_metrics

def classification_metrics(model, loader):

    stats_metrics = {}

    x_test, y_test = loader.dataset.x_test, loader.dataset.y_test
    y_map, y_mu, y_std, py = model.posterior(x_test, loader)

    y = y_test.detach().cpu().numpy()
    dist_y_pred = torch.softmax(py, -1).detach().cpu().numpy()

    accuracy = np.array([metrics.accuracy_score(y, np.argmax(dist_y_pred[i], -1)) for i in range(dist_y_pred.shape[0])])
    stats_metrics['Accuracy'] = [np.mean(accuracy), np.std(accuracy)]

    nll = np.array([metrics.log_loss(y, dist_y_pred[i], labels=np.arange(py.shape[-1])) for i in range(dist_y_pred.shape[0])])
    stats_metrics['NLL'] = [np.mean(nll), np.std(nll)]

    if py.shape[-1] == 2:
        brier = np.array([metrics.brier_score_loss(y, dist_y_pred[i, :, 1]) for i in range(dist_y_pred.shape[0])])
        stats_metrics['Brier'] = [np.mean(brier), np.std(brier)]

    ece = np.array([ECE(bins=10).measure(dist_y_pred[i], y) * 100 for i in range(dist_y_pred.shape[0])])
    stats_metrics['ECE'] = [np.mean(ece), np.std(ece)]

    mce = np.array([MCE(bins=10).measure(dist_y_pred[i], y) * 100 for i in range(dist_y_pred.shape[0])])
    stats_metrics['MCE'] = [np.mean(mce), np.std(mce)]

    return stats_metrics


def get_regression_fig(model, loader, device):

    all_x = torch.linspace(-3, 10, 100).float().to(device).view(-1, 1)
    y_map, y_mu, y_std, py = model.posterior(all_x, loader)

    x_train, y_train = loader.dataset.x_train.detach().cpu().numpy()[:, 0], loader.dataset.y_train.detach().cpu().numpy()[:, 0]
    x_test, y_test = loader.dataset.x_test.detach().cpu().numpy()[:, 0], loader.dataset.y_test.detach().cpu().numpy()[:, 0]
    all_x = all_x.detach().cpu().numpy()[:, 0]
    y_mu, y_std = y_mu.detach().cpu().numpy()[:, 0], y_std.detach().cpu().numpy()[:, 0]
    y_map = y_map.detach().cpu().numpy()[:, 0]

    fig = plt.figure()

    plt.scatter(x_train, y_train, color='tab:blue')
    plt.scatter(x_test, y_test, color='tab:red')
    plt.plot(all_x, y_map, color='tab:green', linewidth=3)
    plt.fill_between(all_x, y_map - y_std, y_map + y_std, alpha=0.5, color='tab:green')
    for y_sample in py:
        plt.plot(all_x, y_sample.detach().cpu().numpy()[:, 0], color='tab:orange', alpha=0.1)
    plt.ylim(-4, 4)

    return fig



def get_banana_fig(model, loader, device):

    x_test, y_test = loader.dataset.x_test.detach().cpu().numpy(), loader.dataset.y_test.detach().cpu().numpy()

    N_grid = 100
    offset = 2
    x1min = x_test[:, 0].min() - offset
    x1max = x_test[:, 0].max() + offset
    x2min = x_test[:, 1].min() - offset
    x2max = x_test[:, 1].max() + offset

    x_grid = np.linspace(x1min, x1max, N_grid)
    y_grid = np.linspace(x2min, x2max, N_grid)
    XX1, XX2 = np.meshgrid(x_grid, y_grid)
    X_grid = np.column_stack((XX1.ravel(), XX2.ravel()))
    all_x = torch.from_numpy(X_grid).float().to(device)

    y_map, y_mu, y_std, py = model.posterior(all_x, loader)

    y_map = torch.reshape(y_map, (N_grid, N_grid, 2)).detach().cpu().numpy()
    y_map = softmax(y_map, -1)
    py = torch.reshape(py, (-1, N_grid, N_grid, 2)).detach().cpu().numpy()
    py_class = (py[:, :, :, 0] > py[:, :, :, 1]) * 1.
    unc = np.std(py_class, 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    contour_like = axes[0].contourf(
        XX1,
        XX2,
        y_map[:, :, 0],
        alpha=0.8,
        antialiased=True,
        cmap="PuOr",
        levels=np.arange(-0.3, 1.3, 0.1),
    )
    plt.colorbar(contour_like, ax=axes[0], orientation='vertical')
    axes[0].scatter(x_test[:, 0], x_test[:, 1],
                    c=y_test, cmap=ListedColormap(["tab:purple", "tab:orange"]),
                    edgecolor='black', linewidth=0.15, s=5, zorder=1, alpha=1.0)
    axes[0].set_title("Likelihood Uncertainty")
    contour_unc = axes[1].contourf(
        XX1,
        XX2,
        unc,
        alpha=0.8,
        antialiased=True,
        cmap="Blues",
        levels=np.arange(np.min(unc) - 0.1, np.max(unc) + 0.1, 0.01),
    )
    plt.colorbar(contour_unc, ax=axes[1], orientation='vertical')
    axes[1].scatter(x_test[:, 0], x_test[:, 1],
                    c=y_test, cmap=ListedColormap(["tab:purple", "tab:orange"]),
                    edgecolor='black', linewidth=0.15, s=5, zorder=1, alpha=1.0)
    axes[1].set_title("Model Uncertainty")
    plt.tight_layout()

    return fig




























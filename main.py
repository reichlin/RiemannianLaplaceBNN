import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

from models_utils import regression_metrics, classification_metrics, get_regression_fig, get_banana_fig
from datasets import Regression, Banana, Uci, Mnist
from vi_model import VIBNN
from la_models import LABNN
from test_model import Test_Model


parser = argparse.ArgumentParser()

parser.add_argument('--experiment', default=0, type=int, help="0: regression, 1: banana, 2-7: UCI, 8: MNIST, 9: FashionMNIST")

parser.add_argument('--model_type', default=2, type=int, help="-1: test, 0: VI_BNN, 1: Laplace_BNN, 2: Laplace_BNN_our, 3: RiemannianLaplace_BNN")
parser.add_argument('--model_size', default=0, type=int, help="0: small, 1: big, 2: real")
parser.add_argument('--seed', default=0, type=int, help="seed")

parser.add_argument('--wd', default=0.01, type=float, help="L2 regularization")
parser.add_argument('--kl', default=0.01, type=float, help="KL weighting term")
parser.add_argument('--std', default=0, type=float, help="initial standard deviation")

parser.add_argument('--hessian_type', default=6, type=int, help="0: full, 1: diag, 2: fisher, 3: kron, 4: lowrank, 5: gp, 6:gauss_newton")
parser.add_argument('--prob', default=0, type=int, help="0: deterministic_out, 1: probabilistic_out")

args = parser.parse_args()

assert (args.prob == 1 if args.model_type == 0 else True), "When using Variational Inference the model has to be probabilistic"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

model_type = args.model_type
model_names = {-1: 'test', 0: 'VI_BNN', 1: 'Laplace_BNN', 2: 'Laplace_BNN_our', 3: 'RiemannianLaplace_BNN'}
model_sizes = {0: 'small', 1: 'big', 2: 'real'}
model_size = model_sizes[args.model_size]
experiments_type = {0: 'regression',
                    1: 'banana',
                    2: 'UCI_australian',
                    3: 'UCI_breast',
                    4: 'UCI_glass',
                    5: 'UCI_ionosphere',
                    6: 'UCI_vehicle',
                    7: 'UCI_waveform',
                    8: 'MNIST',
                    9: 'FashionMNIST'}
experiment = experiments_type[args.experiment]

weight_decay = args.wd

name_exp = model_names[model_type] + '_' + model_size + '_' + experiment + '_L2=' + str(weight_decay)



loss_category = 'classification'
''' GENERAL HYPERPARAMETERS '''
if experiment == 'regression':

    # small: L2 1e-3, epochs 7e5
    # big: L2 1e-2, epochs 3.5e4

    loss_category = 'regression'

    batch_size = 32 #150
    n_test_samples = 100
    lr = 1e-3
    EPOCHS = 1000000
    testing_epochs = 10000
    ood = True
    probabilistic = args.prob == 1
    loss_type = 'NLL' if probabilistic else 'mse'

    if model_size == 'small':
        network_specs = {'architecture': [[1, 15], [15, 1+args.prob]], 'activation': nn.Tanh()}
    elif model_size == 'big':
        network_specs = {'architecture': [[1, 10], [10, 10], [10, 1+args.prob]], 'activation': nn.Tanh()}
    else:
        network_specs = {'architecture': [[1, 32], [32, 32], [32, 32], [32, 1+args.prob]], 'activation': nn.ReLU()}

    plotter = get_regression_fig
    get_metrics = regression_metrics

    dataset = Regression(ood, device)

    name_exp += '_n_samples=' + str(n_test_samples) + '_ood=' + str(ood) + '_probabilistic-output=' + str(probabilistic)

elif experiment == 'banana':

    # L2 1e-2
    network_specs = {'architecture': [[2, 16], [16, 16], [16, 2]], 'activation': nn.Tanh()}

    batch_size = 32
    n_test_samples = 100
    lr = 1e-3
    EPOCHS = 2500
    testing_epochs = 250
    probabilistic = False
    loss_type = 'CE'

    plotter = get_banana_fig
    get_metrics = classification_metrics

    dataset = Banana(device)

    name_exp += '_n_samples=' + str(n_test_samples)

elif experiment[:3] == 'UCI':

    data_specs = {  # input, number of categories
        'UCI_australian': [14, 2],
        'UCI_breast': [9, 2],
        'UCI_glass': [9, 8],
        'UCI_ionosphere': [34, 2],
        'UCI_vehicle': [25, 2],
        'UCI_waveform': [21, 3]
    }

    network_specs = {'architecture': [[data_specs[experiment][0], 50], [50, data_specs[experiment][1]]], 'activation': nn.Tanh()}

    batch_size = 32
    n_test_samples = 30
    lr = 1e-3
    EPOCHS = 10000
    testing_epochs = 100
    probabilistic = False
    loss_type = 'CE'

    plotter = lambda a, b, c: None
    get_metrics = classification_metrics

    dataset = Uci(experiment, device)

    name_exp += '_n_samples=' + str(n_test_samples)

elif experiment[-5:] == 'MNIST':

    network_specs = {'architecture': [[1, 5, 4], [4, 5, 4], [4*(4**2), 16], [16, 10], [10, 10], [10, 10]], 'activation': nn.Tanh()}

    batch_size = 32
    n_test_samples = 25
    lr = 1e-3
    EPOCHS = 100
    testing_epochs = 10
    probabilistic = False
    loss_type = 'CE'

    plotter = lambda a, b, c: None
    get_metrics = classification_metrics

    dataset = Mnist(experiment, device)

    name_exp += '_n_samples=' + str(n_test_samples)




''' MODEL-SPECIFIC HYPERPARAMETERS '''
if model_names[model_type] == 'VI_BNN':  # VI hyperparams

    beta_kl = args.kl
    init_vals = {'mu': 1e-2, 'std': args.std}
    diag = True

    model = VIBNN(network_specs, weight_decay, beta_kl, lr, loss_type, diag, n_test_samples, device, init_vals).to(device)

    name_exp += '_beta=' + str(beta_kl) + '_init_vals=' + str(init_vals)

elif model_names[model_type][:11] == 'Laplace_BNN':  # LA hyperparams

    hessian_types = {0: 'full', 1: 'diag', 2: 'fisher', 3: 'kron', 4: 'lowrank', 5: 'gp', 6: 'gauss_newton'}
    hessian_type = hessian_types[args.hessian_type]

    implementation_type = 0 if model_names[model_type] == 'Laplace_BNN' else 1

    marginal_type = 'determinant'

    model = LABNN(implementation_type, loss_category, network_specs, weight_decay, lr, loss_type, n_test_samples, hessian_type, probabilistic, marginal_type, device).to(device)

    name_exp += '_hessian_type=' + hessian_type

else:

    hessian_types = {0: 'full', 1: 'diag', 2: 'fisher', 3: 'kron', 4: 'lowrank', 5: 'gp', 6: 'gauss_newton'}
    hessian_type = hessian_types[args.hessian_type]

    model = Test_Model(loss_category, network_specs, weight_decay, lr, loss_type, n_test_samples, hessian_type, probabilistic, device).to(device)

    name_exp += '_hessian_type=' + hessian_type


writer = SummaryWriter("logs/" + name_exp + "bohboh")


loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)


tensorboard_idx = 0
for epoch in tqdm(range(EPOCHS)):

    model.train()
    for (x, y) in loader:

        y_pred = model(x)
        losses, loss = model.loss(y_pred, y)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        for loss_key in losses:
            writer.add_scalar('Loss/'+loss_key, losses[loss_key], tensorboard_idx)
        tensorboard_idx += 1

    if epoch % testing_epochs == testing_epochs-1:
        model.eval()

        metrics = get_metrics(model, loader)
        for metric_key in metrics:
            writer.add_scalar('Metrics/' + metric_key + '/mean', metrics[metric_key][0], int(epoch / testing_epochs))
            writer.add_scalar('Metrics/' + metric_key + '/std', metrics[metric_key][1], int(epoch / testing_epochs))

        fig = plotter(model, loader, device)
        if fig is not None:
            writer.add_figure('Posterior', fig, epoch)
            plt.close()


writer.close()

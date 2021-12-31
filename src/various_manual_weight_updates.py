from utils import *

### updates of the weights
def dw_sanger(w, x):
    K, D = w.shape
    # forward pass
    y = w @ x.T / math.sqrt(D)
    # backward pass
    dw = 1. / D * (y @ x - torch.tril(y @ y.T) @ w)
    return dw

def dw_mse(w, v, xs, g, dg):
    K, D = w.shape
    # local fields at the hidden layer of neurons of the AE
    lambdas = xs @ w.t() / math.sqrt(D)
    reconstructions = g(lambdas) @ v
    # x_i - \hat x_i
    deltas = xs - reconstructions
    dw = -1 / D * (v @ deltas.T / math.sqrt(D)) * dg(lambdas).T @ xs
    dv = -1 / D * g(lambdas).T @ deltas
    return dw, dv, reconstructions

def dw_mse_truncated(w, v, xs, g, dg):
    K, D = w.shape
    # local fields at the hidden layer of neurons of the AE
    lambdas =  xs@w.T / math.sqrt(D)  # K
    dv    =  - 1. / D * ( g(lambdas).T @ xs - torch.tril(g(lambdas).T @ g(lambdas)) @ v)
    term1 =  -1 / D * (v @ xs.T / math.sqrt(D)) * dg(lambdas).T @ xs
    term2 =  1/D * torch.tril(v@v.t())@(dg(lambdas) * g(lambdas)).T @ xs / sqrt(D)
    dw = term1 + term2
    return dw , dv, None
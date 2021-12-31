import math
import torch
from torch import nn
import torch.nn.functional as F

## defines the autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, D, K, g=F.relu, tied=False, scale = True, init = 0.1, bias = False):
        """
        Parameters:
        -----------
        D : input dimension
        K : number of hidden units
        g : activation function of the hidden layer (and potentially the final layer)
        tied : if True, weights of the encoder and decoder are tied.
        scale : if True, the preactivation will be encoder@x/sqrt(D) else no sqrt(D) is added.
        init : if normal, initialised with normal with std = 1./sqrt(in_features).
               else: no default initialisation
        """
        super(AutoEncoder, self).__init__()
        self.D = D
        self.K = K
        self.g = g
        self.tied = tied
        
        self.scale = scale
        
        self.fce = nn.Linear(D, K, bias=bias)  # encoder weights, shape=(K, N)
        # decoder weights, shape=(D, K)
        self.fcd = nn.Linear(K, D, bias=False) if not tied else None
        
        nn.init.normal_(self.fce.weight, mean=0.0, std=init)
        if self.fcd is not None: 
            nn.init.normal_(self.fcd.weight, mean=0.0, std=init)
           
    def forward(self, x):
        # encoding
        if self.scale: x = self.g(self.fce(x) / math.sqrt(self.D))
        else: x = self.g(self.fce(x))
        # decoding
        x = F.linear(x, self.fce.weight.data.t()) if self.tied else self.fcd(x)
        return x

## defines the autoencoder
class MLAutoEncoder(nn.Module):
    def __init__(self, D, H, K,L, g=F.relu, tied=False):
        """
        Parameters:
        -----------
        D : input dimension
        K : number of hidden units
        g : activation function of the hidden layer (and potentially the final layer)
        tied : if True, weights of the encoder and decoder are tied.
        L : number of layers in encoding and decoding ( L=1 is equivalent to an AutoEncoder class).
        H : hidden layers size
        """
        super(MLAutoEncoder, self).__init__()
        if L==1: 
            print("Please initialise an Autoencoder, you do not need multilayer and this might induce errors")
            return
        self.L = L
        self.H = H
        self.D = D
        self.K = K
        self.g = g
        self.tied = tied
        
        self.layers = nn.ModuleList()
        ## encoder
        self.layers.append(nn.Linear(D, H, bias=False))
        for l in range(L-1):
            self.layers.append(nn.Linear(H, H, bias=False))
        self.layers.append(nn.Linear(H, K, bias=False))
        if not self.tied:
            ## decoder, only initialised if not tied, otherwise will simply repeat
            self.layers.append(nn.Linear(K, H, bias=False))
            for l in range(L-1):
                self.layers.append(nn.Linear(H, H, bias=False))
            self.layers.append(nn.Linear(H, D, bias=False))
        
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=1)

        
    def forward(self, x):
        ## let me first assume that all the layers are small so I only need the sqrt(D)
        x = self.g(self.layers[0](x) / math.sqrt(self.D))
        for l in range(1,len(self.layers)-1):
            x = self.g(self.layers[l](x))
        x = self.layers[-1](x) ## no activation function at the end
        if not self.tied: return x
        x = self.g(x) ## if you are tied the last layer is the one of dim K
        x = self.g(x@self.layers[-1].weight) 
        for l in range(self.L-2, 0,-1):
            x = self.g(x@self.layers[l].weight)
        x = x@self.layers[0].weight
        return x
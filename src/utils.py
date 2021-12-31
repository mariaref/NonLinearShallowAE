import numpy as np
import math
from math import sqrt, pi
import copy
import torch

from torch import nn, optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

## rescaled erf
def erfscaled(x):
    """Output: 
        - f(x) = erf(x/√2)"""
    return torch.erf(x / math.sqrt(2))


def log( msg, logfile, verbose = True):
    """logs a message to a logfile
    Input: 
        - msg     : string to write
        - logfile : dat file where to write
        - verbose : if True print the message"""
    if verbose: 
        print(msg)
    logfile.write(msg + "\n")

def sample(P, evals, evecs):
    """Sample normal random inputs wioth covariance Ω:
    Input: 
        - P     : number of samples 
        - evals : eigenvalues  of Ω
        - evecs : eigenvectors of Ω 
    Returns 
        - xs : P samples distributed as ~ N(0,Ω) """
    ## here you are not normalising the inputs
    D = len(evals)
    zs = torch.randn(P, D)
    sqrtLambda = torch.diag(torch.sqrt(evals))
    xs = zs @ sqrtLambda @ evecs.T
    return xs


def getActfunction(g):
    """
    Input:
       - string name of function : (erf, relu or linear) 
    Returns : 
           - torch functions
    """
    print("taking activation function %s"%g)
    if g=="relu"   :  
        return F.relu, lambda x : (x>0)
    if g=="erf"    :  
        return erfscaled, (lambda x: math.sqrt(2/math.pi)*torch.exp(-x**2/2) )
    if g=="linear" : 
        return (lambda x: x), (lambda x: x*x**-1)
    else: raise NotImplementedError

def computeCovariance(dataset):
    """
    Given a torch dataset computs the covariance
    Inputs:
        - torch dataset with samples
    Outputs:
        - Covariance Ω of the samples
        - One matrix square root of Ω
        - eigenvalues of Ω
        - eigenvectors of Ω
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=10000,shuffle=False)
    xs , _  = iter(loader).next()
    Omega = xs.t()@xs/xs.shape[0]
    evals, evecs = np.linalg.eigh(Omega)
    evals += np.abs(np.min(evals)) # for numerical stability
    sqrtOmega = np.sqrt(np.diag(evals)) @ evecs.T
    return Omega, sqrtOmega, evals, evecs

def preprocessData(dataset, D = None, std=0.5, mean=0.5):
    """removes the vectorial mean and standarnormalise the dasises the data
    Inputs:
        - dataset : torch dataset
        - D       : input dimensions 
        - std     : desired std of the inputs
        - mean    : desired mean of the inputs
    Outputs:
        - dataset with centered and normalised samples"""
    datasize = dataset.data.shape[0]
    loader = torch.utils.data.DataLoader(dataset, batch_size=datasize,shuffle=False)
    xs , ys  = iter(loader).next()
    xs = xs.squeeze().reshape(xs.shape[0], xs.shape[-1]**2)
    if D is not None:
        xs = torch.nn.functional.interpolate(xs.unsqueeze(0), size=D).squeeze() ## rescales to size D
        print(f"shape of xs is {xs.shape}")
    xs /= xs.std()*2
    xs -= xs.mean()
    dataset = torch.utils.data.TensorDataset(xs, ys)
    return dataset

def getDataset(dataset,data_root, bs, test_bs, Din, datasize, gaussian_train, gaussian_test, normalise = True):
    """ Loads a beanchmark dataset
    Inputs:
        - dataset        : name fo the dataset (cifar10_gray or fmnist)
        - data_root      : where to load/save the datasets
        - bs             : batch size of train dataset
        - test_bs        : batch size of test dataset
        - Din            : None
        - datasize       : number of samples in the dataset 
        - gaussian_train : train on gaussian samples with same first two moments as those of the dataset
        - gaussian_test  : test on gaussian samples with same first two moments as those of the dataset
        - normalise      : preprocess data to have 0 mean and std 0.5
    Outputs:
        - trainloader : torch train loader
        - testloader  : torch test loader
        - D           : input dimensions
    """
    print("getting you the dataset %s from root %s"%(dataset,data_root ))
    if dataset=="cifar10_gray":
        if Din is None: D = 32**2
        else: D = Din
        input_transforms = transforms.Compose([transforms.Grayscale(),
                                       transforms.ToTensor()])
        trainset = datasets.CIFAR10(root=data_root, train=True,
                            download=True, 
                            transform=input_transforms)
        if datasize is not None: 
            trainset.data = trainset.data[:datasize] ## if you want to train on fewer samples
            trainset.targets = trainset.targets[:datasize] 
        
        testset = datasets.CIFAR10(root=data_root, train=False,
                           download=True, 
                           transform=input_transforms)
            
        trainset = preprocessData(trainset,Din)
        testset  = preprocessData(testset,Din)
        
    elif dataset=="fmnist":
        if Din is None: D = 28**2
        else: D = Din
        input_transforms = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.FashionMNIST(root=data_root, train=True,
                            download=True, 
                            transform=input_transforms)
        if datasize is not None: 
            trainset.data = trainset.data[:datasize] ## if you want to train on fewer samples
            trainset.targets = trainset.targets[:datasize] 
        
        testset = datasets.FashionMNIST(root=data_root, train=False,
                           download=True, 
                           transform=input_transforms)
        if normalise:    
            trainset = preprocessData(trainset,Din)
            testset  = preprocessData(testset,Din)
    else:
        raise NotImplementedError
    
    if gaussian_train or gaussian_test:
        Omega, sqrtOmega, _, _ = computeCovariance(trainset)
    if gaussian_train: 
        print("generating train set of gaussian inputs")
        
        xs = np.random.randn(trainset.__dict__['tensors'][0].shape[0], trainset.__dict__['tensors'][0].shape[1] ) @ sqrtOmega
        xs = torch.from_numpy(xs).float()
        ys = torch.tensor( trainset.__dict__['tensors'][1])
        trainset = torch.utils.data.TensorDataset(xs, ys)
    if gaussian_test: 
        print("generating test set of gaussian inputs")
        xs = np.random.randn(testset.__dict__['tensors'][0].shape[0], testset.__dict__['tensors'][0].shape[1] ) @ sqrtOmega
        xs = torch.from_numpy(xs).float()
        ys = torch.tensor( testset.__dict__['tensors'][1])
        testset = torch.utils.data.TensorDataset(xs, ys)
        
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                          shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs,
                                         shuffle=False)
    return trainloader, testloader, D


def createGenerativeModel(dataset, D, rank, noise, bs):
    """creates the generative model x = A c + σ xi 
    Inputs : 
        - dataset : sinusoidal or random
        - D       : input dimension
        - rank    : rank of A matrix 
        - sigma   : s.t.d. of the noise
        - bs      : number of inputs
    Outputs:
        - xs  : samples
        """
    Lambdas = torch.tensor([2**m +1 for m in range(1,rank+1)]).float() ## eigenvalues
    if dataset is None: ## if dataset is None the A matrix is random normal
        A     = torch.randn(rank, D)
    elif dataset  == "sinusoidal": ## if dataset is sinusoidal the A matrix has rows which are sinusoids
        print(f"creating a Sinusoidal dataset")
        ## creates the sinusoidal rows of the matrix
        A = torch.stack([torch.tensor([ math.sin((k+2)/100*i) for i in range(D)]) for k in range(rank)])
        ## makes sure the rows are normalised as in main text
        for idB, b in enumerate(A):
            A[idB] = b/b.norm()*math.sqrt(D)   
    c  = torch.randn(bs, rank) * Lambdas.unsqueeze(0).repeat(bs, 1) ## c~normal(0,eigenvalue/D)
    xs = c @ A + noise * torch.randn(bs, D) # x = c A + std normal(0,1) 
    return xs
        

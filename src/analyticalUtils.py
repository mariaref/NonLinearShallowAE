import numpy as np
from scipy.linalg import sqrtm as sqrtm
import scipy as sp
import scipy.special as spe
import math
from math import sqrt as sqrt
from math import pi as pi
from itertools import product
import torch
def erf_np(x):
    return spe.erf(x/math.sqrt(2))
def relu_np(x):
    return x * (x > 0)
def lin_np(x):
    return x
def Drelu(x):
    return np.int_(x>0)
def Dlin(x):
    return np.ones(x.shape)
def Derf(x):
    return math.sqrt(2./math.pi)*np.exp(-x**2/2.)


def I2_erf(C):
    """ for erf analytical value of <g(x1)g(x2)>"""
    res = C[0,1] / math.sqrt((1+C[0,0])*(1+C[1,1]))
    return 2/math.pi * math.asin(res)

def I21_erf(C):
    """ for erf analytical value of <g'(x1) x1 x2>"""
    res = C[0,1]
    res /= (1 + C[0,0])**(3./2.)
    res*= sqrt(2/pi)
    return res

def I22_erf(C):
    """ for erf analytical value of <g'(x1) x2^2 >"""
    res = C[1,1]+ C[0,0] * C[1,1] - C[0,1]**2
    res /= (1 + C[0,0])**(3./2.)
    res*= sqrt(2/pi)
    return res

def J2_erf(C):
    """for erf analytical value of  <x1 g(x2)>"""
    res = C[0,1] * math.sqrt(2)
    res/= math.sqrt(math.pi * (1+C[1,1]))    
    return res

def I3_erf(C):
    """for erf analytical value of  <g'(x1) x2 g(x3)>"""
    lambda3 = (1 + C[0, 0])*(1 + C[2, 2]) - C[0, 2]**2
    if lambda3<=0: 
        print("The Lambda 3 in I3_erf is negative! will result in instability!")
        lambda3 = 1e-5
    return (2 / math.pi / math.sqrt(lambda3) * (C[1, 2]*(1 + C[0, 0]) - C[0, 1]*C[0, 2]) / (1 + C[0, 0]))

def I4_erf(C):
    """for erf analytical value of  <g'(x1) g'(x2) g(x3) g(x4)>"""
    lambda4 = (1 + C[0, 0])*(1 + C[1, 1]) - C[0, 1]**2

    lambda0 = (lambda4 * C[2, 3]
               - C[1, 2] * C[1, 3] * (1 + C[0, 0])
               - C[0, 2]*C[0, 3]*(1 + C[1, 1])
               + C[0, 1]*C[0, 2]*C[1, 3]
               + C[0, 1]*C[0, 3]*C[1, 2])
    lambda1 = (lambda4 * (1 + C[2, 2])
               - C[1, 2]**2 * (1 + C[0, 0])
               - C[0, 2]**2 * (1 + C[1, 1])
               + 2 * C[0, 1] * C[0, 2] * C[1, 2])
    lambda2 = (lambda4 * (1 + C[3, 3])
                           - C[1, 3]**2 * (1 + C[0, 0])
                           - C[0, 3]**2 * (1 + C[1, 1])
                           + 2 * C[0, 1] * C[0, 3] * C[1, 3])

    return (4 / pi**2 / sqrt(lambda4) *
            math.asin(lambda0 / sqrt(lambda1 * lambda2)))

def I2_relu(c11, c12, c22):
    """ for relu analytical value of <g(x1)g(x2)>"""
    if c11 * c22 - c12 **2 < 1e-6 :  return c11/2.
    Sqdet = c11 * c22 - c12 **2
    Sqdet = math.sqrt(Sqdet)
    
    res = 2* Sqdet
    res += c12 * math.pi
    res += math.atan(c12/Sqdet)*2*c12
    res /= 8 * math.pi
    return res
def J2_relu(C):
    """for relu analytical value of  <x1 g(x2)>"""
    return C[0,1] / 2.
def I2_linear(C):
    """ for linear analytical value of <g(x1)g(x2)>"""
    return C[0,1]
def J2_linear(C):
    """for linear analytical value of  <x1 g(x2)>"""
    return C[0,1]
def I21_linear(C):
    """ for linear analytical value of <g'(x1) x1 x2>"""
    return C[0,1]

def I22_linear(C):
    """ for linear analytical value of <g'(x1) x2^2 >"""
    return C[1,1]

def I3_linear(C):
    """for erf analytical value of  <g'(x1) x2 g(x3)>"""
    return C[1,2]

def get_Integrals(activation_function):
    if activation_function == "erf":
        return J2_erf , I2_erf , I21_erf, I22_erf, I3_erf
    elif activation_function == "linear":
        return J2_linear,I2_linear,I21_linear,I22_linear,I3_linear

def print_order_parameters(K,*args):
    msg = ""
    for op in args:
        for k, l in product(range(K), range(K)):
            msg += ",%g" %op[k,l]
    return msg[1:]

def pca_errors(low_rank, evecs, xs):
    D, _ = evecs.shape
    pca_errors = [None] * low_rank
    for m in range(1, low_rank + 1):
        projections = evecs[:, -m:].T
        reconstructions =  xs @ projections.T @ projections 
        pca_errors[m-1] = torch.nn.MSELoss()(reconstructions , xs) / 2
    return pca_errors

def compute_order_parameters_from_weights(W, V, Omega):
    D , _ = Omega.shape
    Q1 = W@Omega@W.T/D
    R1 = V@Omega@W.T/D**1.5
    T1 = V@Omega@V.T/D**2
    Q0 = W@W.T/D
    R0 = W@V.T/D
    T0 = V@V.T/D
    return Q0, R0, T0, Q1, R1, T1

def compute_order_parameters(student, Omega):
    W , V = student.fce.weight.data.detach().numpy() , student.fcd.weight.data.detach().numpy().T
    return compute_order_parameters_from_weights(W, V, Omega)

def get_densities_from_weights(W, V, Omega, psis, rhos):
    D , _ = Omega.shape
    Wtau = W@psis/math.sqrt(D) ## K x D 
    K = Wtau.shape[0]
    Vtau = V@psis/math.sqrt(D) ## K x D
    q, r, t = np.zeros((K,K,D)), np.zeros((K,K,D)), np.zeros((K,K,D))
    for k in range(K):
        for l in range(K):
            q[k,l] = Wtau[k] * Wtau[l]
            r[k,l] = Vtau[k] * Wtau[l] / math.sqrt(D)
            t[k,l] = Vtau[k] * Vtau[l] / D
    return q,r,t
    
def get_densities(student, Omega, psis, rhos):
    W , V = student.fce.weight.data.detach().numpy() , student.fcd.weight.data.detach().numpy().T ## K x D , K x D
    return get_densities_from_weights(W, V, Omega, psis, rhos)


def compute_test_error(T0, Q1, R1, T1, Omega,J2, I2):
    D,_ = Omega.shape
    K , _ = Q1.shape
    C2Vec = np.zeros((K,K, K))
    C2Mat = np.zeros((K,K, K, K))
    
    
    for k in range(K):
        C2Vec[k,0,0],C2Vec[k,0,1],C2Vec[k,1,1] = T1[k,k], R1[k,k], Q1[k,k]
        for l in range(K):
            C2Mat[k,l,0,0],C2Mat[k,l,0,1],C2Mat[k,l,1,1] = Q1[k,k], Q1[k,l], Q1[l,l]
    J2m = sum([J2(C2Vec[k]) for k in range(K)])
    I2m = np.asarray([I2(C2Mat[k,l])* T0[k,l] for k, l in product(range(K),range(K))]) ## of size k,l
    
    error  = 0.5*np.trace(Omega)/D
    error += 0.5*np.sum(I2m)
    error -= np.sum(J2m)
    return error , (0.5*np.trace(Omega)/D, np.sum(J2m), 0.5*np.sum(I2m) )


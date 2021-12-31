from analyticalUtils import *
from itertools import product, combinations_with_replacement
import multiprocessing as mp



def dt_comp(k,l, q,r,t, rhos,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated):
    K, _ = T1.shape
    
    if truncated : idx_lim = k+1
    else: idx_lim = K
        
    C2 = np.zeros((2,2))
    C2[0,0] , C2[0,1] , C2[1,1] = Q1[k,k], Q1[k,k], Q1[k,k]
    dd = J2(C2) / Q1[k,k] * rhos * r[l,k] 
    
    C2vec = np.zeros((idx_lim,2,2))
    for a in range(idx_lim):
        C2vec[a,0,0] , C2vec[a,0,1] , C2vec[a,1,1] = Q1[k,k], Q1[k,a], Q1[a,a]
    dd -= sum([I2(C2vec[a]) * t[a,l] * D for a in range(idx_lim)])
    return dd
    
def update_t(q,r,t, rhos,T0,Q1,R1,T1,lr,K,D,I2,I21,I22,I3,J2, pool,truncated = False):
    
    dkl = [pool.apply(dt_comp, args = (k,l,q,r,t, rhos,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated) ) for k, l in combinations_with_replacement( range(K), 2) ]  ## only updates the l>k components
    dlk = [pool.apply(dt_comp, args = (l,k,q,r,t, rhos,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated) ) for k, l in combinations_with_replacement( range(K), 2) ] 
    
    indices = np.triu_indices_from(T0)
    dkl = lr * np.concatenate( dkl ).reshape(len(indices[0]), t.shape[-1]  )
    dlk = lr * np.concatenate( dlk ).reshape(len(indices[0]), t.shape[-1]  )
    dt = np.zeros(t.shape)
    
    dt[indices] = dkl + dlk
    
    for k in range(K):
        for l in range(k):
            dt[k,l] = dt[l,k]

    return dt / D


def dt0_comp(k,l, T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated):
    dd = 0
    K, _ = T1.shape
    if truncated : idx_lim = k+1
    else: idx_lim = K
    C2 = np.zeros((2,2))
    C2[0,0] , C2[0,1] , C2[1,1] = Q1[k,k], Q1[k,k], Q1[k,k] #T1[l,l], R1[l,k], Q1[k,k]
    dd = J2(C2) / Q1[k,k] * R1[l,k]
    
    C2vec = np.zeros((idx_lim,2,2))
    for a in range(idx_lim):
        C2vec[a,0,0] , C2vec[a,0,1] , C2vec[a,1,1] = Q1[k,k], Q1[k,a], Q1[a,a]
    dd -= sum([I2(C2vec[a]) * T0[a,l] for a in range(idx_lim)])
    return dd
    
def update_t0(T0,Q1,R1,T1,lr,K,D,I2,I21,I22,I3,J2, pool,truncated = False):
    
    dkl = [pool.apply(dt0_comp, args = (k,l,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated) ) for k, l in combinations_with_replacement( range(K), 2) ]  ## only updates the l>k components
    dlk = [pool.apply(dt0_comp, args = (l,k,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated) ) for k, l in combinations_with_replacement( range(K), 2) ] 
    
    dkl = lr * np.asarray( dkl )
    dlk = lr * np.asarray( dlk )
    dt = np.zeros(T0.shape)
    indices = np.triu_indices_from(T0)
    dt[indices] = dkl + dlk
    
    for k in range(K):
        for l in range(k):
            dt[k,l] = dt[l,k]
    
    return dt

def dq_comp(k,l,q,r,t, rhos,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated):
    
    K, _ = Q1.shape

    if truncated : idx_lim = k+1
    else: idx_lim = K
        
    ### line 1 ####
    C2 = np.zeros((2,2))
    C2[0,0],C2[0,1],C2[1,1] = Q1[k,k], R1[k,k], T1[k,k]
    A = 0
    det = Q1[k,k]* T1[k,k] - R1[k,k]**2
    
    C  = I21(C2) *( T1[k,k]*q[k,l] -  R1[k,k]*r[k,l]) 
    C += I22(C2) *( Q1[k,k]*r[k,l] -  R1[k,k]*q[k,l])
    C /= det
    
    for a in range(idx_lim):
        Aa = 0
        if a==k: continue
        det=Q1[k,k]*Q1[a,a]-Q1[k,a]**2
        C3 = np.zeros((3,3))
        C3[0,0] = Q1[k,k]
        C3[0,1] = Q1[k,k]
        C3[1,0] = Q1[k,k]
        C3[1,1] = Q1[k,k]
        C3[0,2] = Q1[k,a]
        C3[2,0] = Q1[k,a]
        C3[1,2] = Q1[k,a]
        C3[2,1] = Q1[k,a]
        C3[2,2] = Q1[a,a]
        Aa  = I3(C3) * (Q1[a,a]*q[k,l] - Q1[k,a]*q[a,l])
        C3[0,1] = Q1[k,a]
        C3[1,0] = Q1[k,a]
        C3[1,1] = Q1[a,a]
        C3[1,2] = Q1[a,a]
        C3[2,1] = Q1[a,a]
        Aa += I3(C3) * (Q1[k,k]*q[a,l]  -  Q1[k,a]*q[k,l] ) 
        Aa *= T0[k,a]/det
        A  -= Aa
    
    C3 = np.ones((3,3)) * Q1[k,k]
    B  = -T0[k,k]/Q1[k,k] * I3(C3) * q[k,l]
    return  (A+B+C)*rhos
    
def update_q(q,r,t, rhos,T0,Q1,R1,T1,lr,K,D,I2,I21,I22,I3,J2, pool,truncated = False):
    
    dkl = [pool.apply(dq_comp, args = (k,l,q,r,t, rhos,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated) ) for k, l in combinations_with_replacement( range(K), 2) ]  ## only updates the l>k components
    dlk = [pool.apply(dq_comp, args = (l,k,q,r,t, rhos,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated) ) for k, l in combinations_with_replacement( range(K), 2) ] 
    
    indices = np.triu_indices_from(T0)
    dkl = lr * np.concatenate( dkl ).reshape(len(indices[0]), t.shape[-1]  )
    dlk = lr * np.concatenate( dlk ).reshape(len(indices[0]), t.shape[-1]  )
    dq = np.zeros(t.shape)
    
    dq[indices] = dkl + dlk
    for k in range(K):
        for l in range(k):
            dq[k,l] = dq[l,k]         
    return dq/D

def dr_dv_comp(k,l,q,r,t, rhos,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated):
    # dv^k w^l D 
    
    if truncated : idx_lim = k+1
    else: idx_lim = K
        
    dd = 0
    K, _ = R1.shape
    C2 = np.zeros((2,2))
    C2[0,0] , C2[0,1], C2[1,0] , C2[1,1] = Q1[k,k], Q1[k,k], Q1[k,k], Q1[k,k]
    dd += J2(C2) / Q1[k,k] * q[k,l] * rhos/D
    
    for a in range(idx_lim):
        C2[0,0] , C2[0,1], C2[1,0] , C2[1,1] = Q1[k,k], Q1[k,a], Q1[k,a], Q1[a,a]
        dd -= I2(C2)* r[a,l] 
    return dd
    
def dr_dw_comp(k,l,q,r,t, rhos,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated):
    K, _ = Q1.shape
    
    if truncated : idx_lim = l+1
    else: idx_lim = K
        

    ### line 1 ####
    C2 = np.zeros((2,2))
    C2[0,0],C2[0,1], C2[1,0],C2[1,1] = Q1[l,l], R1[l,l], R1[l,l], T1[l,l]
    
    
    det = Q1[l,l]* T1[l,l] - R1[l,l]**2
    C  = I21(C2)* ( T1[l,l] *  r[k,l] - R1[l,l] * t[k,l])
    C += I22(C2)* ( Q1[l,l] *  t[k,l] - R1[l,l] * r[k,l])
    C /= det
    
    A = 0
    for a in range(idx_lim):
        Aa = 0
        if a==l: continue
            
        det=Q1[l,l]*Q1[a,a]-Q1[l,a]**2
        
        C3 = np.zeros((3,3))
        C3[0,0] = Q1[l,l]
        C3[0,1] = Q1[l,l]
        C3[1,0] = Q1[l,l]
        C3[1,1] = Q1[l,l]
        C3[0,2] = Q1[l,a]
        C3[2,0] = Q1[l,a]
        C3[1,2] = Q1[l,a]
        C3[2,1] = Q1[l,a]
        C3[2,2] = Q1[a,a]
        
        Aa  = I3(C3) * ( Q1[a,a] * r[k,l] - Q1[l,a] * r[k,a])
        
        C3[0,1] = Q1[l,a]
        C3[1,0] = Q1[l,a]
        C3[1,1] = Q1[a,a]
        C3[1,2] = Q1[a,a]
        C3[2,1] = Q1[a,a]
        
        Aa += I3(C3) * ( Q1[l,l] * r[k,a] - Q1[l,a] * r[k,l])
        Aa *= T0[l,a]/det
        A  -= Aa
    
    C3 = np.ones((3,3))* Q1[l,l]
    B  = -T0[l,l]/Q1[l,l] * I3(C3) * r[k,l]
    return  (A+B+C)*rhos/D
    
def update_r(q,r,t, rhos,T0,Q1,R1,T1,lr,K,D,I2,I21,I22,I3,J2, pool,truncated = False):    
    
    resw = [pool.apply(dr_dw_comp, args = (k,l,q,r,t, rhos,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated)) for k, l in product( range(K), range(K)) ] 
    resv = [pool.apply(dr_dv_comp, args = (k,l,q,r,t, rhos,T0,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated)) for k, l in product( range(K), range(K)) ] 
    
    resw = lr * np.concatenate( resw ).reshape(r.shape)
    resv = lr * np.concatenate( resv ).reshape(r.shape)
    dr = resw + resv

    return dr

def update_TBulk_comp(k,l, T1bulk,Q1,R1,T1,K,D,I2,I21,I22,I3,J2, truncated):
    dd = 0
    K, _ = T1.shape
    
    if truncated : idx_lim = k+1
    else: idx_lim = K
        
    C2vec = np.zeros((idx_lim,2,2))
    for a in range(idx_lim) :
        C2vec[a,0,0] , C2vec[a,0,1] , C2vec[a,1,1] = Q1[k,k], Q1[k,a], Q1[a,a]
    dd = - sum([I2(C2vec[a]) * T1bulk[a,l] for a in range(idx_lim)])
    return dd
    
def update_TBulk(T1bulk,Q1,R1,T1,lr,K,D,I2,I21,I22,I3,J2, pool,truncated = False):
    
    dkl = [pool.apply(update_TBulk_comp, args = (k,l,T1bulk,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated) ) for k, l in combinations_with_replacement( range(K), 2) ]  ## only updates the l>k components
    
    dlk = [pool.apply(update_TBulk_comp, args = (l,k,T1bulk,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated) ) for k, l in combinations_with_replacement( range(K), 2) ] 
    
    dkl = lr * np.asarray( dkl )
    dlk = lr * np.asarray( dlk )
    dt = np.zeros(T1bulk.shape)
    indices = np.triu_indices_from(T1bulk)
    dt[indices] = dkl + dlk
    for k in range(K):
        for l in range(k):
            dt[k,l] = dt[l,k]
    return dt

def update_RBulk_comp(k,l,R1bulk,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated):
    # dv^k w^l
    dd = 0
    K, _ = R1.shape
    
    if truncated : idx_lim = k+1
    else: idx_lim = K
        
    C2vec = np.zeros((idx_lim,2,2))
    for a in range(idx_lim):
        C2vec[a,0,0] , C2vec[a,0,1] , C2vec[a,1,1] = Q1[k,k], Q1[k,a], Q1[a,a]
    dd = - sum([I2(C2vec[a]) * R1bulk[a,l] for a in range(idx_lim)])   
    return dd
    
def update_RBulk(R1bulk,Q1,R1,T1,lr,K,D,I2,I21,I22,I3,J2, pool,truncated = False):    
    resv = [pool.apply(update_RBulk_comp, args = (k,l,R1bulk,Q1,R1,T1,K,D,I2,I21,I22,I3,J2,truncated)) for k, l in product( range(K), range(K)) ] 
    dR = lr * np.asarray( resv ).reshape(R1bulk.shape)
    return dR
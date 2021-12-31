import sys
sys.path.insert(1, "src/")
from utils import *
from analyticalUtils import *
from analytical_updates import *
from various_manual_weight_updates import *
import argparse

data_root = '~/datasets'
bs = 1; test_bs = 10000 ; test_datasize = 10000;
datasize = int(1e4)


def test(w, v,g, dg, test_xs):
    K, D = w.shape
    if v is None: 
        reconstructions = g( test_xs @ w.T / math.sqrt(D) )@ w 
    else:
        reconstructions = g( test_xs @ w.T / math.sqrt(D) ) @ v 
    errors = torch.nn.MSELoss()(reconstructions , test_xs)
    return errors

def update_data(data,logfname, steps, w_erf, v_erf, w_sanger, w_erf_truncated, v_erf_truncated, eg_erf, eg_sanger, eg_erf_truncated, eg_ana, eg_ana_truncated):
    """usefull to update the dictionary which saves the data"""
    data["steps"], \
    data["w_erf"], data["v_erf"], \
    data["w_erf_truncated"], data["v_erf_truncated"],\
    data["w_sanger"], \
    data["eg_erf"], data["eg_sanger"], data["eg_erf_truncated"], data["eg_ana"], data["eg_ana_truncated"] = steps, \
    {"0" : w_erf[0], "-1" : w_erf[-1]}, \
    {"0" : v_erf[0], "-1" : v_erf[-1]}, \
    {"0" : w_erf_truncated[0], "-1" : w_erf_truncated[-1]}, \
    {"0" : v_erf_truncated[0], "-1" : v_erf_truncated[-1]}, \
    {"0" : w_sanger[0], "-1" : w_sanger[-1]},\
    eg_erf, eg_sanger, eg_erf_truncated, eg_ana, eg_ana_truncated
    torch.save(data, logfname)  
    
def main(args):
    
    ################################################## Parameters ############################################################
    g_name = args.gname; 
    if args.analytical==1: # perform the analytucal updates
        Integrals =  get_Integrals(g_name)
    else: 
        Integrals = None   
        
    low_rank = args.low_rank
    if args.K is None:
        K = low_rank # for simplificity we use matched case if not given
    else:
        K = args.K
        
    D = args.D
    std0 = args.std0
    NUM_TESTSAMPLES = args.test_bs
    lr = args.lr
    num_steps = args.steps * D
    
    torch.manual_seed(args.seed) # for reproducibility
    
    loc = (args.loc+ "/" if args.loc !="" else "")
    if args.dataset is None:
        logfname = loc + "Plot2_D%d_K%d_LR%d_std0%.2f_lr%.2f_steps%d_s%d.pyT"%(D, K, low_rank,std0, lr, args.steps, args.seed)
    else:
        logfname = loc + "Plot2_%s_K%d_LR%d_std0%.2f_lr%.2f_steps%d_s%d.pyT"%(args.dataset, K, low_rank,std0, lr, args.steps, args.seed)
    print("Hello and welcome to this module where I train an autoencoder with manual weight updates")
    print("I am saving in %s"%logfname)
    data = {}
    data["args"] = args
    g, dg = getActfunction(g_name)
    
    ################################################## SAMPLING ############################################################
    ## creates the generative model x = A c + xi
    if args.dataset is None or args.dataset  == "sinusoidal":
        print(f"creating a generative model")
        D = args.D
        test_xs_raw = createGenerativeModel(args.dataset, D, low_rank, 1., NUM_TESTSAMPLES) 
        raw_mean = test_xs_raw.mean()
        raw_std = test_xs_raw.std()
        
    else: # if training on "realistic" datasets : CIFAR10 or FashionMNIST
        trainloader, testloader, D = getDataset(args.dataset, data_root, bs, test_bs, None, args.datasize, False,False)
        low_rank = D ## there is no clear spike bulk separation
        test_xs_raw , _  = iter(testloader).next()
        raw_std = torch.ones(1) # the inputs are already normalised and centered so no need to have the shift
        raw_mean = torch.zeros(1)
    
    test_xs = (test_xs_raw - raw_mean)  / raw_std
    covariance    = test_xs_raw.t()@test_xs_raw / test_bs
    evals, evecs  = torch.symeig( covariance, eigenvectors=True) # obtains eigenvalues and eigenvectors 

    data["evals"] = evals
    data["evecs"] = evecs    
    
    ################################################## PCA ############################################################
    if args.dataset is None:
        pca_errors = [None] * low_rank
    else:
        pca_errors = [None] * K
    for m in range(1, len(pca_errors) + 1):
        projections = evecs[:, -m:].T * math.pow(D, 0.25)
        pca_errors[m-1] = test(projections,None,lambda x: x,lambda x: x*x**-1, test_xs).item()
    data["PCA"] = pca_errors
    
    ################################################## Train ############################################################
    w0 = std0 * torch.randn(K, D) # initialises the encoder's weights 
    v0 = std0 * torch.randn(K, D) # initialises the decoder's weights 
    
    Integrals =  get_Integrals(g_name)
    analytical_updates  = args.analytical==1
    if analytical_updates:
        J2 , I2 , I21, I22, I3 = Integrals
        ## the numpy quantities are usefull for analytical
        Omega= covariance.detach().numpy() / raw_std.item()**2 ## this is because I am dividing by the std the dataset
        psis = evecs.detach().numpy() * np.sqrt(D)
        rhos = evals.detach().numpy() / raw_std.item()**2
        pool = mp.Pool(mp.cpu_count())
    
    steps = []; 
    w_erf = [w0.clone().detach()] ; v_erf = [v0.clone().detach()]
    w_sanger = [w0.clone().detach()] ; 
    w_erf_truncated = [w0.clone().detach()] ; v_erf_truncated = [v0.clone().detach()]
    eg_erf_truncated  = []; eg_erf = []; eg_sanger = []
    eg_ana = []; eg_ana_truncated = [];
    
    if analytical_updates:
        ## defines the bulk order parameters
        Q0, R0, T0, Q1, R1, T1 = compute_order_parameters_from_weights(w_erf[-1].detach().numpy(), v_erf[-1].detach().numpy(), Omega)
        q,r,t = get_densities_from_weights(w_erf[-1].detach().numpy(), v_erf[0].detach().numpy(), Omega, psis, rhos)
        ## selects only the spikes
        qana,rana,tana                  = q, r, t
        Q1ana, R1ana, T1ana             = Q1, R1, T1
        T0ana                           = T0
        
        ## truncated algorithm
        _,_, T0, Q1, R1, T1 = compute_order_parameters_from_weights(w_erf_truncated[-1].detach().numpy(), v_erf_truncated[-1].detach().numpy(), Omega)
        q,r,t = get_densities_from_weights(w_erf_truncated[-1].detach().numpy(), v_erf_truncated[0].detach().numpy(), Omega, psis, rhos)
        ## selects only the spikes
        qana_tr,rana_tr,tana_tr                  = q, r, t
        Q1ana_tr, R1ana_tr, T1ana_tr             = Q1, R1, T1
        T0ana_tr                                 = T0
    
    update_data(data,logfname, steps, w_erf, v_erf, w_sanger, w_erf_truncated, v_erf_truncated, eg_erf, eg_sanger, eg_erf_truncated, eg_ana, eg_ana_truncated)
    
    
    step = 0
    dstep = 1./D
    num_prints = 200
    
    end = torch.log10(torch.tensor([1. * num_steps])).item()
    steps_to_print = list(torch.logspace(-2, end, steps = num_prints))
    steps_to_save = steps_to_print[::3]
    print("I am going to print for %d steps"%len(steps_to_print))
    
    while step <= num_steps:# num_steps :
        if torch.any(torch.isnan(w_erf[-1])) or torch.any(torch.isnan(v_erf[-1])):
            print("is nan")
            break
        
        if step/D >= steps_to_print[0]:
            
            steps_to_print.pop(0)
            msg = f"{step/D}"
            steps += [step/D]
            ## full update 
            preds = g( test_xs @ w_erf[-1].t() / math.sqrt(D) )  @ v_erf[-1]
            eg_erf    += [  F.mse_loss(preds , test_xs) * 0.5 ]
            
            ## truncated update 
            preds = g( test_xs @ w_erf_truncated[-1].t() / math.sqrt(D) )  @ v_erf_truncated[-1]
            eg_erf_truncated    += [  F.mse_loss(preds , test_xs) * 0.5 ]
            
            msg += f", {eg_erf[-1] },{eg_erf_truncated[-1]}"
            
            if analytical_updates:
                analytical_loss, _  = compute_test_error(T0ana, Q1ana, R1ana, T1ana, Omega,J2, I2)
                msg += ",%g"%(analytical_loss)
                eg_ana += [analytical_loss]   

                analytical_loss, _  = compute_test_error(T0ana_tr, Q1ana_tr, R1ana_tr, T1ana_tr, Omega,J2, I2)
                msg += ",%g"%(analytical_loss)
                eg_ana_truncated += [analytical_loss]   
                
            update_data(data,logfname, steps, w_erf, v_erf, w_sanger, w_erf_truncated, v_erf_truncated, eg_erf, eg_sanger, eg_erf_truncated, eg_ana, eg_ana_truncated)
            print(msg)
        
        # new input
        if args.dataset is None or args.dataset == "sinusoidal":
            x = ( sample(1, evals, evecs) - raw_mean) / raw_std ## would be equivalent to getting the raw 
        else:
            x , _ = iter(trainloader).next()
            x = ( x - raw_mean) / raw_std
        
        dw, dv, rec2 = dw_mse(w_erf[-1], v_erf[-1] , x, g, dg)
        w_erf[-1] = w_erf[-1] - lr / D  * dw
        v_erf[-1] = v_erf[-1] - lr * dv
        
        dw, dv, rec2 = dw_mse_truncated(w_erf_truncated[-1], v_erf_truncated[-1] , x, g, dg)
        w_erf_truncated[-1] = w_erf_truncated[-1] - lr / D * dw
        v_erf_truncated[-1] = v_erf_truncated[-1] - lr * dv
        
        if analytical_updates:
            
            ### VALUES OF SIMULATIONS ###
            for k, l in product(range(K),range(K)):
                Q1ana[k,l] = np.sum(rhos*qana[k,l]) / D
                R1ana[k,l] = np.sum(rhos*rana[k,l]) / D
                T1ana[k,l] = np.sum(rhos*tana[k,l]) / D
            
            dt  = update_t(qana,rana,tana,rhos,T0ana,Q1ana,R1ana,T1ana,args.lr,K,D,I2,I21,I22,I3,J2, pool)
            dq  = update_q(qana,rana,tana,rhos,T0ana,Q1ana,R1ana,T1ana,args.lr,K,D,I2,I21,I22,I3,J2, pool)
            dr  = update_r(qana,rana,tana,rhos,T0ana,Q1ana,R1ana,T1ana,args.lr,K,D,I2,I21,I22,I3,J2, pool)
            dt0 = update_t0(T0ana,Q1ana,R1ana,T1ana,args.lr,K,D,I2,I21,I22,I3,J2, pool)
            
            tana  += dt  * dstep
            T0ana += dt0 * dstep
            qana  += dq  * dstep
            rana  += dr  * dstep 
            
            ### Truncated Algorithm ###
            for k, l in product(range(K),range(K)):
                Q1ana_tr[k,l] = np.sum(rhos*qana_tr[k,l]) / D  
                R1ana_tr[k,l] = np.sum(rhos*rana_tr[k,l]) / D  
                T1ana_tr[k,l] = np.sum(rhos*tana_tr[k,l]) / D  
            dt_tr  = update_t(qana,rana,tana,rhos,T0ana,Q1ana,R1ana,T1ana,args.lr,K,D,I2,I21,I22,I3,J2, pool, True)
            dq_tr  = update_q(qana,rana,tana,rhos,T0ana,Q1ana,R1ana,T1ana,args.lr,K,D,I2,I21,I22,I3,J2, pool, True)
            dr_tr  = update_r(qana,rana,tana,rhos,T0ana,Q1ana,R1ana,T1ana,args.lr,K,D,I2,I21,I22,I3,J2, pool, True)
            dt0_tr = update_t0(T0ana,Q1ana,R1ana,T1ana,args.lr,K,D,I2,I21,I22,I3,J2, pool, True)
            
            tana_tr  += dt_tr  * dstep
            T0ana_tr += dt0_tr * dstep
            qana_tr  += dq_tr  * dstep
            rana_tr  += dr_tr  * dstep
        step += 1
    print("Thank you!")

if __name__ == '__main__':
    
     # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-K', '--K', metavar='K', type=int, default=None,
                        help="size of the student's intermediate layer")
    parser.add_argument('-D', '--D', type=int, default=500,
                        help="dimension")
    parser.add_argument('-dataset', '--dataset', metavar='dataset', type=str, default=None,
                        help="Dataset, cifar10_gray")
    parser.add_argument('-low_rank', '--low_rank', type=int, default=2,
                        help="number of spikes")
    parser.add_argument('-test_bs', '--test_bs', type=int, default=int(1e4),
                        help="number of test samples")
    parser.add_argument('-lr', '--lr', type=float, default=0.4,
                        help="learning rate")
    parser.add_argument('-gname', '--gname', type=str, default="erf",
                        help="acitivation functions")
    parser.add_argument('-std0', '--std0', type=float, default = 0.1,
                        help="number of test samples")
    parser.add_argument('-steps', '--steps', type=int, default=50,
                        help="number of sgd steps")
    parser.add_argument('-seed', '--seed', type=int, default=0,
                        help="random seed")
    parser.add_argument('-loc', '--loc', metavar='loc', type=str, default="",
                        help="location to save")
    parser.add_argument('-analytical', '--analytical', type=int, default=0,
                        help="location to save")
    
    args = parser.parse_args()
    main(args)
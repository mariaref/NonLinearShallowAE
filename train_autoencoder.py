#!/usr/bin/env python3
#
# Auto-encoder on structured data
#
# Date: April 2021
#
# Author: Maria Refinetti <mariaref@gmail.com>
import sys
sys.path.insert(1, "src/")
from utils import *
from models import *
from analyticalUtils import *
from analytical_updates import *
from various_manual_weight_updates import *

import argparse
import multiprocessing as mp
data_root = '~/datasets'
noise = 1.
##default arguments! check when you want to change them ###
bs = 1; test_bs = 10000 ; test_datasize = 10000;
MontecarloBS = 10000;

def test(student, data, criterion, D, device, w, v):
    if w is None and v is None:
        data = data.view(-1,D)
        if device is not None: data.to(device)
        prediction = student(data)
        total_loss = criterion(prediction, data)
    elif v is None:
        reconstructions = data @ w.T @ w / math.sqrt(D)
        total_loss =  criterion(reconstructions, data).item()
    else:
        reconstructions = student.g( data @ w.t() / math.sqrt(D) )  @ v
        total_loss = criterion(reconstructions, data).item()
    return total_loss

def train_online(student,x, criterion, D,device,wd,optimizer): 
    student.train() 
    optimizer.zero_grad()
    prediction = student(x)
    loss =  criterion(prediction, x)
    loss.backward()
    optimizer.step()
    # manual addition of weight decay
    with torch.no_grad():
        if wd>0:
            student.fce.weight.data -= wd/D * student.fce.weight.data ##
            student.fcd.weight.data -= wd/D * student.fcd.weight.data  ##
    if torch.any(torch.isnan(student.fce.weight.data)):
        print("attention! weight is NAN!!")
        raise NotImplementedError

    
def main(args, logfname):
    
    device = torch.device("cpu")        
    torch.manual_seed(args.seed)
    online = ( args.online==1 ) # to train online or minibatch>1 with finite dataset
    
    # analytical tools to integrate ODEs
    g, dg = getActfunction(args.activation_function)
    J2 , I2 , I21, I22, I3 =  get_Integrals(args.activation_function)
    
    bias = (args.bias0>0) ## bias0 defines the initial value of the biases if bias0==0: do not train the biases
    logfname=logfname[:-4]+"_bias%g.dat"%args.bias0
    
    # dataset
    
    if args.dataset is None or args.dataset  == "sinusoidal":
        print(f"creating a generative model")
        D = args.D
        low_rank = args.K
        xs_raw = createGenerativeModel(args.dataset, D, low_rank, noise, test_bs) 
        if args.dataset is None:
            raw_std = 1.
            raw_mean = 0.
        elif args.dataset  == "sinusoidal":
            raw_mean = xs_raw.mean().item()
            raw_std = xs_raw.std().item()        
    else: # if training on "realistic" datasets : CIFAR10 or FashionMNIST
        trainloader, testloader, D = getDataset(args.dataset, data_root, bs, test_bs, None, args.datasize, False,False)
        low_rank = D ## there is no clear spike bulk separation
        xs_raw , _  = iter(testloader).next()
        raw_std  = 1. # the inputs are already normalised and centered so no need to have the shift
        raw_mean = 0.
        
    covariance    = xs_raw.t()@xs_raw / test_bs
    evals, evecs  = torch.symeig( covariance, eigenvectors=True) # obtains eigenvalues and eigenvectors 
    if torch.min(evals)<0:
        print("###################### some eigenvalues are smaller than 0! shifting by %g"%torch.min(evals))
        evals += torch.abs(torch.min(evals)) # for numerical stability
    

    xs = (xs_raw - raw_mean) / raw_std  ## note that by doing so our dataset now has covariance matrix Omega/std**2     
    print( f"the std of my data is {xs.std()}")
    print( f"the norm^2 of my data is {xs.norm(dim=1).mean()**2}")   
    
    Omega= covariance.detach().numpy()/ raw_std**2  ## this is because I am dividing by the std the dataset
    psis = evecs.detach().numpy() * np.sqrt(args.D) ## eigenvectors scaled as in main text
    rhos = evals.detach().numpy()  / raw_std**2     ## eigenvalues 

    ### for logging
    welcome = ("#  D=%d, K=%d, lr=%g, seed=%d, bias=%d" % (D, args.K, args.lr, args.seed, int(bias)))
    msg = ("# Using device: %s" % device)
    
    ## create a one-hidden layer autoencoder
    student = AutoEncoder(D, args.K, tied=(args.tied==1), g=g, scale = (args.optimizer=="sgd"), init = args.init, bias = args.bias0)
    
    # loads the weights from the prefix file
    time0 = 0 # number of SGD steps at which trining starts
    if args.prefix is not None:
        try: # statements that can raise exceptions
            run = torch.load(args.prefix)
        except: # statements that will be executed to handle exceptions
            print("The prefix is not valid, starting with random weights");
            msg += "# The prefix is not valid, starting with random weights"
            return
        else: # statements that will be executed if there is no exception
            weights = run['weights']
            time0 = 'end'
            logfname = logfname[:-4]+f"_i{time0}.dat"
            msg += "# starting with weights from %s"%args.prefix
            student.load_state_dict(weights[-1])
            
    if args.initialisation == "informed": # starts from informed initialisation where weights are proportional to data eigenvectors
        msg += "# starting from an informed solution"
        # W K x D # V D x K
        student.fce.weight.data = psis[:,-args.K:].t().detach().clone()/math.sqrt(D)
        student.fcd.weight.data = psis[:,-args.K:].detach().clone()
    
    ### if train using sanger's training rule ######
    if args.sanger == 1:
        w = student.fce.weight.data.clone().detach()
        v = None
    else: 
        w = None
        v = None
    student.to(device)
    
    ## defines the loss that is to be used
    criterion = lambda x, y: 0.5 * nn.MSELoss()(x, y)
    
    ## optimiser
    if args.optimizer == 'sgd':
        if args.tied==0:
            params = []
            if args.bias0>0:
                params += [{'params': student.fce.weight}]
                params += [{'params': student.fce.bias,               'lr': args.lr}]
            else:
                params += [{'params': student.fce.parameters()}]
            params += [{'params': student.fcd.parameters(),
                        'lr': args.lr}]
            optimizer = optim.SGD(params, lr=args.lr/D)   ## note : also bias_lr scales as lr/D
        else:
            params = []
            params += [{'params': student.fce.parameters()}]
            optimizer = optim.SGD(params, lr=args.lr)        
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    else: raise NotImplementedError
    
    ## for saving: 
    PCA_err =  pca_errors(low_rank, evecs, xs)
    weights = {}
    weights[0] = copy.deepcopy(student.state_dict())
    run = {'args':args,'testloss':None,'trainloss':None,\
           'weights':{0: weights[0] , -1 :weights[0]} ,\
           'finished': False, \
           'w': ({0: w[0] , -1 :w[-1]} if w is not None else None) ,\
           'v': ({0: v[0] , -1 :v[-1]} if v is not None else None), \
           'evals': evals , 'evecs' : ( evecs[:,-low_rank:] if low_rank is not None else evecs),\
           'PCA' : PCA_err }
    
    torch.save(run, logfname[:-4] + '.pyT') 
    if args.steps is None: steps = 1e7
    else: steps = args.steps
    end = torch.log10(torch.tensor([1. * steps])).item()
    steps_to_print = list(torch.logspace(-2, end, steps = args.print_every))
    print("I am going to print for %d steps"%len(steps_to_print))
    print("saving in %s"%logfname)
    logfile = open(logfname, "w", buffering=1)
    log(msg, logfile)
    
    step = 0
    dstep = 1./D
    epoch = 0
    
    ### initial value of order parameters for integrations of ODES ########
    if args.analytical_updates==1:
        
        logfile_analytical = open(logfname[:-4]+'_ana.dat', "w", buffering=1)
        pool = mp.Pool(mp.cpu_count()) # multiprocessing to update indices in parallel
        K = args.K
        Q0, R0, T0, Q1, R1, T1 = compute_order_parameters(student, Omega)
        
        
        ## selects the spikes and the bulk
        Bulk   = rhos[:-low_rank]
        Spikes = rhos[-low_rank:]
        
        ## defines the bulk order parameters
        Q1bulk = np.zeros(Q1.shape)
        R1bulk = np.zeros(R1.shape)
        T1bulk = np.zeros(T1.shape)
        q,r,t = get_densities(student, Omega, psis, rhos)
        for k, l in product(range(K),range(K)):
            Q1bulk[k,l] = np.sum(Bulk*q[k,l,:-low_rank]) / D
            R1bulk[k,l] = np.sum(Bulk*r[k,l,:-low_rank]) / D
            T1bulk[k,l] = np.sum(Bulk*t[k,l,:-low_rank]) / D
        
        ## selects only the spikes
        qana,rana,tana = (q.T[-low_rank:]).T, (r.T[-low_rank:]).T , (t.T[-low_rank:]).T
        Q1ana, R1ana, T1ana = Q1, R1, T1
        Q1bulkana, R1bulkana, T1bulkana = Q1bulk,R1bulk,T1bulk
        T0ana = T0
   
    # starts the training
    while len(steps_to_print)>0:
        # prints 
        if step >= steps_to_print[0]:
            student.eval()
            with torch.no_grad():
                test_loss = test(student, xs, criterion, D,device, w, v)
                msg = "%g, %g , %g" % (step, 0, test_loss)     
                if args.K<=10 and (args.tied==0):
                    Q0, R0, T0, Q1, R1, T1 = compute_order_parameters(student, Omega)
                    msg += ", " + print_order_parameters(args.K,T0, Q1, R1, T1) 
                log(msg, logfile,verbose = True)
                if args.analytical_updates==1:
                    analytical_loss, _  = compute_test_error(T0ana, Q1ana, R1ana, T1ana, Omega,J2, I2)
                    msg = "%g, 0, 0,%g"%(step,analytical_loss)
                    msg += ", "+print_order_parameters(args.K,T0ana, Q1ana, R1ana, T1ana)
                    log(msg, logfile_analytical)
                steps_to_print.pop(0)
                
                print("saving")
                run = {'args':args,'testloss':test_loss,\
                       'weights': {0: weights[0] , -1 : copy.deepcopy(student.state_dict())} ,\
                       'finished': False, \
                       'w': ({0: w[0] , -1 :w[-1]} if w is not None else None) ,\
                       'v': ({0: v[0] , -1 :v[-1]} if v is not None else None),\
                       'evals':evals , \
                       'evecs' :( evecs[:,-low_rank:] if low_rank is not None else evecs),\
                      'PCA' : PCA_err }
                torch.save(run, logfname[:-4] + '.pyT') # overwrite
                
        ## analytical updates
        if args.analytical_updates==1:
            
            # computes the OP from the densities
            for k, l in product(range(K),range(K)):
                Q1ana[k,l] = np.sum(Spikes*qana[k,l]) / D   ## the bulk evolves independently form the spikes
                R1ana[k,l] = np.sum(Spikes*rana[k,l]) / D   ## the bulk evolves independently form the spikes
                T1ana[k,l] = np.sum(Spikes*tana[k,l]) / D   ## the bulk evolves independently form the spikes
                if low_rank == D:
                    Q1ana[k,l] += Q1bulkana[k,l]
                    R1ana[k,l] += R1bulkana[k,l]
                    T1ana[k,l] += T1bulkana[k,l]
            if low_rank==D:
                dt  = update_t(qana,rana,tana,rhos,T0ana,Q1ana,R1ana,T1ana,args.lr,args.K,D,I2,I21,I22,I3,J2, pool)
                dq  = update_q(qana,rana,tana,rhos,T0ana,Q1ana,R1ana,T1ana,args.lr,args.K,D,I2,I21,I22,I3,J2, pool)
                dr  = update_r(qana,rana,tana,rhos,T0ana,Q1ana,R1ana,T1ana,args.lr,args.K,D,I2,I21,I22,I3,J2, pool)
                dt0 = update_t0(T0ana,Q1ana,R1ana,T1ana,args.lr,args.K,D,I2,I21,I22,I3,J2, pool)
            
            else:
                # If you do use the spike and bulk simplification, numerics is sensitive to finite D corrections:
                # we integrate each OP independently to improve stability at small D
                # 1) compute the OP from simulations
                Q0, R0, T0, Q1, R1, T1 = compute_order_parameters(student, Omega)
                q,r,t = get_densities(student, Omega, psis, rhos)
                Q1bulk = np.zeros(Q1.shape)
                R1bulk = np.zeros(R1.shape)
                T1bulk = np.zeros(T1.shape)
                for k, l in product(range(K),range(K)):
                    Q1bulk[k,l] = np.sum(Bulk*q[k,l,:-low_rank]) / D
                    R1bulk[k,l] = np.sum(Bulk*r[k,l,:-low_rank]) / D
                    T1bulk[k,l] = np.sum(Bulk*t[k,l,:-low_rank]) / D
                q,r,t = (q.T[-low_rank:]).T, (r.T[-low_rank:]).T , (t.T[-low_rank:]).T
                # 2) updates the spikes independently for each OP 
                dt  = update_t(q,r,tana,Spikes,T0,Q1,R1,T1ana,args.lr,args.K,D,I2,I21,I22,I3,J2, pool)
                dq  = update_q(qana,r,t,Spikes,T0,Q1ana,R1,T1,args.lr,args.K,D,I2,I21,I22,I3,J2, pool)
                dr  = update_r(q,rana,t,Spikes,T0,Q1,R1ana,T1,args.lr,args.K,D,I2,I21,I22,I3,J2, pool)
                dt0 = update_t0(T0ana,Q1,R1,T1,args.lr,args.K,D,I2,I21,I22,I3,J2, pool)
                # 3) updates the bulk independently for each OP 
                dR1bulkana = update_RBulk(R1bulkana,Q1,R1ana,T1,args.lr,args.K,D,I2,I21,I22,I3,J2, pool)
                dT1bulkana = update_TBulk(T1bulkana,Q1,R1,T1ana,args.lr,args.K,D,I2,I21,I22,I3,J2, pool)
            
            tana  += dt  * dstep
            T0ana += dt0 * dstep
            qana  += dq  * dstep
            rana  += dr  * dstep
            
            if low_rank < D:
                R1bulkana += dR1bulkana * dstep
                T1bulkana += dT1bulkana * dstep
            
        ####
        if online: # if online minibatch size is 1 and draw input at all steps
            xs_train =( sample(1, evals, evecs) - raw_mean) / raw_std ## samples one input for online sgd training
        else: # if dataset is finite
            xs_train , _ = iter(trainloader).next()
            xs_train = ( xs_train - raw_mean) / raw_std
        if args.sanger==1: # just in case you are using sanger's algorithm
            w += args.lr / D * dw_sanger(w, xs_train)
        else: # training vanilla SGD
            train_online(student,xs_train, criterion,D,device,args.wd,optimizer ) ## trains for multiple steps while no printing 
        
        step+= dstep
        
    if args.analytical_updates==1: pool.close() # close the pool of workers used for multiprocessing
    
    # save at the end    
    run = {'args':args,'testloss':test_loss,\
           'weights': {0: weights[0] , -1 : copy.deepcopy(student.state_dict())} ,\
           'finished': True, \
           'w': ({0: w[0] , -1 :w[-1]} if w is not None else None)  ,\
           'v': ({0: v[0] , -1 :v[-1]} if v is not None else None) ,
           'evals':evals , \
           'evecs' :( evecs[:,-low_rank:] if low_rank is not None else evecs),\
          'PCA' : PCA_err }
    torch.save(run, logfname[:-4] + '.pyT') # overwrite
        
    goodbye = "# Thank you, please come again"
    log(goodbye, logfile)

if __name__ == '__main__': 
     # read command line arguments
    parser = argparse.ArgumentParser()
    
    ## for the dataset
    parser.add_argument('-D', '--D', metavar='D', type=int, default=None,
                        help="input dimension, is overwritten if dataset is synthetic")
    parser.add_argument('-Lambdas', '--Lambdas', type=str, default="1",
                        help="Value of the spike eigenvalues is eigenvalue = lamda * D")
    parser.add_argument('-datasize', '--datasize', metavar='datasize', type=int, default=None,
                        help="If not online size of the dataset")
    parser.add_argument("-dataset", "--dataset", type=str, default=None,
                        help="dataset if real data, cifar10_gray or fmnist")
    
    ## for the autoencoder
    parser.add_argument('-K', '--K', metavar='K', type=int, default=4,
                        help="size of the student's intermediate layer")
    parser.add_argument('-activation_function', '--activation_function',  type=str, default="erf",
                        help="autoencoder activation function. Default= 'erf'")
    parser.add_argument('-tied', '--tied',type=int, help="tied weights", default = 0)
    parser.add_argument("-bias0", "--bias0", type=float, default=0, 
                        help="If bias0>0 adds the bias and initialises to bias0") 
    
    ## for training
    parser.add_argument("-lr", "--lr", type=float, default=0.1,
                        help="learning constant")
    parser.add_argument("-wd", "--wd", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--steps", type=int, default=int(1e5),
                        help="steps. Default=1e5.")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        help="adam or sgd. Default=sgd.")
    parser.add_argument("-sanger","--sanger", type=int, default=0,
                        help="use sanger rule to update. Default 0")
    parser.add_argument("-online", "--online", type=int, default=1, 
                        help="Train_online, either 0 or 1")   
    
    ## initialisation
    parser.add_argument("-prefix", "--prefix", type=str, default=None,
                        help="file name to load weights from  Default=None")
    parser.add_argument('-init', '--init',  type=float, default=0.1,
                        help="Initial std of the weights. Default 0.1.")
    parser.add_argument('-initialisation', '--initialisation', type=str, default="none",
                        help="Informed : start from an informed solution where the weights are equal to the eigenvectors")
    
    ## additional
    parser.add_argument("-seed", "--seed", type=int, default=0,
                        help="random number generator seed. Default=0")
    
    ## for logging
    parser.add_argument("-loc", "--loc", type=str, default="",
                        help="where to save the datafile. Default=""")
    parser.add_argument("-comment", "--comment", type=str, default="",
                        help="A nice comment to add to the output file. Default=""")
    parser.add_argument("-analytical_error", "--analytical_error", type=int, default=0,
                        help="Do you want me to compute the analytical error? Default=0")
    parser.add_argument("-analytical_updates", "--analytical_updates", type=int, default=0,
                        help="Do you want me to compute the updates analytically? Default=0")
    parser.add_argument("-print_every", "--print_every", type=int, default=50,
                        help="How many times to print. Default=50")
    
    args = parser.parse_args()
    
    loc = args.loc
    if args.loc !="": loc+="/"
    
    if args.online==1:
        ds_str = "_online"
    elif args.datasize is not None:
        ds_str = "_P%d"%args.datasize
    else: ds_str = ""
    
    logfname = loc+ "ae%s%s%s%s_g%s%s%s_K%d_init%g%s_lr%g%s_%s_steps%g_s%d.dat" %(
                  (ds_str),
                  args.comment, 
                  ("_%s"%args.dataset if args.dataset is not None else ""),
                  ("_sanger" if args.sanger==1 else ""),
                   args.activation_function,
                   ("_tied" if (args.tied==1) else ""),
                   ("_D%d"%args.D if args.D is not None else ""),
                   args.K,
                   args.init,
                   ("_informed"  if args.initialisation== "informed" else ""),
                   args.lr,
                   ("_wd%g"%args.wd if args.wd>0 else ""),
                   args.optimizer,
                   args.steps,
                   args.seed
                   )
    
    main(args, logfname)
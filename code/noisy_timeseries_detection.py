import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gamma,gammaln,erf
import itertools as it
from autograd import grad
from autograd.util import quick_grad_check
from scipy.optimize import minimize
from sklearn.metrics import f1_score
import sys
import time
import theano
import theano.tensor as tensor
from rutils import keyboard
import scipy.stats as sts
import unittest
from joblib import Parallel, delayed
from pyhmc import hmc
import dill
from util import *

def dummy_inf_fun(mdl,x,t,y,w,sigma,mu,pi,gamma,g): 
    return mdl.cy_log_likelihood(x,t,y,w,sigma,mu,pi,gamma,g)

class NoiseyTimeseriesDetection:
    def __init__(   self,lambda_0=1.0,mu=0.0,eta=1.0,sigma=1.0,alpha=1.0,beta=1.0,
                    pi=np.array([-100.0,100.0]),max_iter=1000,verbose=0,tol=1e-4,
                    track_score=False,warm_start=None,
                    learn_sigma=True,learn_mu=True,learn_pi=True,backend='theano',
                    gamma=None,K=1,learn_gamma=False,psi_0=np.array([5.0,1.0]),
                    psi_1=np.array([5.0,1.0]),n_restarts=0,pi_offset=0.0,max_observation_distance=1000.0,
                    n_cores=1):
                    
        '''
            Parent class for estimating detectors from temporally noisy labels.
        '''
        self.__dict__.update(locals())
        self.initialize()
        
    def initialize_prior_model(self):
        pass
        
    def initialize(self):
        '''
            Initialize noise model parameters and initialize the prior model.
        '''
        # Initialize parameters
        if self.gamma is None:
            self.gamma = np.zeros((self.K,))
        self.sigma = self.sigma*np.ones((self.K,))
        self.mu = self.mu*np.ones((self.K,))

        # Count the parameters
        self.n_params = self.get_n_prior_params()
        if self.learn_sigma:
            self.n_params += self.K
        if self.learn_mu:
            self.n_params += self.K
        if self.learn_pi:
            self.n_params += 2
        if self.learn_gamma:
            self.n_params += self.K
            
        # Initialize the prior model
        self.initialize_prior_model()
        
    def seperate_params(self,w):
        ''' Separates the noise model and prior model parameters '''
        prior_model_parameters = w
        if self.learn_sigma:
            self.sigma = np.exp(w[-self.K:])
            prior_model_parameters = w[:-self.K]
        if self.learn_mu:
            self.mu = prior_model_parameters[-self.K:]
            prior_model_parameters = prior_model_parameters[:-self.K]
        if self.learn_pi:
            self.pi = prior_model_parameters[-2:]
            prior_model_parameters = prior_model_parameters[:-2]
        if self.learn_gamma:
            self.gamma = prior_model_parameters[-self.K:]
            prior_model_parameters = prior_model_parameters[:-self.K]
        return prior_model_parameters
        
    def build_params(self,prior_model_parameters,sigma,mu,pi,gamma):
        ''' Concatenates all parameters into a vector (i.e. inverts seperate_params)'''
        w = np.zeros(self.n_params)
        w[:prior_model_parameters.shape[0]] = prior_model_parameters
        idx = prior_model_parameters.shape[0]
        if self.learn_gamma:
            w[idx:idx+self.K] = gamma
            idx += self.K
        if self.learn_pi:
            w[idx:idx+2] = pi
            idx += 2
        if self.learn_mu:
            w[idx:idx+self.K] = mu
            idx += self.K
        if self.learn_sigma:
            w[idx:idx+self.K] = np.log(sigma)
            idx += self.K
        assert self.n_params == idx
        return w
        
    def base_classifier_regularizer(self,wf,g=False):
        ''' Default base classifier regularizer (l2 regularization)'''
        if g:
            return 2.0*self.lambda_0*wf
        else:
            return self.lambda_0*np.dot(wf,wf)
    
    def get_theano_log_likelihood(self,g=False):
        ''' If using the theano backend, compiles the symbolic log likelihood 
            into a function and returns said function. 
        
            Arguments:
                g (bool): 
                    if True, return the function for the gradient of the log
                    likelihood instead.
        '''
        # Define symbolic variables
        x_s = tensor.matrix('x')
        t_s = tensor.vector('t')
        y_s = tensor.vector('y')
        w_s = tensor.vector('w')
        sigma_s = tensor.vector('sigma')
        mu_s = tensor.vector('mu')
        pi_s = tensor.vector('pi')
        gamma_s = tensor.vector('gamma')
        
        # Get the symbolic log likelihood from the priod model
        ll = self.symbolic_log_likelihood(x_s,t_s,y_s,w_s,sigma_s,mu_s,pi_s,gamma_s)
        
        # Compile and return the function
        if g:
            # If returning the gradient function, get the sybolic gradients and concat them
            g_w,g_sigma,g_mu,g_pi,g_gamma = theano.grad(ll,[w_s,sigma_s,mu_s,pi_s,gamma_s])
            g_params = g_w
            if self.learn_gamma:
                g_params = tensor.concatenate([g_params,g_gamma])
            
            if self.learn_pi:
                g_params = tensor.concatenate([g_params,g_pi])
                
            if self.learn_mu:
                g_params = tensor.concatenate([g_params,g_mu])
            
            if self.learn_sigma:
                g_params = tensor.concatenate([g_params,g_sigma])
                
            return theano.function(inputs=[x_s,t_s,y_s,w_s,sigma_s,mu_s,pi_s,gamma_s],outputs=g_params,on_unused_input='warn')
        else:
            return theano.function(inputs=[x_s,t_s,y_s,w_s,sigma_s,mu_s,pi_s,gamma_s],outputs=ll,on_unused_input='warn')
    
    def get_obj(self,X,T,Z):
        ''' Get the objective function for use in scipy.optimize.minimize 
            
            Arguments:
                X (list): List of feature sequences (structure will vary with prior model)
                T (list): List of (L_i,) numpy arrays containing instance timestamps
                Z (list): List of observation sequence (structure will vary with prior model)
        
            Returns:
                obj (function): Returns a function that takes a parameter vector and returns the regularized negative log likelihood
        '''
        if self.backend == 'theano':
            ll_fun = self.get_theano_log_likelihood()
            
        def obj(w):
            # init nll value
            f = 0.0
            
            # seperate the parameters
            wf = self.seperate_params(w)
            
            # if using only a single core
            if self.n_cores == 1:
                # for each session, calculate the log marginal likelihood, p(z|x,t)
                for x,t,z in zip(X,T,Z):
                    # if self.backend == 'autograd':
                    #     f += self.log_likelihood(x,t,z,wf)
                    if self.backend == 'theano':
                        f += ll_fun(x,t,z,wf,self.sigma,self.mu,self.pi,self.gamma)
                    elif self.backend == 'cython':
                        f_xy = self.cy_log_likelihood(x,t,z,wf,self.sigma,self.mu,self.pi,self.gamma)
                        f += f_xy
                    

            # if using more than one core, paralellize log marginal likelihood calculations 
            # over self.n_cores cores
            elif self.n_cores > 1:
                if self.backend == 'theano':
                    fs = Parallel(n_jobs=self.n_cores)(delayed(ll_fun)(x,t,z,wf,self.sigma,self.mu,self.pi,self.gamma) for x,t,z in zip(X,T,Z))
                    
                elif self.backend == 'cython':
                    fs = Parallel(n_jobs=self.n_cores)(delayed(dummy_inf_fun)(self,x,t,z,wf,self.sigma,self.mu,self.pi,self.gamma,False) for x,t,z in zip(X,T,Z))
                    
                for fi in fs: 
                    f += fi
                    
            
            # add regularizers
            f = -f + self.base_classifier_regularizer(wf)
            if self.learn_sigma:
                f -= inverse_gamma_log_pdf(self.sigma,self.alpha,self.beta).sum()
            if self.learn_mu:
                f -= normal_log_pdf(self.mu,0.0,self.eta).sum()
            if self.learn_pi:
                f -= beta_log_pdf(1.0-sigmoid(self.pi[0],0.0),self.psi_0[0],self.psi_0[1])
                f -= beta_log_pdf(sigmoid(self.pi[1],0.0),self.psi_1[0],self.psi_1[1])
                
            return f
                
        return obj
    
    def get_grad(self,X,T,Z):
        ''' Get the objective function for use in scipy.optimize.minimize 
            
            Arguments:
                X (list): List of feature sequences (structure will vary with prior model)
                T (list): List of (L_i,) numpy arrays containing instance timestamps
                Z (list): List of observation sequence (structure will vary with prior model)
        
            Returns:
                obj (function): Returns a function that takes a parameter vector and returns the regularized negative log likelihood
        '''
        # if self.backend == 'autograd':
        #     obj = self.get_obj(X,Y)
        #     return grad(obj)   
        if self.backend == 'theano':
            g_fun_s = self.get_theano_log_likelihood(g=True)

        def g_fun(w):
            # init gradient
            g = 0.0
            
            # seperate parameters
            wf = self.seperate_params(w)
            
            # if using a single core
            if self.n_cores == 1:
                for x,t,z in zip(X,T,Z):
                    if self.backend == 'theano':
                        g += g_fun_s(x,t,z,wf,self.sigma,self.mu,self.pi,self.gamma)
                    elif self.backend == 'cython':
                        g += self.cy_log_likelihood(x,t,z,wf,self.sigma,self.mu,self.pi,self.gamma,g=True)
                        
            elif self.n_cores > 1:
                if self.backend == 'theano':
                    gs = Parallel(n_jobs=self.n_cores)(delayed(g_fun_s)(x,t,z,wf,self.sigma,self.mu,self.pi,self.gamma) for x,t,z in zip(X,T,Z))
                elif self.backend == 'cython':
                    # def inf_fun(mdl,x,t,y,w,sigma,mu,pi,gamma):
                    #     return mdl.cy_log_likelihood(x,t,y,w,sigma,mu,pi,gamma,g=True)
                    gs = Parallel(n_jobs=self.n_cores)(delayed(dummy_inf_fun)(self,x,t,z,wf,self.sigma,self.mu,self.pi,self.gamma,True) for x,t,z in zip(X,T,Z))

                for gi in gs: 
                    g += gi
                    
            # adjust for logspace sigma
            if self.learn_sigma:
                g[-self.K:] *= self.sigma
                
            g = -g
                
            # add regularizer gradients
            offset = wf.shape[0]
            g[:offset] = g[:offset] + self.base_classifier_regularizer(wf,g=True)
            if self.learn_gamma:
                g[offset:offset+self.K] = g[offset:offset+self.K]
                offset += self.K
            if self.learn_pi:
                g[offset:offset+2] = g[offset:offset+2] - self.beta_grad(self.pi)
                offset += 2
            if self.learn_mu:
                g[offset:offset+self.K] = g[offset:offset+self.K] - self.n_grad(self.mu)
                offset += self.K
            if self.learn_sigma:
                g[offset:offset+self.K] = g[offset:offset+self.K] - self.sigma*self.ig_grad(self.sigma)
            return g
            
        return g_fun
        
    def set_grads(self):
        ''' Get local gradient functions '''
        self.ig_grad = get_ig_grad(self.alpha,self.beta)
        self.n_grad = get_n_grad(0.0,self.eta)
        self.base_reg_grad = get_n_grad(0.0,self.lambda_0)
        self.beta_grad = get_beta_grad(self.psi_0,self.psi_1,0.0)
        
    def initialize_base_classifier(self,X,T,Z):
        ''' Initialize the base classifier 
            Should be implemented in the base classifier
        '''
        pass
        
    def preprocess_data(self,X,T,Z,is_kernel_data=False):
        return X
        
    def get_bounds(self,X,T,Z):
        return None
        
    def fit(self,X,T,Z,val_data=None):
        ''' Estimate the model parameters by maximizing the log marginal likelihood p(Z|X,T)
        
            Arguments:
                X (list): List of feature sequences (structure will vary with prior model)
                T (list): List of (L_i,) numpy arrays containing instance timestamps
                Z (list): List of observation sequence (structure will vary with prior model)
        '''
        
        # Initialize the base classifier
        self.initialize_base_classifier(X,T,Z)
        
        # Initialize self
        self.initialize()
        
        # Preprocess the data
        # e.g. perform prefiltering
        X_aug = self.preprocess_data(X,T,Z,True)
        
        # Set gradient functions for regularizers
        self.set_grads()
        
        # Get the objective and gradient functions
        obj = self.get_obj(X_aug,T,Z)
        g_fun = self.get_grad(X_aug,T,Z)

        # get any parameter bounds
        bounds = self.get_bounds(X_aug,T,Z)
        
        # Initialize parameter history
        self.param_history = []
        
        # Initialize trakcing
        if self.track_score:
            self.train_scores = []
            self.val_scores = []
            if val_data is not None:
                X_val = self.preprocess_data(val_data[0],False)
                Y_val = val_data[1]
        
        # define tracking callback
        def callback(w):
            self.param_history.append(w)
            # if self.track_score:
            #     if val_data is not None:
            #         self.train_scores.append(self.score(X_aug,Y))
            #         self.val_scores.append(self.score(X_val,Y_val))
            #
            # sys.stdout.flush()
        
        # get start tiem
        start_time = time.time()
        
        # Run learning n_restarts times and store each result
        results = []
        n_restarts = self.n_restarts
        while n_restarts >= 0:
            # Perform warm start
            if self.warm_start is not None:
                if self.warm_start.shape != (self.n_params,): raise ValueError("Warm start wrong size: %s, %s"%(self.warm_start.shape, (self.n_params,)))
                w0 = self.warm_start
            else:
                w0 = self.init_params(X,T,Z)
            
            # If testing gradient
            # TODO: get rid of this?
            # if self.test_grad and self.backend == 'autograd':
            #     quick_grad_check(obj,w0)
            
            # Run learning
            res = minimize(obj,w0,jac=g_fun,method='L-BFGS-B',tol=self.tol,callback=callback,options={"disp":self.verbose,"maxiter":self.max_iter},bounds=bounds)
            results.append(res)
            n_restarts -= 1
            
        # Store total training time    
        self.train_time = time.time() - start_time
        if self.verbose > 0: print "Train time:", self.train_time

        # find the best result over all restarts
        best_res = None
        best_obj = np.inf
        for res in results:
            if res.fun < best_obj:
                best_obj = res.fun
                best_res = res
        
        # Set parameters
        if res.success:
            self.w = self.seperate_params(best_res.x)
        else:
            self.w = self.seperate_params(np.zeros(self.n_params))
            
        # Clear gradient functions
        # TODO: move this to pickle helper functions
        self.ig_grad = None
        self.n_grad = None
        self.beta_grad = None
        self.base_reg_grad = None
        
    def get_params(self, deep=True):
        ''' Get params (necessary for sklearn interface) '''
        return {"lambda_0":self.lambda_0,"max_iter":self.max_iter}

    def set_params(self, **parameters):
        ''' Set params (necessary for sklearn interface) '''
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        self.initialize()
            
        return self
        
    def hmc(self,X,T,Z,n_samples=100,n_steps=10,epsilon=0.2,seed=None):
        ''' 
            Draws n_samples samples from the posterior distribution using hamiltonian monte
            carlo (implemented in pyhmc).
        
            Arguments:
                X (list): List of feature sequences (structure will vary with prior model)
                T (list): List of (L_i,) numpy arrays containing instance timestamps
                Z (list): List of observation sequence (structure will vary with prior model)
                n_samples: number of samples returned
                n_steps: number of hamiltonian steps taken between samples
                epsilon: step size
                seed: random seed for initialization
        
            Returns:
                self
                
        '''
        # get seed
        if seed is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(seed)
        
        # Initialize the base classifier
        self.initialize_base_classifier(X,Y)
        
        # Initialize self
        self.initialize()
        
        # Preprocess the data
        # e.g. perform prefiltering
        X_aug = self.preprocess_data(X,Y,True)
        
        # Set gradient functions for regularizers
        self.set_grads()
        
        # Get the objective and gradient functions
        obj = self.get_obj(X_aug,Y)
        g_fun = self.get_grad(X_aug,Y)
        
        # get any parameter bounds
        # keyboard()
        bounds = self.get_bounds(X_aug,Y)
        
        # get start tiem
        start_time = time.time()
        
        def logp_and_grad(w):
            return -obj(w),-g_fun(w)
        
        # Samples
        if self.warm_start:
            w0 = self.init_params(X,Y)
            res = minimize(obj,w0,jac=g_fun,method='L-BFGS-B',tol=self.tol,options={"disp":self.verbose,"maxiter":25},bounds=bounds)
            w0 = res.x
        else:
            w0 = np.random.randn(self.n_params)
            
        self.param_samples,self.sampling_logps,self.sampling_diagnostics = hmc(logp_and_grad,x0=w0,n_samples=n_samples,n_steps=n_steps,epsilon=epsilon,display=True,return_logp=True,return_diagnostics=True)
            
        # Store total training time    
        self.train_time = time.time() - start_time
        if self.verbose > 0: print "Train time:", self.train_time
            
        # Clear gradient functions
        # TODO: move this to pickle helper functions
        self.ig_grad = None
        self.n_grad = None
        self.beta_grad = None
        self.base_reg_grad = None
        
        return self
        
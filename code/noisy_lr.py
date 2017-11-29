import autograd.numpy as np
import theano.tensor as T
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gamma
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
from noisy_timeseries_detection import *
from sklearn.metrics.pairwise import rbf_kernel
from theano import printing
from util import *
    
# def inverse_gamma_log_pdf(x,alpha,beta):
#     return alpha*np.log(beta) - (alpha - 1.0)*np.log(x) - beta/x - np.log(gamma(alpha))

def symbolic_ll_iter(log_prior_i,t_i,alpha,y,sigma,mu,pi,log_mixture_dist,K,pi_offset,max_observation_distance,inference_type='marginal'):
    ''' A single simbolic iteration of the inference DP in Adams and Marlin (2017)'''
    m = y.shape[0]
    missed_neg_case = alpha + log_prior_i[0] + tensor.log(1.0 - symbolic_sigmoid(pi[0],pi_offset))
    missed_pos_case = alpha + log_prior_i[1] + tensor.log(1.0 - symbolic_sigmoid(pi[1],pi_offset))
    
    mu_hat = t_i+mu.reshape((1,K))
    
    # calculate max and min values for t_i
    min_t = mu_hat-max_observation_distance
    max_t = mu_hat+max_observation_distance
    component_contributions = symbolic_normal_log_pdf(y.reshape((m,1)),mu_hat,sigma.reshape((1,K)),max_observation_distance)
    component_contributions = tensor.switch(tensor.lt(y.reshape((m,1)),min_t),-1e100*tensor.ones_like(component_contributions),component_contributions)
    component_contributions = tensor.switch(tensor.gt(y.reshape((m,1)),max_t),-1e100*tensor.ones_like(component_contributions),component_contributions)
    log_p_y = symbolic_logsumexp(component_contributions + log_mixture_dist.reshape((1,K)),axis=1,keepdims=False)    
    observed_pos_case = alpha[:-1] + log_p_y + log_prior_i[1] + tensor.log(symbolic_sigmoid(pi[1],pi_offset))
    observed_neg_case = alpha[:-1] + log_p_y + log_prior_i[0] + tensor.log(symbolic_sigmoid(pi[0],pi_offset))
    
    tmp1 = tensor.stack(missed_neg_case[1:],missed_pos_case[1:],observed_pos_case,observed_neg_case)
    tmp2 = symbolic_logsumexp(tmp1,axis=0,keepdims=False)
    tmp3 = tensor.shape_padright(symbolic_logsumexp(tensor.stack([missed_neg_case[0],missed_pos_case[0]]),keepdims=False))
    
    return tensor.concatenate([tmp3,tmp2])

class NoiseyLogisticRegression(NoiseyTimeseriesDetection):
    def __init__(   self,scorer='accuracy',prior_model="logistic_regression",hidden_layer_sizes=(5,),
                    n_features=10,kernel='linear',kernel_bandwidth=1.0,warm_start_w_fixed_pi=False,**kwargs):
        '''
            Initialize Noisy LR instance. This class defines inference and learning for the model described
            in Adams and Marlin (2017). In particular, the complete model is given by:
                    
                                     p(mu) = N(mu; 0, eta)
                             p(exp(sigma)) = Inv-Gamma(exp(sigma); alpha, beta)
                                  p(theta) = N(theta; 0, 1/lambda_0 * I)
                                  p(1-pi[0]) = Beta(psi_0[0],psi_0[1]) 
                                  p(pi[1]) = Beta(psi_1[0],psi_1[1]) 
                              p(y|x,theta) = BaseClassifier(y; x, theta)
                         p(o_i=1|y_i=v,pi) = sigmoid(pi[v])
                   p(z_m | t_i, mu, sigma) = N(z_m; t + mu, exp(sigma))                
            p(y,o,z|x,t,theta,pi,mu,sigma) = p(y|x,theta) p(o|y,pi) p(z|o,t,mu,sigma)
                    
            Parameters are estimated using MAP estimation and optimization is performed using L-BFGS as
            implemented in scipy.optimize. For more details see Adams and Marlin (2017).
                    
            Multiple methods take as input X, T, Z, Y, and w. These arguments are described in detail below. x, t, z, and y, represent components of 
            each of these lists. Notation is consistent with Adams and Marlin (2017):
                    
            X - Length N list of shape (L_n,n_features) arrays where L_n is the length of sequence n. These are instance level features.
            T - Length N list of shape (L_n,) arrays containing the timestamps for each input sequence.
            Z - Length N list of shape (M_n,) arrays containing the observation timestamps for each session.
            Y - Length N list of shape (L_n,) arrays containing the binary sequence labels for each session.
            w - The complete vector of parameters including base classifier and noise model parameters.
                    
            Arguments:
                prior_model (string): Specifies the base classifier type. May be either 'logistic_regression' or 'mlp'.
                n_features (int): Number of instance level features. Default=10.
                warm_start_w_fixed_pi (bool): If True, first learns a model with pi fixed at [-100.0,100.0] and then continues training the model without pi fixed. Default=False.
                max_iter (int): Max number of iterations to run learning.
                lambda_0 (float): The l2 regularization strength on the base classifier parameters. Default=1.0.
                eta (float): The variance of the prior on mu. Default=1.0.
                alpha (float): The shape parameter of the Inverse-Gamma prior on exp(sigma). Default=1.0.
                beta (float): The scale parameter of the Inverse-Gamma prior on exp(sigma). Default=1.0.
                psi_0 (np.array): Length 2 array containing the parameters of the beta distribution on 1-pi[0]. Default=np.array([5.0,1.0])
                psi_1 (np.array): Length 2 array containing the parameters of the beta distribution on pi[1]. Default=np.array([5.0,1.0])
                verbose (int): Verbosity level. Default=0.
                tol (float): L-BFGS tolerance. See scipy.optimize for details. Default=1e-4
                scorer (string): The scorer used for hyperparameter tuning. Takes values in {'accuracy','f1','marginal_likelihood','balanced_accuracy'}. Default='accuracy'.
                hidden_layer_sizes (tuple): Only used when prior_model='mlp'. Speficies the number of hidden units per layer in the MLP base classifier. Default=(5,).
                kernel (string): Only used when prior_model='logistic_regression'. Specifies a kernel for kernel logistic regression. Takes values in {'linear','rbf'}. Default='linear'.
                kernel_bandwidth (float): Only used when kernel='rbf'. Speficies the bandwidth of the kernel. Default=1.0.
                learn_sigma (bool): Specifies whether sigma should be learned or fixed. Default=True.
                learn_mu (bool): Specifies whether mu should be learned or fixed. Default=True. 
                learn_pi (bool): Specifies whether pi should be learned or fixed. Default=True. 
                learn_gamma (bool): Not implemented.
                mu (float): Value of mu if learn_mu=False. Default=0.0.
                sigma (float): Value of sigma if learn_sigma=False. Default=1.0.
                pi (np.array): Length 2 float array. Value of pi if learn_pi=False. Default=np.array([-100.0,100.0]).
                gamma (np.array): Not implemented.
                track_score (bool): Not implemented.
                warm_start (np.ndarray): Length n_params array containing initial parameter values.
                backend (string): Which AD backend to use. May take values in {'autograd','theano'}. Default='theano'.
                K (int): Not implemented.
                n_restarts (int): Number times to run learning. Parameters are set to the result that maximizes the marginal likelihood.
                pi_offset (float): Not implemented.
                max_observation_distance (float): Specifies the maximum allowed distance between an observation and the instance that generated it. Default=1000.0.
                n_cores (int): Number of cores to run inference on. Inference for individual sessions in the training set are split across n_cores cores using joblib. Default=1.
        '''
        if hidden_layer_sizes == (1,):
            prior_model = "logistic_regression"
            kernel = 'linear'
        self.__dict__.update(locals())
        NoiseyTimeseriesDetection.__init__(self,**kwargs)
        
    def get_n_base_classifier_params(self):
        ''' Returns the number of base classifier parameters '''
        return self.n_params - self.K*(self.learn_sigma + self.learn_mu + self.learn_gamma) - 2*self.learn_pi
        
    def get_n_prior_params(self):
        ''' Returns the number of base classifier parameters '''
        if self.prior_model == "logistic_regression":
            if self.kernel == "rbf":
                if hasattr(self,'x_size'):
                    assert self.x_size > 0
                    n_params = self.x_size
                else:
                    n_params = 0
            elif self.kernel == 'linear':
                n_params = self.n_features
        elif self.prior_model == "mlp":
            n_params = self.n_features*self.hidden_layer_sizes[0] + self.hidden_layer_sizes[0]
            for i in range(len(self.hidden_layer_sizes)-1):
                n_params += self.hidden_layer_sizes[i]*self.hidden_layer_sizes[i+1] + self.hidden_layer_sizes[i+1]
            n_params += self.hidden_layer_sizes[-1] + 1
        else:
            print "Invalid prior model: ", self.prior_model
            
        return n_params
        
    # def prior_log_partition(self,X,w):
    #     energy = np.dot(X,w)
    #     return np.sum(np.log1p(np.exp(energy)))
        
    def log_joint(self,x,t,y,o,z,w):
        ''' 
            Returns p(Z,O,Y|X,T,w). Mainly for testing.
        '''
        assert np.sum(o) == z.shape[0]
        n = x.shape[0]
        m = z.shape[0]
        
        # p(y|x)
        p = np.sum(self.log_prior(x,w)[np.arange(n),y.astype(int)])
        
        # p(o|y)
        for ov in range(2):
            for yv in range(2):
                if np.sum(o[y==yv]==ov) > 0:
                    log_p = np.log(sigmoid(self.pi[yv],self.pi_offset)) if ov == 1 else np.log(1.0-sigmoid(self.pi[yv],self.pi_offset))
                    p += log_p*np.sum(o[y==yv]==ov)
        
        # p += np.log(sigmoid(self.pi))*np.sum(O[Z==1]==1)
        # if np.sum(O[Z==1]==0) > 0:
        #     p += np.log(1.0-sigmoid(self.pi))*np.sum(O[Z==1]==0)
        log_mixture_dist = self.gamma - logsumexp(self.gamma)
        mu_hat = t[o==1].reshape((m,1))+self.mu.reshape((1,self.K))
        min_z = mu_hat - self.max_observation_distance
        max_z = mu_hat + self.max_observation_distance
        component_contributions = normal_log_pdf(z.reshape((m,1)),mu_hat,self.sigma.reshape((1,self.K)),self.max_observation_distance)
        component_contributions[z.reshape((m,1)) < min_z] = -1e100
        component_contributions[z.reshape((m,1)) > max_z] = -1e100
        log_p_z = logsumexp(component_contributions + log_mixture_dist.reshape((1,self.K)), axis=1)
        p += np.sum(log_p_z)

        return p
        
    def log_prior(self,x,w):
        ''' 
            Returns log(p(Y|x,w))
        '''
        n = x.shape[0]
        if self.prior_model == "logistic_regression":
            negative_energy = np.dot(x,w)
            return np.vstack((-np.log1p(np.exp(negative_energy))*np.ones(x.shape[0]),negative_energy - np.log1p(np.exp(negative_energy)))).T
        elif self.prior_model == "mlp":
            cur_idx = self.n_features*self.hidden_layer_sizes[0]
            wi = w[:cur_idx].reshape((self.n_features,self.hidden_layer_sizes[0]))
            bi = w[cur_idx:cur_idx+self.hidden_layer_sizes[0]]
            ho = np.dot(x,wi) + bi
            hi = np.tanh(ho)
            cur_idx += self.hidden_layer_sizes[0]
            for i in range(len(self.hidden_layer_sizes) - 1):
                wi = w[cur_idx:cur_idx+self.hidden_layer_sizes[i]*self.hidden_layer_sizes[i+1]].reshape((self.hidden_layer_sizes[i],self.hidden_layer_sizes[i+1]))
                cur_idx += self.hidden_layer_sizes[i]*self.hidden_layer_sizes[i+1]
                bi = w[cur_idx:cur_idx+self.hidden_layer_sizes[i+1]]
                cur_idx += self.hidden_layer_sizes[i+1]
                ho = np.dot(hi,wi) + bi
                hi = np.tanh(ho)
                # cur_idx = cur_idx+self.n_hidden_units**2
            negative_energy = np.dot(hi,w[cur_idx:-1]) + w[-1]
            negative_energy = np.vstack((np.zeros(n),negative_energy)).T
            return negative_energy - logsumexp(negative_energy,axis=1).reshape((n,1))
        else:
            raise ValueError("Invalid prior model: %s"%self.prior_model)
        
    def log_likelihood(self,x,t,z,w,inference_type='marginal'):
        '''
            Performs MAP or marginal inference. If inference_type == 'marginal', returns p(Z|X,T,w).
            If inference_type == 'map', returns (o_star,y_star,v_star) where
        
            o_star,y_star = argmax_{o,y} p(Z,o,y|X,T,w)
                   v_star = max_{o,y} p(Z,o,y|X,T,w)
        '''
        if inference_type == 'marginal':
            inf_fun = logsumexp
        elif inference_type == 'map':
            inf_fun = np.max
        
        n = x.shape[0]
        m = z.shape[0]
        log_prior = self.log_prior(x,w)
        log_mixture_dist = self.gamma - logsumexp(self.gamma)
            
        alpha = np.zeros(1)
        if inference_type == 'map':
            map_o_idxs = -np.ones((n+1,m+1),dtype=int)
            map_y_idxs = -np.ones((n+1,m+1),dtype=int)
        for i in range(1,n+1):
            # TODO: OHNOOOOOOOOS SO MUCH CODE REPLICATION!!!!.....!
            if i <= m:
                missed_neg_case = np.hstack([alpha[1:i],-np.inf])
                missed_pos_case = np.hstack([alpha[1:i],-np.inf])
            else:
                missed_neg_case = alpha[1:m+1]
                missed_pos_case = alpha[1:m+1]
                
            mu_hat = t[i-1]+self.mu.reshape((1,self.K))
            min_z = mu_hat - self.max_observation_distance
            max_z = mu_hat + self.max_observation_distance
            component_contributions = normal_log_pdf(z[:min(m,i)].reshape((min(m,i),1)),mu_hat,self.sigma.reshape((1,self.K)),self.max_observation_distance)
            component_contributions[z[:min(m,i)].reshape((min(m,i),1)) < min_z] = -1e100
            component_contributions[z[:min(m,i)].reshape((min(m,i),1)) > max_z] = -1e100
            log_p_y = logsumexp(component_contributions + log_mixture_dist.reshape((1,self.K)),axis=1)
                
            observed_neg_case = np.log(sigmoid(self.pi[0],self.pi_offset)) + alpha[:min(m,i)] + log_p_y + log_prior[i-1,0]
            missed_neg_case = missed_neg_case + np.log(1.0-sigmoid(self.pi[0],self.pi_offset)) + log_prior[i-1,0] 
            observed_pos_case = np.log(sigmoid(self.pi[1],self.pi_offset)) + alpha[:min(m,i)] + log_p_y + log_prior[i-1,1]
            missed_pos_case = missed_pos_case + np.log(1.0-sigmoid(self.pi[1],self.pi_offset)) + log_prior[i-1,1]
            alpha_0 = inf_fun(np.array([log_prior[i-1,0] + np.log(1.0-sigmoid(self.pi[0],self.pi_offset)) + alpha[0],log_prior[i-1,1] + np.log(1.0-sigmoid(self.pi[1],self.pi_offset)) + alpha[0]]))
            alpha = np.hstack([alpha_0,inf_fun(np.vstack([missed_neg_case,observed_neg_case,missed_pos_case,observed_pos_case]),axis=0)])
            if inference_type == 'map':
                map_idxs = np.argmax(np.vstack([missed_neg_case,observed_neg_case,missed_pos_case,observed_pos_case]),axis=0)
                # print "%d:"%i,
                # print map_idxs
                map_o_idxs[i,1:1+map_idxs.shape[0]] = map_idxs%2
                map_o_idxs[i,0] = 0
                map_y_idxs[i,1:1+map_idxs.shape[0]] = np.floor(map_idxs/2)
                map_y_idxs[i,0] = np.argmax(np.array([log_prior[i-1,0] + np.log(1.0-sigmoid(self.pi[0],self.pi_offset)) + alpha[0],log_prior[i-1,1] + np.log(1.0-sigmoid(self.pi[1],self.pi_offset)) + alpha[0]]))
            
        # print alpha
        # keyboard()
        if inference_type == 'marginal':
            return alpha[-1]
        elif inference_type == 'map':
            map_o = -np.ones(n,dtype=int)
            map_y = -np.ones(n,dtype=int)
            l = m
            for i in range(n)[::-1]:
                map_o[i] = map_o_idxs[i+1,l]
                map_y[i] = map_y_idxs[i+1,l]
                l -= map_o[i]
                
            assert np.all(map_o >= 0) and np.all(map_o <= 1)
            assert np.all(map_y >= 0) and np.all(map_y <= 1)
            return map_o,map_y,alpha[-1]
        
    def symbolic_log_prior(self,x,w):
        '''
            Symbolic version of log_prior
        '''
        n = x.shape[0]
        # print_n_op = printing.Print('n')
        # n = print_n_op(n)
        if self.prior_model == "logistic_regression":
            negative_energy = tensor.dot(x,w)
            negative_energy = tensor.stack([tensor.zeros_like(negative_energy),negative_energy]).T
            return negative_energy - symbolic_logsumexp(negative_energy,axis=1,keepdims=False).reshape((n,1))
        elif self.prior_model == "mlp":
            cur_idx = self.n_features*self.hidden_layer_sizes[0]
            wi = w[:cur_idx].reshape((self.n_features,self.hidden_layer_sizes[0]))
            bi = w[cur_idx:cur_idx+self.hidden_layer_sizes[0]].reshape((1,self.hidden_layer_sizes[0]))
            ho = tensor.dot(x,wi) + bi
            hi = tensor.tanh(ho)
            # print_h0_op = printing.Print('h0')
            # hi = print_h0_op(hi)
            cur_idx += self.hidden_layer_sizes[0]
            for i in range(len(self.hidden_layer_sizes) - 1):
                wi = w[cur_idx:cur_idx+self.hidden_layer_sizes[i]*self.hidden_layer_sizes[i+1]].reshape((self.hidden_layer_sizes[i],self.hidden_layer_sizes[i+1]))
                cur_idx += self.hidden_layer_sizes[i]*self.hidden_layer_sizes[i+1]
                bi = w[cur_idx:cur_idx+self.hidden_layer_sizes[i+1]].reshape((1,self.hidden_layer_sizes[i+1]))
                cur_idx += self.hidden_layer_sizes[i+1]
                ho = tensor.dot(hi,wi) + bi
                hi = tensor.tanh(ho)
            negative_energy = tensor.dot(hi,w[cur_idx:-1]) + w[-1]
            negative_energy = tensor.stack([tensor.zeros_like(negative_energy),negative_energy]).T

            # print_energy_op = printing.Print('energy')
            # negative_energy = print_energy_op(negative_energy)
            return negative_energy - symbolic_logsumexp(negative_energy,axis=1,keepdims=False).reshape((n,1))
        else:
            raise ValueError("Invalid prior model: %s"%self.prior_model)
        
    def symbolic_log_likelihood(self,x,t,z,w,sigma,mu,pi,gamma):
        ''' 
            Symbolic version of log_likelihood.
        '''
        n = x.shape[0]
        m = z.shape[0]
        log_prior = self.symbolic_log_prior(x,w)
        # print log_prior.ndim
            
        alpha0 = tensor.concatenate([tensor.shape_padright(0.0),-1e100*tensor.ones((m,))])
        log_mixture_dist = gamma - symbolic_logsumexp(gamma)
        # print alpha0.ndim
        results,_ = theano.scan(fn=symbolic_ll_iter,sequences=[log_prior,t],outputs_info=[alpha0],non_sequences=[z,sigma,mu,pi,log_mixture_dist,self.K,self.pi_offset,self.max_observation_distance])
            
        return results[-1][-1]
        
    def predict(self,X):
        '''
            Returns [argmax_y p(y|x) for x in X]
        '''
        X_k = self.preprocess_data(X,None,False)
        P = [np.exp(self.log_prior(x,self.w)[:,1]) for x in X_k]
        Y_hat = [1*(p > .5) for p in P]
        return Y_hat
        
    def predict_proba(self,X):
        '''
            Returns [p(y|x) for x in X]
        '''
        
        X_k = self.preprocess_data(X,None,False)
        
        P = []
        for x in X_k:
            probs = np.exp(self.log_prior(x,self.w))
            P.append(probs)
        return P
        
    def score(self,X,Y):
        '''
            Makes a prediction for X and returns the score evaluated on Y.
        '''
        Y_hat = np.hstack(self.predict(X))
        if self.scorer == 'accuracy':
            score = np.mean(np.hstack(Y) == Y_hat)
        elif self.scorer == 'f1':
            score = f1_score(np.hstack(Y),Y_hat)
        # elif self.scorer == 'marginal_likelihood':
        #     score = np.sum([self.log_likelihood(x,y[0],self.w) for x,y in zip(X,Y)])
        elif self.scorer == "balanced_accuracy":
            gt = np.hstack(Y)
            pos_acc = np.mean(Y_hat[gt==1]==1)
            neg_acc = np.mean(Y_hat[gt==0]==0)
            return 0.5*(pos_acc+neg_acc)
        return score
        
    def initialize_base_classifier(self,X,T,Z):
        self.x_size = np.sum([x.shape[0] for x in X])
        
    def init_params(self,X=None,T=None,Z=None):
        '''
            Initializes parameters
        '''
        if self.warm_start_w_fixed_pi:
            warm_start_model = NoiseyLogisticRegression(lambda_0=self.lambda_0,sigma=self.sigma,pi=np.array([-np.inf,np.inf]),
                                                        n_features=self.n_features,max_iter=self.max_iter,
                                                        verbose=self.verbose,tol=self.tol,test_grad=False,track_score=False,
                                                        scorer=self.scorer,alpha=self.alpha,beta=self.alpha,
                                                        learn_sigma=self.learn_sigma,prior_model=self.prior_model,
                                                        backend=self.backend,learn_mu=self.learn_mu,learn_pi=False,
                                                        learn_gamma=self.learn_gamma,K=self.K,n_restarts=self.n_restarts,
                                                        kernel=self.kernel,pi_offset=self.pi_offset,warm_start_w_fixed_pi=False,
                                                        hidden_layer_sizes=self.hidden_layer_sizes)
                                 

            warm_start_model.fit(X,T,Z)
            w = self.build_params(warm_start_model.w,warm_start_model.sigma,warm_start_model.mu,np.array([-1.0,1.0]),warm_start_model.gamma)
            return w
        
        if self.prior_model == "logistic_regression":
            n = self.get_n_base_classifier_params()
            wf = np.random.rand(n)*0.25
            sigma = np.ones(self.K) + np.random.rand(self.K)*0.1
            mu = np.random.randn(self.K)*0.01
            pi = np.array([-1.0,1.0])
            gamma = np.zeros(self.K)
            w = self.build_params(wf,sigma,mu,pi,gamma)
            return w
        elif self.prior_model == "mlp":
            n = self.get_n_base_classifier_params()
            wf = np.random.rand(n)*0.25
            sigma = np.ones(self.K) + np.random.rand(self.K)*0.1
            mu = np.random.randn(self.K)*0.01
            pi = np.array([-1.0,1.0])
            gamma = np.zeros(self.K) + np.random.rand(self.K)*0.1
            w = self.build_params(wf,sigma,mu,pi,gamma)
            return w
            
    def preprocess_data(self,X,T,Z,is_kernel_data=False):
        '''
            Preprocess data if using kernel logistic regression.
        '''
        if self.prior_model == 'logistic_regression' and self.kernel == 'rbf':
            if is_kernel_data:
                self.kernel_data = np.vstack(X)
                X_out = []
                for x in X:
                    x_k = rbf_kernel(x,self.kernel_data,self.kernel_bandwidth)
                    X_out.append(x_k)
                self.kernel_matrix = np.vstack(X_out)
                return X_out
            else:
                X_out = []
                for x in X:
                    x_k = rbf_kernel(x,self.kernel_data,self.kernel_bandwidth)
                    X_out.append(x_k)
                return X_out
        else:
            return X
            
    def base_classifier_regularizer(self,wf,g=False):
        '''
            If g==False, return base classifier regularizer value, else
            return the gradient of the base classifier regularizer.
        '''
        if self.kernel == 'rbf':
            if g:
                return 2.0*self.lambda_0*np.dot(wf,self.kernel_matrix)
            else:
                return self.lambda_0*np.dot(np.dot(wf,self.kernel_matrix),wf)
            
        if g:
            return 2.0*self.lambda_0*wf
        else:
            return self.lambda_0*np.dot(wf,wf)
            
    def initialize_prior_model(self):
        '''
            Init the prior model. (Not used)
        '''
        if self.hidden_layer_sizes == (1,):
            self.prior_model = "logistic_regression"
            self.kernel = 'linear' 
            
    def map_inference(self,X,T,Z):
        '''
            Wrapper for map inference. Returns [o_star,y_star,v_star for x in X]
        '''
        return [self._map_inference(x,t,z) for x,t,z in zip(X,T,Z)]
        
    def _map_inference(self,x,t,z):
        map_o,map_y,score = self.log_likelihood(x,t,z,self.w,inference_type='map')
        return map_o,map_y,score
    
# def test_fit(prior_model,seed):
#     np.random.seed(seed)
#     S = 100
#     max_n = 20
#     nf = 10
#
#     X = []
#     Y = []
#     w = np.random.randn(nf)
#     while len(X) < S-1:
#         n = np.random.randint(1,max_n+1)
#         x_f = np.random.randn(n,nf)
#         t = np.arange(1,n+1)
#         pos_idxs = np.where(np.dot(x_f,w) > 0)
#         z = np.zeros(n,dtype=int)
#         z[pos_idxs] = 1
#         if np.sum(z) == 0:
#             print "here"
#             continue
#         # y = np.random.choice(t,n_pos,replace=False)
#         y = t[pos_idxs]
#         X.append([x_f,t])
#         Y.append([y,z])
#
#     x_size = np.sum([x[0].shape[0] for x in X])
#     kernel = 'linear'
#     nlr = NoiseyLogisticRegression(lambda_0=1.0,sigma=0.001,pi=1.0,verbose=1,test_grad=True,learn_sigma=True,alpha=2.0,beta=3.0,prior_model=prior_model,n_features=nf,backend='theano',learn_mu=True,learn_pi=True,learn_gamma=True,K=1,kernel=kernel,hidden_layer_sizes=(1,))
#     nlr.fit(X,Y)
#     print nlr.score(X,Y)
#
# def test_time(prior_model):
#     S = 25
#     n = 300
#     nf = 125
#
#     X = []
#     Y = []
#     w = np.random.randn(nf)
#     for s in range(S):
#         x_f = np.random.randn(n,nf)
#         t = np.arange(1,n+1)
#         pos_idxs = np.where(np.dot(x_f,w) > 10.0)
#         z = np.zeros(n,dtype=int)
#         z[pos_idxs] = 1
#         # y = np.random.choice(t,n_pos,replace=False)
#         y = t[pos_idxs]
#         X.append([x_f,t])
#         Y.append([y,z])
#     # print np.mean(np.hstack(zip(*Y)[1]))
#
#     # nlr = NoiseyLogisticRegression(lambda_0=0.0,sigma=0.001,verbose=1,test_grad=False,learn_sigma=True,alpha=2.0,beta=3.0,prior_model=prior_model,n_features=nf,n_hidden_units=5,n_hidden_layers=2,backend='theano')
#     nlr = NoiseyLogisticRegression(prior_model=prior_model,backend='theano')
#
#     obj = nlr.get_obj(X,Y)
#     g_fun = nlr.get_grad(X,Y)
#
#     n_iter = 5
#     for f,nm in [(obj,'obj'),(g_fun,'grad')]:
#         w = np.random.randn(nlr.n_params)
#         start = time.time()
#         for i in range(n_iter):
#             f(w)
#         print "%s time: "%nm,(time.time()-start)/float(n_iter)
    
    

        
    
# if __name__=="__main__":
#     # seed = int(sys.argv[1])
#     # print "****** Testing LR ******"
#     # test_log_partition(seed,"logistic_regression")
#     # test_fit("logistic_regression",seed)
#     # print "****** Testing MLP ******"
#     # test_log_partition(int(sys.argv[1]),"mlp")
#     # test_fit("mlp",seed)
#     # print "****** Time tests ******"
#     # test_time("logistic_regression")
#     # test_time("mlp")
#     unittest.main()
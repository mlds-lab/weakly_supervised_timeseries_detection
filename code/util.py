import autograd.numpy as np
from autograd.scipy.special import gamma,gammaln,erf
from autograd import grad
import theano
import theano.tensor as tensor

def symbolic_logsumexp(x, axis=None, keepdims=True):
    ''' Numerically stable theano version of the Log-Sum-Exp trick'''
    x_max = tensor.max(x, axis=axis, keepdims=True)
    preres = tensor.log(tensor.sum(tensor.exp(x - x_max), axis=axis, keepdims=keepdims))
    return preres + x_max.reshape(preres.shape)
    
def standard_normal_cdf(x):
    ''' Standard normal CDF '''
    return (1.0 + erf(x/np.sqrt(2.0)))/2.0

def normal_log_pdf(x,mu,sigma2,max_d=1e100):
    ''' Truncated normal log pdf with mean mu, variance sigma2, and max distance from the mean max_d '''
    return -0.5*((x-mu)**2/sigma2 + np.log(sigma2*2.0*np.pi)) - np.log(erf(max_d/(np.sqrt(sigma2*2.0))))
    
def symbolic_standard_normal_cdf(x):
    ''' Standard normal cdf (theano) '''
    return (1.0 + tensor.erf(x/tensor.sqrt(2.0)))/2.0
    
def symbolic_normal_log_pdf(x,mu,sigma,max_d=1e100):
    ''' Truncated normal log pdf with mean mu, variance sigma2, and max distance from the mean max_d (theano) '''
    log_p = -0.5*((x-mu)**2/sigma + tensor.log(sigma*2.0*np.pi)) - tensor.log(tensor.erf(max_d/(tensor.sqrt(sigma*2.0))))
    return log_p
    
def inverse_gamma_log_pdf(x,alpha,beta):
    ''' Inverse gamma log pdf '''
    return alpha*np.log(beta) - (alpha + 1.0)*np.log(x) - beta/x - np.log(gamma(alpha))

def get_ig_grad(alpha,beta):
    ''' Returns function that calculates the gradient of the inverse gamma log pdf '''
    def fun(s):
        return inverse_gamma_log_pdf(s,alpha,beta).sum()
    return grad(fun)
    
def sigmoid(pi,offset=0.0):
    ''' Sigmoid with offset '''
    return offset + (1-offset)*1.0/(1.0 + np.exp(-pi))
    
def symbolic_sigmoid(pi,offset):
    ''' Symbolic sigmoid with offset '''
    return offset + (1-offset)*tensor.nnet.sigmoid(pi)
    
def get_n_grad(mu,eta):
    ''' Returns a function that calculates the gradient of the normal log pdf wrt the mean and variance '''
    def fun(s):
        res = normal_log_pdf(s,mu,eta).sum()
        return res
    return grad(fun)
    
def beta_log_pdf(x,a,b):
    ''' Beta log pdf '''
    return (a-1.0)*np.log(x) + (b-1.0)*np.log(1.0-x) - gammaln(a) - gammaln(b) + gammaln(a+b)
    
def get_beta_grad(psi_0,psi_1,offset):
    ''' Returns a function that calculates the gradient of the beta log pdf '''
    def fun(pi):
        out = 0.0
        out += beta_log_pdf(1.0-sigmoid(pi[0],offset),psi_0[0],psi_0[1])
        out += beta_log_pdf(sigmoid(pi[1],offset),psi_1[0],psi_1[1])
        return out
    
    return grad(fun)
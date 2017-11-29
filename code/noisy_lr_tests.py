import numpy as np
from noisy_lr import NoiseyLogisticRegression
import unittest
import itertools as it
from scipy.misc import logsumexp

class NLRTester(unittest.TestCase):
    
    def test_build_params(self):
        K = 1
        ntd = NoiseyLogisticRegression(learn_sigma=True,learn_mu=True,learn_pi=True,learn_gamma=True,K=K)
        n = ntd.get_n_base_classifier_params()
        wf = np.arange(n)
        sigma = np.random.rand(K)
        mu = np.random.randn(K)
        pi = np.random.randn(2)
        gamma = np.random.randn(K)
        
        w1 = ntd.build_params(wf,sigma,mu,pi,gamma)
        
        w2 = np.hstack((wf,gamma,pi,mu,np.log(sigma)))
        
        print w1.shape,w2.shape
        print w1
        print w2
        self.assertTrue(np.all(w1==w2))
        
    def _test_KLR(self):
        K=1
        mdl = NoiseyLogisticRegression(learn_sigma=True,learn_mu=True,learn_pi=True,learn_gamma=True,K=K,verbose=True,kernel="rbf",backend="theano")
        raw_data = np.load("../data/toy_data.npy")[0]
        n_sequences = len(raw_data["features"])
        n_features = raw_data["features"][0].shape[1]

        X0 = raw_data["features"]
        Z = raw_data["labels"]
        Y = raw_data["observations"]
        T = raw_data["timestamps"]
        
        X_processed = mdl.preprocess_data(zip(X0,T),True)
        
        mdl.fit(zip(X0,T),zip(Y,Z))
        
    def _test_hmc(self):
        K=1
        mdl = NoiseyLogisticRegression(learn_sigma=True,learn_mu=True,learn_pi=True,learn_gamma=True,K=K,verbose=True,kernel="linear",backend="theano")
        raw_data = np.load("../data/toy_data.npy")[0]
        n_sequences = len(raw_data["features"])
        n_features = raw_data["features"][0].shape[1]

        X0 = raw_data["features"]
        Z = raw_data["labels"]
        Y = raw_data["observations"]
        T = raw_data["timestamps"]
                
        mdl.hmc(zip(X0,T),zip(Y,Z),n_samples=50,n_steps=10,epsilon=0.05)
        
    def test_log_partition(self):
        print
        for pm in ["logistic_regression","mlp"]:
            for seed in range(10):
                self._test_log_partition(seed,pm)
        
    def _test_log_partition(self,seed,prior_model):
        np.random.seed(seed)
        n = 5
        nf = 3
        K = 1
        mod=10000
        x = np.random.randn(n,nf)
        t = np.arange(n)
        z = np.array([1.0,3.0])
        y = np.array([0,1,0,1,0])
        nlr = NoiseyLogisticRegression( lambda_0=0.0,sigma=1.0,prior_model=prior_model,
                                        n_features=nf,learn_sigma=True,learn_mu=True,
                                        pi=np.array([0.0,0.0]),learn_pi=True,K=K,learn_gamma=True,
                                        pi_offset=0.5,max_observation_distance=mod)
                                        
        nfp = nlr.get_n_base_classifier_params()
        w = np.random.randn(nfp)
    
        bf_log_partition = -np.inf
        bf_map_score = -np.inf
        bf_map_o = None
        bf_map_y = None
        nlr.sigma = np.exp(np.random.randn(K)*np.ones((K,)))
        nlr.mu = np.random.randn(K)*np.ones((K,))
        nlr.pi = np.random.randn(2)
        nlr.gamma = np.random.randn(K)
        nlr.w = w
        # print nlr.mu, nlr.sigma, nlr.pi
        for y_t in it.product(range(2),repeat=n):

            for o_t in it.product(range(2),repeat=n):
                o = np.array(o_t)
                y = np.array(y_t)
                if np.sum(o) != z.shape[0]:
                    continue
                score = nlr.log_joint(x,t,y,o,z,w)
                bf_log_partition = logsumexp([bf_log_partition, score])
                if bf_map_score < score:
                    bf_map_score = score
                    bf_map_o = o
                    bf_map_y = y
                # bf_map_score = max(bf_map_score,nlr.log_joint(X,z,o,y,w))
                
    
        # Test log partition
        a = nlr.log_likelihood(x,t,z,w)
        ll_fun = nlr.get_theano_log_likelihood()
        b = ll_fun(x,t,z,w,nlr.sigma,nlr.mu,nlr.pi,nlr.gamma)
        print seed,":",bf_log_partition, a, b
        
        self.assertAlmostEqual(bf_log_partition,a,places=6)
        self.assertAlmostEqual(bf_log_partition,b,places=6)
    
        # Test map inference
        # map_score = nlr.log_likelihood(X,z,w,inference_type='map')
        map_o,map_y,map_score = nlr._map_inference(x,t,z)
        self.assertAlmostEqual(bf_map_score,map_score,places=6)
        self.assertTrue(np.all(bf_map_o==map_o))
        self.assertTrue(np.all(bf_map_y==map_y))

        # Test grad
        # nlr.backend = 'theano'
        # nlr.set_grads()
        # g_fun2 = nlr.get_grad([X],[[z,y]])
        
if __name__=="__main__":
    unittest.main()
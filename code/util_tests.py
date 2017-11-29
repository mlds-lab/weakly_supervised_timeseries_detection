import numpy as np
import theano
import theano.tensor as tensor
import unittest
from util import *

class UtilTester(unittest.TestCase):
    def get_theano_normal_log_pdf(self):
        x_s = tensor.scalar()
        mu_s = tensor.scalar()
        std_s = tensor.scalar()
        d_s = tensor.scalar()
        
        out_s = symbolic_normal_log_pdf(x_s,mu_s,std_s,d_s)
        return theano.function(inputs=[x_s,mu_s,std_s,d_s],outputs=out_s)
        
    def test_normal_log_pdf(self):
        from scipy.stats import truncnorm
        mu = 3.0
        var = 2.0
        d = 1.0
        normal_log_pdf2 = self.get_theano_normal_log_pdf()
        for x in np.linspace(mu-d,mu+d,50):
            # print x
            expected = truncnorm.logpdf(x, -d/np.sqrt(var), d/np.sqrt(var), mu, np.sqrt(var))
            got1 = normal_log_pdf(x,mu,var,d)
            got2 = normal_log_pdf2(x,mu,var,d)
            # print x,expected,got1,got2
            self.assertAlmostEqual(got1,expected,places=6)
            self.assertAlmostEqual(got2,expected,places=6)
            
    def test_inverse_gamma_log_pdf(self):
        from scipy.stats import invgamma
        alpha = 3.0
        beta = 2.0
        for x in np.linspace(0.001,10,25):
            expected = invgamma.logpdf(x,alpha,scale=beta)
            got = inverse_gamma_log_pdf(x,alpha,beta)
            # print x,got,expected
            self.assertAlmostEqual(got,expected,places=6)
            
    def test_beta_log_pdf(self):
        from scipy.stats import beta
        a = 3.0
        b = 2.0
        for x in np.linspace(0.001,0.999,25):
            expected = beta.logpdf(x,a,b)
            got = beta_log_pdf(x,a,b)
            # print x,got,expected
            self.assertAlmostEqual(got,expected,places=6)
            
if __name__=="__main__":
    unittest.main()
import unittest
import torch

from . import KL_multiple_univariate_gaussians

class MyTestCase(unittest.TestCase):

    def test_equality(self):
        mu = torch.randn(10,3) # batch size 10, 3 dimensions
        log_var = torch.randn(10,3)
        KL_loss = KL_multiple_univariate_gaussians(mu, mu, log_var, log_var, reduce=True)
        self.assertEqual(KL_loss,0.)

    def test_standard_gaussian(self):
        mu = torch.randn(10,3) # batch size 10, 3 dimensions
        log_var = torch.randn(10,3)
        KL_loss = KL_multiple_univariate_gaussians(mu, torch.zeros_like(mu),
                                                   log_var, torch.zeros_like(log_var),
                                                   reduce=True)
        KL_loss_simplified = -0.5 * torch.sum(1 + 2*log_var - mu.pow(2) - log_var.exp().pow(2), 1).mean()
        self.assertTrue(torch.eq(KL_loss_simplified, KL_loss))



if __name__ == '__main__':
    unittest.main()

import unittest
from . import cyclic_beta_schedule, linear_determ_warmup

class BetaScheduleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cycle_len = 10000
        cls.warmup_time = 10000
        cls.initial_beta = 10.0

    def test_beginning_cycle(self):
        beta0 = cyclic_beta_schedule(self.initial_beta, 0)
        beta10k = cyclic_beta_schedule(self.initial_beta, 10000)
        beta50k = cyclic_beta_schedule(self.initial_beta, 50000)
        beta10k2 = cyclic_beta_schedule(self.initial_beta, 10050) # it should increase continuously
        self.assertEqual(0.0, beta0)
        self.assertEqual(0.0, beta10k)
        self.assertEqual(0.0, beta50k)
        self.assertNotEqual(0.0, beta10k2)

    def test_increase_step_cycle(self):
        iter_num = 3000
        iter_num2 = 4000
        beta = cyclic_beta_schedule(self.initial_beta, iter_num)
        beta2 = cyclic_beta_schedule(self.initial_beta, iter_num2)
        weight = min(((2*iter_num)/self.cycle_len),1.0)
        self.assertEqual(beta, weight*self.initial_beta)
        weight2 = min(((2*iter_num2)/self.cycle_len),1.0)
        self.assertEqual(beta2, weight2*self.initial_beta)

    def test_constant_cycle(self):
        beta50 = cyclic_beta_schedule(self.initial_beta, 5000)
        beta60 = cyclic_beta_schedule(self.initial_beta, 6000)
        beta70 = cyclic_beta_schedule(self.initial_beta, 7000)
        self.assertEqual(beta50,beta60)
        self.assertEqual(beta50,beta70)

    def test_increase_step_warmup(self):
        iter_num = 3000
        iter_num2 = 4000
        beta = linear_determ_warmup(self.initial_beta, iter_num)
        beta2 = linear_determ_warmup(self.initial_beta, iter_num2)
        weight = min(((iter_num)/self.cycle_len),1.0)
        self.assertEqual(beta, weight*self.initial_beta)
        weight2 = min(((iter_num2)/self.cycle_len),1.0)
        self.assertEqual(beta2, weight2*self.initial_beta)

    def test_constant_warmup(self):
        beta100 = linear_determ_warmup(self.initial_beta, self.cycle_len+10)
        beta300 = linear_determ_warmup(self.initial_beta, self.cycle_len*3+10)
        self.assertEqual(beta100,beta300)
        self.assertEqual(beta100,self.initial_beta)



if __name__ == '__main__':
    unittest.main()

import unittest
from . import cyclic_beta_schedule, linear_determ_warmup

class BetaScheduleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cycle_len = 10000
        cls.warmup_time = 10000
        cls.increase_every = 100
        cls.initial_beta = 10.0

    def test_beginning_cycle(self):
        beta0 = cyclic_beta_schedule(self.initial_beta, 0)
        beta10k = cyclic_beta_schedule(self.initial_beta, 10000)
        beta50k = cyclic_beta_schedule(self.initial_beta, 50000)
        beta10k2 = cyclic_beta_schedule(self.initial_beta, 10050) # it should increase only every 100 steps
        self.assertEqual(0.0, beta0)
        self.assertEqual(0.0, beta10k)
        self.assertEqual(0.0, beta50k)
        self.assertEqual(0.0, beta10k2)

    def test_increase_step_cycle(self):
        slope = self.initial_beta/(self.cycle_len//(2*self.increase_every))
        beta2 = cyclic_beta_schedule(self.initial_beta, self.increase_every*2+10)
        beta3 = cyclic_beta_schedule(self.initial_beta, self.increase_every*3+10)
        self.assertAlmostEqual(beta3-beta2, slope)

    def test_constant_cycle(self):
        beta50 = cyclic_beta_schedule(self.initial_beta, self.cycle_len*50+10)
        beta60 = cyclic_beta_schedule(self.initial_beta, self.cycle_len*60+10)
        beta70 = cyclic_beta_schedule(self.initial_beta, self.cycle_len*70+10)
        self.assertEqual(beta50,beta60)
        self.assertEqual(beta50,beta70)

    def test_increase_step_warmup(self):
        slope = self.initial_beta/(self.warmup_time//self.increase_every)
        beta2 = linear_determ_warmup(self.initial_beta, self.increase_every*2+10)
        beta3 = linear_determ_warmup(self.initial_beta, self.increase_every*3+10)
        self.assertAlmostEqual(beta3-beta2, slope)

    def test_constant_warmup(self):
        beta100 = linear_determ_warmup(self.initial_beta, self.cycle_len*100+10)
        beta300 = linear_determ_warmup(self.initial_beta, self.cycle_len*300+10)
        self.assertEqual(beta100,beta300)
        self.assertEqual(beta100,self.initial_beta)



if __name__ == '__main__':
    unittest.main()

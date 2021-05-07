import unittest
from . import cyclic_beta_schedule

class BetaScheduleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cycle_len = 10000
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

    def test_increase_step(self):
        slope = self.initial_beta/(self.cycle_len//(2*self.increase_every))
        beta2 = cyclic_beta_schedule(self.initial_beta, self.increase_every*2+10)
        beta3 = cyclic_beta_schedule(self.initial_beta, self.increase_every*3+10)
        self.assertAlmostEqual(beta3-beta2, slope)

    def test_constant(self):
        beta50 = cyclic_beta_schedule(self.initial_beta, self.cycle_len*50+10)
        beta60 = cyclic_beta_schedule(self.initial_beta, self.cycle_len*60+10)
        beta70 = cyclic_beta_schedule(self.initial_beta, self.cycle_len*70+10)
        self.assertEqual(beta50,beta60)
        self.assertEqual(beta50,beta70)



if __name__ == '__main__':
    unittest.main()

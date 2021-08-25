"""Set of utilities for experimetns managers"""
import numpy as np


def cyclic_beta_schedule(initial_beta, iter_num):
    """ Implements cyclic scheduling for beta to solve KL annealing problem
    - initial_beta: the original value for beta to take
    - iter_num: number of current iteration
    Idea taken from: https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/"""
    cycle_len = 10000 # 10k iterations to complete one cycle
    relative_iter = iter_num%cycle_len
    weight = min(((2*relative_iter)/cycle_len),1.0) #half of the cycle constant at the maximum value
    return initial_beta*weight

def linear_determ_warmup(initial_beta, iter_num):
    """ Implements linear deterministich warm-up for beta to solve KL annealing problem
    - initial_beta: the original value for beta to take
    - iter_num: number of current iteration
    Taken from (Bowman et al.,2015; Sønderby et al., 2016)
    """
    warmup_time = 10000
    weight = min((iter_num/warmup_time),1.0)
    return weight*initial_beta

scheduler_switch = {
    'cyclic':cyclic_beta_schedule,
    'linear':linear_determ_warmup
}

def temperature_exponential_annealing(iter_num):
    """Annealing schedule fot temperature used in Gumbel softmax distribution as
    suggested in https://arxiv.org/pdf/1611.01144.pdf """
    r = 10e-3
    tau = max(min(1.0, np.exp(-r*iter_num)),0.01)
    return tau

class SchedulersManager():
    """ Responsible for updating all training hyperparameters subjected to schedules"""

    def __init__(self, model_name:str, params:dict):

        print("Initialising schedulers Manager...")
        self.variational = 'VAE' in model_name
        self.explicit = 'X' in model_name
        self.causal = 'C' in model_name
        self.weights = {}

        if self.variational:
            print("Variational mode ON")
            self.beta_scheduler = scheduler_switch[params["opt_params"]["beta_schedule"]]
            self.KL_weight_init = max(1.0, params["model_params"]["latent_size"]/params['data_params']['batch_size']) #this might be too high for the big datasets
            self.weights['KL_weight'] = self.KL_weight_init

        if self.causal:
            print("Causal mode ON")
            self.lamda_scheduler = scheduler_switch[params['opt_params']['inv_lamda_schedule']]
            self.invariance_lamda_init = params['model_params']['invariance_lamda']
            self.weights['invariance_lamda'] = self.invariance_lamda_init

        if self.explicit:
            print("Explicit mode ON")


    def update_weights(self, model, step_num):
        if self.variational:
            self.weights['KL_weight'] = self.beta_scheduler(self.KL_weight_init, step_num) # decaying the KL term
        if self.causal:
            self.weights['invariance_lamda'] = self.lamda_scheduler(self.invariance_lamda_init, step_num)
        if self.explicit:
            model.tau = temperature_exponential_annealing(step_num)

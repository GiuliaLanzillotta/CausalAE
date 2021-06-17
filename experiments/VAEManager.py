from experiments.data import DatasetLoader
from experiments.BaseManager import BaseExperiment
from models import VAE

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

class VAEXperiment(BaseExperiment):

    def __init__(self, params: dict) -> None:
        loader = DatasetLoader(params["data_params"])
        dim_in =  loader.data_shape # C, H, W
        model = VAE(params["model_params"], dim_in)
        super(VAEXperiment, self).__init__(params, model, loader)
        # Additional initialisations (used in training and validation steps)
        self.KL_weight = max(1.0, self.params["model_params"]["latent_size"]/self.params['data_params']['batch_size']) #this might be too high for the big datasets
        self.beta_scheduler = scheduler_switch[self.params["opt_params"]["beta_schedule"]]


    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        KL_weight = self.beta_scheduler(self.KL_weight, self.global_step) # decaying the KL term
        train_loss = self.model.loss_function(*results,
                                              X = input_imgs,
                                              KL_weight =  KL_weight)
        # Logging
        self.log('train_loss', train_loss["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log_dict({key: val.item() for key, val in train_loss.items()})
        self.log('beta', KL_weight*self.model.beta, prog_bar=True)
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            figure = self.visualiser.plot_training_gradients(self.global_step)
            self.logger.experiment.add_figure("gradient", figure, global_step=self.global_step)

        return train_loss["loss"]


    def score_FID(self, batch_idx, inputs, results):
        if batch_idx==0:
            self._fidscorer.start_new_scoring(self.params['data_params']['batch_size']*self.num_FID_steps,device=self.device)
        if  batch_idx<=self.num_FID_steps:#only one every 50 batches is included to avoid memory issues
            try: self._fidscorer.get_activations(inputs, self.model.act(results[0])) #store activations for current batch
            except Exception as e:
                print(e)
                print("Reached the end of FID scorer buffer")

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        KL_weight = self.beta_scheduler(self.KL_weight, self.global_step)# decaying the KL term
        val_loss = self.model.loss_function(*results, X = input_imgs, KL_weight = KL_weight)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_loss["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if (self.num_val_steps)%(self.score_every)==0 and self.num_val_steps!=0:
            self.score_FID(batch_idx, input_imgs, results)
        return val_loss


    def test_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)#
        test_loss = self.model.loss_function(*results, X = input_imgs, KL_weight = self.KL_weight)# no decay in KL weight
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('test_loss', test_loss["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.score_FID(batch_idx, input_imgs, results)


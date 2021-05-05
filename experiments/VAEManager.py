from experiments.data import DatasetLoader
from experiments.BaseManager import BaseExperiment
from models import VAE

class VAEXperiment(BaseExperiment):

    def __init__(self, params: dict) -> None:
        loader = DatasetLoader(params["data_params"])
        dim_in =  loader.data_shape # C, H, W
        model = VAE(params["model_params"], dim_in)
        super(VAEXperiment, self).__init__(params, model, loader)
        # Additional initialisations (used in training and validation steps)
        self.KL_weight = self.loader.num_samples/self.params['data_params']['batch_size']

    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        KL_weight = self.KL_weight*(self.params['opt_params']["KL_decay"]**(self.current_epoch//10)) # decaying the KL term
        train_loss = self.model.loss_function(*results,
                                              X = input_imgs,
                                              KL_weight =  KL_weight)
        # Logging
        self.log('train_loss', train_loss["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log_dict({key: val.item() for key, val in train_loss.items()})
        self.log('step', self.global_step, prog_bar=True)
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            self.visualiser.plot_training_gradients(self.global_step)

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        KL_weight = self.KL_weight*(self.params['opt_params']["KL_decay"]**(self.current_epoch//10)) # decaying the KL term
        val_loss = self.model.loss_function(*results, X = input_imgs, KL_weight = KL_weight)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_loss["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if (self.num_val_steps)%(self.score_every)==0 and self.num_val_steps!=0:
            if batch_idx==0:
                self._fidscorer.start_new_scoring(self.params['data_params']['batch_size']*self.num_FID_steps,device=self.device)
            if  batch_idx<=self.num_FID_steps:#only one every 50 batches is included to avoid memory issues
                try: self._fidscorer.get_activations(input_imgs, self.model.act(results[0])) #store activations for current batch
                except: print(self._fidscorer.start_idx)
        return val_loss

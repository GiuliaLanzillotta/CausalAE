import pytorch_lightning as pl
import torch
from torch import Tensor
from torch import optim
from experiments.data import DatasetLoader
from models import VAE
from visualisations import ModelVisualiser
from metrics import FIDScorer

class VAEXperiment(pl.LightningModule):

    def __init__(self, params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.params = params
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        self.loader = DatasetLoader(params["data_params"])
        dim_in =  self.loader.data_shape # C, H, W
        self.model = VAE(params["model_params"], dim_in)
        self.visualiser = ModelVisualiser(self.model,
                                          params["logging_params"]["name"],
                                          params["logging_params"]["version"],
                                          self.loader.test,
                                          **params["vis_params"])
        # Additional initialisations (used in training and validation steps)
        self.KL_weight = self.loader.num_samples/self.params['data_params']['batch_size']
        # For tensorboard logging (saving the graph)
        self.example_input_array = torch.rand((1,) + self.loader.data_shape)
        self._fidscorer = FIDScorer()


    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

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

        return train_loss["loss"]

    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch%self.params['vis_params']['plot_every']==0:
            self.visualiser.plot_training_gradients(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        if self.current_epoch%self.params['logging_params']['score_every']==0 and self.current_epoch!=0:
            if batch_idx==0:# initialise the scoring for the current epoch
                self._fidscorer.start_new_scoring(
                    self.params['data_params']['batch_size']*len(self.val_dataloader())//20,
                    device=self.device)
            if batch_idx%20==0:#only one every 20 batches is included to avoid memory issues
                try: self._fidscorer.get_activations(input_imgs, self.model.act(results[0])) #store activations for current batch
                except: print(self._fidscorer.start_idx)
        KL_weight = self.KL_weight*(self.params['opt_params']["KL_decay"]**(self.current_epoch//10)) # decaying the KL term
        val_loss = self.model.loss_function(*results,
                                            X = input_imgs,
                                            KL_weight = KL_weight)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_loss["loss"], prog_bar=True, logger=True)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        if (self.current_epoch%self.params['vis_params']['plot_every']==0) or \
                self.current_epoch==self.params["trainer_params"]["max_epochs"]:
            self.visualiser.plot_reconstructions(self.current_epoch, device=self.device)
            self.visualiser.plot_samples_from_prior(self.current_epoch, device=self.device)
            self.visualiser.plot_latent_traversals(self.current_epoch, device=self.device)
        if self.current_epoch%self.params['logging_params']['score_every']==0 and self.current_epoch!=0:
            # compute and store the fid scoring
            fid_score = self._fidscorer.calculate_fid()
            self.log("FID", fid_score, prog_bar=True)
        self.log("val_loss",avg_val_loss, prog_bar=True)

    def test_step(self, *args, **kwargs):
        #TODO
        pass

    def configure_optimizers(self):
        opt_params = self.params["opt_params"]
        optimizer = optim.Adam(self.model.parameters(),
                               lr=opt_params['LR'],
                               weight_decay=opt_params['weight_decay'])
        """
        if opt_params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = opt_params['scheduler_gamma'])
            return optimizer
        """
        return optimizer

    def train_dataloader(self):
        return self.loader.train

    def val_dataloader(self):
        return self.loader.val

    def test_dataloader(self):
        return self.loader.test

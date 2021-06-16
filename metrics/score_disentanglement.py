""" Module responsible for evaluating a model with multiple disentanglement metrics"""
from . import DCI, BetaVAE, IRS, MIG, ModExp, SAP
import numpy as np
from torch.utils.data import DataLoader

class ModelDisentanglementEvaluator(object):
    """ Takes a (trained) model and scores it against multiple disentanglement metrics. """
    def __init__(self, model, test_dataloader:DataLoader):
        self.model = model
        self.dataloader = test_dataloader
        self.test_input, _ = next(iter(test_dataloader))
        # Fix the random seed for reproducibility.
        self.random_state = np.random.RandomState(0)

    def score_model(self):
        complete_scores = {}
        disentanglement_scores = {}
        # DCI -----
        dci_results = DCI.compute_dci(self.dataloader, self.model, batch_size=self.dataloader.batch_size)
        disentanglement_scores['DCI'] = dci_results['disentanglement']
        complete_scores['DCI'] = dci_results
        # BetaVAE ----
        betaVAE_results = BetaVAE.compute_beta_vae_sklearn(self.dataloader.dataset, self.model,
                                                           batch_size=self.dataloader.batch_size)
        disentanglement_scores['BVAE'] = betaVAE_results['eval_accuracy']
        complete_scores['BVAE'] = betaVAE_results
        # IRS -----
        irs_results = IRS.compute_irs(self.dataloader, self.model, batch_size=self.dataloader.batch_size)
        disentanglement_scores['IRS'] = irs_results['disentanglement_scores']
        complete_scores['IRS'] = irs_results
        # MIG -----
        mig_results = MIG.compute_mig(self.dataloader, self.model, batch_size=self.dataloader.batch_size)
        disentanglement_scores['MIG'] = mig_results['discrete_mig']
        complete_scores['MIG'] = mig_results
        # ModExp -----
        modexp_results = ModExp.compute_modularity_explicitness(self.dataloader, self.model, batch_size=self.dataloader.batch_size)
        disentanglement_scores['ModExp'] = modexp_results['modularity_score']
        complete_scores['ModExp'] = modexp_results
        # SAP -----
        sap_results = SAP.compute_sap(self.dataloader, self.model, batch_size=self.dataloader.batch_size)
        disentanglement_scores['SAP'] = sap_results['SAP_score']
        complete_scores['SAP'] = sap_results
""" Module responsible for evaluating a model with multiple disentanglement metrics"""
from . import DCI, BetaVAE, IRS, MIG, ModExp, SAP
import numpy as np
from torch.utils.data import DataLoader
from models.BASE import GenerativeAE

class ModelDisentanglementEvaluator(object):
    """ Takes a (trained) model and scores it against multiple disentanglement metrics. """
    def __init__(self, model:GenerativeAE, test_dataloader:DataLoader):
        self.model = model
        self.dataloader = test_dataloader
        self.test_input, _ = next(iter(test_dataloader))
        # Fix the random seed for reproducibility.
        self.random_state = np.random.RandomState(0)

    def score_model(self, betaVAE=False, device='cpu'):
        print("Scoring model disentanglement.")
        complete_scores = {}
        disentanglement_scores = {}
        TRAIN_NUM, TEST_NUM = 1000,500
        # DCI -----
        print("DCI scoring")
        dci_results = DCI.compute_dci(self.dataloader, self.model.encode_mu, num_train=TRAIN_NUM, num_test=TEST_NUM,
                                      batch_size=self.dataloader.batch_size, device=device)
        disentanglement_scores['DCI'] = dci_results['disentanglement']
        complete_scores['DCI'] = dci_results
        if betaVAE:
            # BetaVAE ----
            print("BetaVAE scoring")
            betaVAE_results = BetaVAE.compute_beta_vae_sklearn(self.dataloader.dataset, self.model.encode_mu, num_train=TRAIN_NUM//10,
                                                               num_eval=TEST_NUM//10, batch_size=self.dataloader.batch_size, device=device)
            disentanglement_scores['BVAE'] = betaVAE_results['eval_accuracy']
            complete_scores['BVAE'] = betaVAE_results
        # IRS -----
        print("IRS scoring")
        irs_results = IRS.compute_irs(self.dataloader, self.model.encode_mu, num_train=TRAIN_NUM, batch_size=self.dataloader.batch_size, device=device)
        disentanglement_scores['IRS'] = irs_results['IRS']
        complete_scores['IRS'] = irs_results
        # MIG -----
        print("MIG scoring")
        mig_results = MIG.compute_mig(self.dataloader, self.model.encode_mu, num_train=TRAIN_NUM, batch_size=self.dataloader.batch_size, device=device)
        disentanglement_scores['MIG'] = mig_results['discrete_mig']
        complete_scores['MIG'] = mig_results
        # ModExp -----
        print("Modularity explicitness scoring")
        modexp_results = ModExp.compute_modularity_explicitness(self.dataloader, self.model.encode_mu, num_train=TRAIN_NUM, num_test=TRAIN_NUM,
                                                                batch_size=self.dataloader.batch_size, device=device)
        disentanglement_scores['ModExp'] = modexp_results['modularity_score']
        complete_scores['ModExp'] = modexp_results
        # SAP -----
        print("SAP scoring")
        sap_results = SAP.compute_sap(self.dataloader, self.model.encode_mu, num_train=TRAIN_NUM, num_test=TEST_NUM,
                                      batch_size=self.dataloader.batch_size, device=device)
        disentanglement_scores['SAP'] = sap_results['SAP_score']
        complete_scores['SAP'] = sap_results

        return disentanglement_scores, complete_scores
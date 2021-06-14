""" Module responsible for evaluating a model with multiple disentanglement metrics"""
from . import _compute_dci

class ModelDisentanglementEvaluator(object):
    #TODO
    """ Takes a (trained) model and scores it against multiple disentanglement metrics. """
    def __init__(self, model, test_dataloader):
        pass

    def score_model(self):
        #DCI = ...
        pass
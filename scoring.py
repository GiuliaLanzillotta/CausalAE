"""Simple script to run the scoring of given models"""
import os
from pathlib import Path
from experiments.EvaluationManager import ModelHandler, VectorModelHandler

if __name__ == '__main__':

    # Experiments settings ......
    #data_versions = ["standard","discrete","continuous","big"]
    #model_names = ["VecESAE","VecSAE","VecVAE","VecRSAE", "VecRAE","VecAE"]
    #model_versions = [["standard"],["standard","full"],["standard"],["standard", "full"],["standard"],["standard"]]

    model_names = ["BaseSAE","RSAE","RAE","AE"]
    #model_versions = [["standard","standardS"],["standard","standardS"],
    #                  ["standard","standardS"],["standard","standardS"],
    #                  ["standard","standardS"],["standard","standardS"]]
    model_versions = [["v121"],["v121"],["v121"],["v121"]]

    """
    for data_v in data_versions:
        for model_n, model_vs in zip(model_names, model_versions):
            for model_v in model_vs:
                handler = VectorModelHandler.from_config(model_name=model_n, model_version=model_v,
                                               data="SynthVec", data_version=data_v)
                handler = ModelHandler.from_config(model_name=model_n, model_version=model_v,
                                       data="MNIST", data_version=data_versions[0])
                handler.load_checkpoint() # loading latest checkpoint saved
                #handler.switch_labels_to_noises()
                handler.score_model(FID=True, disentanglement=False, orthogonality=True,
                                    save_scores=True, full=False, name="scoring_noises")
                #handler.latent_responses(num_batches=10, num_samples=100, store=True)
    """
    for model_n, model_vs in zip(model_names, model_versions):
        for model_v in model_vs:
            handler = ModelHandler.from_config(model_name=model_n, model_version=model_v, data="3DS")
            handler.load_checkpoint() # loading latest checkpoint saved
            handler.score_model(FID=True, disentanglement=False, orthogonality=True,
                                save_scores=True, full=False, name="scoring_noises")
            handler.latent_responses(num_batches=10, num_samples=100, store=True)

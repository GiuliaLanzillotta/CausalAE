"""Simple script to run the scoring of given models"""
import os
from pathlib import Path
from experiments.EvaluationManager import VectorModelHandler

if __name__ == '__main__':

    # Experiments settings ......
    data_versions = ["standard","discrete","continuous","big"]
    model_names = ["VecESAE","VecSAE","VecVAE"]
    for data_v in data_versions:
        for model_n in model_names:
            handler = VectorModelHandler(model_name=model_n, model_version="standard",
                                               data="SynthVec", data_version=data_v)
            handler.load_checkpoint() # loading latest checkpoint saved
            #handler.switch_labels_to_noises()
            handler.score_model(FID=False, disentanglement=True, orthogonality=True,
                                save_scores=True, full=False, name="scoring_noises")

"""Simple script to run the scoring of given models"""
import os
from pathlib import Path
from experiments.EvaluationManager import VectorModelHandler

if __name__ == '__main__':
    #handler = ModelHandler(model_name="BaseSAE", model_version="dummy", data="MNIST")
    handler = VectorModelHandler(model_name="VecESAE",  model_version="standard", data="SynthVec", data_version="standard")
    handler.load_checkpoint()
    scores = handler.score_model(FID=False, disentanglement=True, orthogonality=True, save_scores=True)


    handler = VectorModelHandler(model_name="VecESAE",  model_version="standard", data="SynthVec", data_version="continuous")
    handler.load_checkpoint()
    scores = handler.score_model(FID=False, disentanglement=True, orthogonality=True, save_scores=True)


    handler = VectorModelHandler(model_name="VecESAE",  model_version="standard", data="SynthVec", data_version="discrete")
    handler.load_checkpoint()
    scores = handler.score_model(FID=False, disentanglement=True, orthogonality=True, save_scores=True)


    handler = VectorModelHandler(model_name="VecESAE",  model_version="standard", data="SynthVec", data_version="big")
    handler.load_checkpoint()
    scores = handler.score_model(FID=False, disentanglement=True, orthogonality=True, save_scores=True)
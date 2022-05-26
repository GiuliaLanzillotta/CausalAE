"""Script for emergency evaluation of CelebA models"""

import os
from pathlib import Path
from typing import List

_cwd = os.getcwd()
from experiments.EvaluationManager import VisualModelHandler

if __name__ == '__main__':

    # General parameters --------------
    fig_path = Path("/cluster/scratch/glanzillo/figures/")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    figure_params = {"figsize":(32,32), "new_batch":True, "num_plots":8, "num_pics":8,
                     "grid_size":4, "num_samples":1, "steps": 6}

    # All model configurations ---------------
    base_version_name = "v32_big"
    rs_versions = ["13","17","37","121"]
    all_versions = [base_version_name] + [base_version_name+"random_seed"+rs for rs in rs_versions]
    models = ["AE","XAE","XCAE","BetaVAE","XVAE","XCVAE"]

    # Main loop
    print("CelebA evaluation script started.")
    print("-"*20)
    for model_name in models:
        variational = "V" in model_name #flag to use later
        for v in all_versions:
            params = {"model_name":model_name,
                      "model_version":v,
                      "data" : "CelebA"}
            # load handler
            handler = VisualModelHandler.from_config(**params)
            handler.load_checkpoint()
            """

            # Plotting -------------
            if not variational: # warming up the prior
                for i in range(10): handler.plot_model(do_reconstructions=True, **figure_params)

            # Plotting and saving reconstructions, random samples & multiple traversals
            print("Plotting ... ")
            res = handler.plot_model(do_reconstructions=True,
                                     do_random_samples=True,
                                     do_traversals=True,
                                     do_hybrisation=True,
                                     do_interpolationN=True,
                                     **figure_params)
            for k,fig in res.items():
                if type(fig) == List:
                    for i,f in enumerate(fig):
                        fname = fig_path / ("_".join([k,str(i),model_name,v]) + ".png")
                        f.savefig(fname, bbox_inches='tight', pad_inches=0)
                else:
                    fname = fig_path / ("_".join([k,model_name,v]) + ".png")
                    fig.savefig(fname, bbox_inches='tight', pad_inches=0)
            """

            print("Scoring ...")
            handler.score_model(save_scores=True,
                                update_general_scores=True,
                                random_seed=handler.config["logging_params"]["manual_seed"],
                                inference=False, **handler.config['eval_params'])
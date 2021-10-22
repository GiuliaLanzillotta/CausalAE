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
    models = {"AE":["v32_big"],
              "XAE":["v32_big","v32_x4_big"],
              "XCAE":["v32_big","v32_x4_big","v32_bigE"],
              "BetaVAE":["v32_big"],
              "XVAE":["v32_big","v32_x4_big"],
              "XCVAE":["v32_big","v32_x4_big","v32_bigE"]}

    # Main loop
    print("CelebA evaluation script started.")
    print("-"*20)
    for model_name, versions in models.items():
        variational = "V" in model_name #flag to use later
        for v in versions:
            params = {"model_name":model_name,
                      "model_version":v,
                      "data" : "CelebA"}
            # load handler
            handler = VisualModelHandler.from_config(**params)
            handler.load_checkpoint()

            # Plotting first -------------
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
                                random_seed=handler.config["logging_params"]["manual_seed"],
                                inference=False, **handler.config['eval_params'])"""
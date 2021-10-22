# Study on interventional consistency 

Codebase for the 'Study on Interventional Consistency'. 

<div align="center">
<img src="https://github.com/GiuliaLanzillotta/CausalAE/blob/main/random_samples_XCAExunit_dim4.png" 
alt="Random Samples MNIST" width="300"  border="30" />
<img src="https://github.com/GiuliaLanzillotta/CausalAE/blob/main/random_samples_AE_v32_big.png" 
alt="Random Samples CelebA" width="300"  border="30" />
<img src="https://github.com/GiuliaLanzillotta/CausalAE/blob/main/traversals_XAE_v32_x4_big_cut2.png" 
alt="Traversals CelebA" width="300"  border="30" />
</div>



### Dependencies
This repository uses *Pytorch 1.9* and *Pytorch Lightning*. <br>
The full set of dependencies is available in `requirements.txt`. 

### Executing an experiment
Base command to launch experiment: <br>
      `python main.py --name [MODEL NAME] --data [DATASET NAME] --version [VERSION NAME]` 
      
The supported models are listed in `models/__init__.py` together with their respective names. The model name identifies the directory to open under `configs/models`. Likewise, the dataset name identifies the dataset config file among those in `configs/data`. Finally, the model version determines the model config file among those in `configs/models/[MODEL NAME]/`. The model version determines the size of the model and training hyperparameters. The full config file is built based on the standard configuration given in `configs/standard.yaml`. For almost every model a dummy version is defined. The dummy version has a smaller size in order for the training to fit in approximately 2GB. 

The test routine (no training) can be actiivated with the command: <br>
      `python main.py --name [MODEL NAME] --data [DATASET NAME] --version [VERSION NAME] --test True` 

At the end of training or testing the model performance can be scored against multiple metrics by activating the score parameter:
      `python main.py --name [MODEL NAME] --data [DATASET NAME] --version [VERSION NAME] --score True`

Scoring is active by default.



### Structure of the repo 

#### `experiments` module 
The structure of each experiment (training/validating/testing) is defined by the respective functions from the `GenerativeAEExperiment`class, and the `BaseVisualExperiment`class, both subclassing the Pytorch Lightning `LightningModule` class. The evaluation routine for each experiments is managed by the `ModelHandler` class, which offers API to load trained models and score them against multiple metrics. The `VisualModelHandler` class additionally offers visual evaluation tools.

#### `models` module 
Each model family is implemented in a different module. Different versions of the same model families are defined in the same module. The `BASE.py` module defines the abstract base classes at the root of the ereditary tree. The `CAE.py` module contains the definition of the consistency trained versions of the baselines. 

#### `metrics` module 
Contains the definition of the metrics used to evaluate the models. Available metrics include: popular disentanglement metrics (BetaVAE, DCI, IRS, MIG, Modularity-Explicitness, SAP), Frechet Inception Distance (FID) score, Response matrix score, empirical dimensionality independence scores, Interventional consistency scores (INV,EQV,SCN). 

#### `datasets` module 
Implementation of dataset manager classes for external datasets. Implementation is provided for the Robot Finger Dataset (RFD), 3DShapes dataset, Synthetic vector dataset. The RFD dataset is offered in multiple versions: random access (`RFDh5`, based on HDF5 storage) and iterative (`RFD_IT`, based on .tar storage). Additionally, available datasets include: `MNIST`, `FashionMNIST`, `Cifar10`, `SVHN`, `CelebA`. The data loading operations are handled by the `DatasetLoader` class offered in the `experiments` module.


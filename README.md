# Interventional consistency

Codebase for the 'Study on Interventional Consistency'. 


### Dependencies

### Executing an experiment
Each experiment is defined by its configuration files. All the existing configurations are contained in the `config` module. 

### Structure of the repo 

#### `experiments` module 
The structure of each experiment (training/validating/testing) is defined by the respective functions from the `GenerativeAEExperiment`class, and the `BaseVisualExperiment`class, both subclassing the Pytorch Lightning `LightningModule` class. The evaluation routine for each experiments is managed by the `ModelHandler` class, which offers API to load trained models and score them against multiple metrics. The `VisualModelHandler` class additionally offers visual evaluation tools.

#### `models` module 
Each model family is implemented in a different module. Different versions of the same model families are defined in the same module. The `BASE.py` module defines the abstract base classes at the root of the ereditary tree. The `CAE.py` module contains the definition of the consistency trained versions of the baselines. 

#### `metrics` module 
Contains the definition of the metrics used to evaluate the models. Available metrics include: popular disentanglement metrics (BetaVAE, DCI, IRS, MIG, Modularity-Explicitness, SAP), Frechet Inception Distance (FID) score, Response matrix score, empirical dimensionality independence scores, Interventional consistency scores (INV,EQV,SCN). 

#### `datasets` module 
Implementation of dataset manager classes for external datasets. Implementation is provided for the Robot Finger Dataset (RFD), 3DShapes dataset, Synthetic vector dataset. The RFD dataset is offered in multiple versions: random access (`RFDh5`, based on HDF5 storage) and iterative (`RFD_IT`, based on .tar storage). Additionally, available datasets include: `MNIST`, `FashionMNIST`, `Cifar10`, `SVHN`, `CelebA`. The data loading operations are handled by the `DatasetLoader` class offered in the `experiments` module.


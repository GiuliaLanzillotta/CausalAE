"""Script for the creation and maintanance of synthetic datasets """
from pathlib import Path

""" Implementation of loading functions for 3dshpaes dataset """
from typing import Any, Optional, Callable, List
from .disentanglement_datasets import DisentanglementDataset
import numpy as np
import networkx as nx
from torch import nn, distributions
from models import layers, utils
import h5py
import torch
import urllib
import pickle
import os
from torchvision.datasets import VisionDataset
from .utils import gen_bar_updater, transform_discrete_labels

def matrix_poly_np(matrix, d):
    x = np.eye(d) + matrix/d
    return np.linalg.matrix_power(x, d)

def expm_np(A, m):
    expm_A = matrix_poly_np(A, m)
    h_A = np.trace(expm_A) - m
    return h_A

def generate_random_acyclic_graph(num_nodes, seed:int):
    exp_edges = .4
    p = exp_edges / (num_nodes - 1) #probabilityte edge from i
    acyclic = False
    while not acyclic:
        graph = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes, p, directed=True, seed=seed)
        acyclic = expm_np(nx.to_numpy_matrix(graph), num_nodes) == 0

    node_order = list(nx.topological_sort(graph))
    node_ancestors = [list(graph.predecessors(node)) for node in node_order]
    return (graph, node_order, node_ancestors)

def get_random_MLP(num_layers:int, din:int, dout:int=1) -> nn.Module:
    sizes = [din for _ in range(num_layers-1)]
    sizes.append(dout)
    mlp = layers.FCBlock(din, sizes, act=nn.ELU)
    #initialisation
    mlp.apply(lambda m: utils.standard_initialisation(m, "elu"))
    #setting grad tracking off
    for param in mlp.parameters():
        param.requires_grad = False
    return mlp

def build_equations(node_order: List[int], node_ancestors) -> List[nn.Module]:
    """Returns a list of non-linear equations (parametrised by random mlp) - one
    for each node in the input graph"""
    eqs = [None]*len(node_order) #one equation for each node
    num_layers=2
    for node in node_order:
        din = 1 + len(node_ancestors[node]) # noise + ancestors --> this will be the dimensionality of the input for the mlp
        equation = get_random_MLP(num_layers,din)
        eqs[node] = equation
    return eqs

def sample_statistics():
    mu = torch.rand(1)*50 - 25
    std = torch.rand(1)*10
    return mu,std

def sample_weights():
    num_values = torch.randint(2, 21, (1,1))[0][0]
    weights = torch.rand(num_values)
    return weights

def get_samplers(num_variables:int, all_discrete:bool, all_continuous:bool):
    """Returns a list of probability distributions (implementing .sample())
    that constitute the noise associated to each causal variable.
    If the variable is continuous the distribution will be Normal,
    if it is discrete it will be categorical."""
    if not (all_discrete or all_continuous):
        num_discrete = torch.randint(0,num_variables+1,(1,1))[0][0]
    elif all_discrete: num_discrete = num_variables
    else: num_discrete = 0
    num_continuous = num_variables-num_discrete
    samplers = []
    for i in range(num_continuous):
        mu, std = sample_statistics()
        samplers.append(distributions.Normal(mu,std))
    for i in range(num_discrete):
        weights = sample_weights()
        samplers.append(distributions.Categorical(weights))
    return samplers



class SynthVec(DisentanglementDataset):
    """
    Synthetic vectors dataset
    ----------------------------
    This dataset is generated in 2 steps:
    1) the 'causal'/'parent' variables are obtained by ancestral sampling from a random acyclic graph
    2) the 'observations' are computed as a non-linear function (parametrised by a mlp) of the causal variables
    """

    data_file = 'synth_vec.h5'
    factors_dict_file = 'factors.pkl'
    _FACTORS_IN_ORDER = None
    _NUM_VALUES_PER_FACTOR = None #property of the dataset -not of the generation process itself (obtained in factorisation step)
    NUM_TRAIN = 100000
    NUM_TEST = 2000
    MAPPING_DEPTH = 5

    def __init__(self,
                 root: str,
                 num_factors:int,
                 dim_observations:int,
                 allow_continuous:bool=False, #whether to include continuous variables in the causal graph
                 allow_discrete:bool=True, #whether to include discrete variables   //          //
                 generate: bool = False) -> None:

        super(SynthVec, self).__init__()

        self.root = root
        self.dim_in = num_factors
        self.dim_out = dim_observations
        self.allow_continuous = allow_continuous
        self.allow_discrete = allow_discrete

        if generate: self.generate()

        if not self._check_generated():
            raise RuntimeError('Dataset not found.' +
                               ' You can use generate=True to synthetise it')

        # --------- load it
        X,Y = self.read_source_file()
        # --------- load factors dictionary
        self.factors = self.factorise()

        print("Dataset loaded.")


    def categorise_labels(self, labels:np.ndarray):
        """Turn labels into categorical variables, and store them as integers.
        labels: numpy array of shape (num_samples, num_factors) containing the labels."""
        raise NotImplementedError

    def __repr__(self):
        """ Str representation of the dataset """
        raise NotImplementedError

    def read_source_file(self):
        """ Reads the .h5 file into training and test arrays"""
        raise NotImplementedError

    def check_and_substitute(self, factors:np.ndarray, other_factors:np.ndarray, index:int):
        """Checks if all the factors in the factors array exists in the dataset
        - overrides the implementation given in DisentanglementDataset superclass"""
        raise NotImplementedError

    def factorise(self):
        """ Creates the factors dictionary, i.e. a dictionary storing the index relative
        to any factor combination. This is the core of sample_observations_from_factors."""
        raise NotImplementedError

    def sample(self, batch_size, node_order, node_ancestors, samplers, equations):
        """
        Sampling batch_size samples from the generative model by passing through the variables
        in ancestral orders.
        Returns both the sampled data and the noise values.
        """
        updater = gen_bar_updater()
        x = torch.zeros(batch_size, self.dim_in)
        noises = {}
        for i,node in enumerate(node_order):
            noise = samplers[node].sample((batch_size,))
            noise = noise.reshape(batch_size, -1)
            noises[node] = noise.view(batch_size)
            if len(node_ancestors[node]):
                noise = torch.cat([noise, x[:,node_ancestors[node]]], -1)
            x[:,node] = equations[node](noise).squeeze(-1)
            updater(i, 1, self.dim_in) #visual feedback of the progress
        if batch_size is None:
            x = x.squeeze(0)
        return x,noises

    def generate(self, store:bool=True):
        """ Downloading source files (.h5)"""
        os.makedirs(self.raw_folder, exist_ok=True)
        # --------- download source file
        if self._check_generated():
            print("Files already there. Proceed to reading.")
            return
        # first generate the generating structure
        graph, node_order, node_ancestors = generate_random_acyclic_graph(self.dim_in, seed=11)
        samplers = get_samplers(self.dim_in, not self.allow_continuous, not self.allow_discrete)
        equations = nn.ModuleList(build_equations(node_order, node_ancestors))
        obs_mapping = get_random_MLP(num_layers=self.MAPPING_DEPTH, din=self.dim_in, dout=self.dim_out)
        #then generate causal variables and observations
        print("Generate training samples")
        y_train, noises_train = self.sample(self.NUM_TRAIN, node_order, node_ancestors, samplers, equations)
        x_train = obs_mapping(y_train)
        print("Generate testing samples")
        y_test, noises_test = self.sample(self.NUM_TEST, node_order, node_ancestors, samplers, equations)
        x_test = obs_mapping(y_test)
        # grouping all generative information
        metadata = {
            "graph":graph,
            "node_order":node_order,
            "node_ancestors":node_ancestors,
            "samplers":samplers,
            "obs_mapping":obs_mapping
        }

        if store:
            print("Storing generated data.")
            with h5py.File(Path(self.raw_folder)/self.data_file, "w") as hf:
                X1 = hf.create_dataset("x_train", data=x_train)
                Y1 = hf.create_dataset("y_train", data=y_train)
                X2 = hf.create_dataset("y_train", data=x_test)
                Y2 = hf.create_dataset("y_test", data=y_test)
            print("Storing metadata")
            with open(Path(self.raw_folder)/"metadata.pkl","w") as mf:
                pickle.dump(metadata, mf)
        print("Done!")

        return X1,Y1,X2,Y2,metadata

    def _check_generated(self):
        raise NotImplementedError

    def _check_factorised(self):
        """Checking the existence of the factors dictionary."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def raw_folder(self) -> str:
        # raw folder should be ("./datasets/SynthVec/SynthVec/raw
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        # raw folder should be ("./datasets/SynthVec/SynthVec/processed
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def num_factors(self):
        return self.dim_in

    @property
    def factors_names(self):
        return self._FACTORS_IN_ORDER

    @property
    def factors_num_values(self):
        return self._NUM_VALUES_PER_FACTOR




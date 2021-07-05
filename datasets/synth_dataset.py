"""Script for the creation and maintanance of synthetic datasets """
from pathlib import Path
import scipy.stats
from typing import Any, Optional, Callable, List, Union, overload
from .disentanglement_datasets import DisentanglementDataset
import numpy as np
import networkx as nx
from torch import nn, distributions, Tensor
from models import layers, utils
from metrics.utils import _histogram_discretize
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
    print("Generating graph...")
    exp_edges = .65
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
    num_values = torch.randint(2, 16, (1,1))[0][0] # max 15 dimensional categorical variables
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
    is_discrete = []
    for i in range(num_continuous):
        mu, std = sample_statistics()
        samplers.append(distributions.Normal(mu,std))
        is_discrete.append(False)
    for i in range(num_discrete):
        weights = sample_weights()
        samplers.append(distributions.Categorical(weights))
        is_discrete.append(True)
    return samplers, is_discrete

def discretise_Normal(mu,std, num_bins=20):
    """Transforming continuous variable drawn from normal distribution into a
    discrete variable by exploiting its quantiles (in this way we can generalise
    to unseen samples)
    Returns a list of num_bins centers for the new categorical
    """
    centers = []
    probs = np.linspace(0.,1., num_bins+1, endpoint=False)[1:]
    for i in range(num_bins):
     centers.append(scipy.stats.norm.ppf(probs[i], loc=mu, scale=std))
    return centers

def categorise_label(label:np.ndarray, discrete:bool, distrib, num_categories):
    """Turn labels into categorical variables, and store them as integers.
    labels: numpy array of shape (num_samples, num_factors) containing the labels."""
    if len(label.shape)>1: label.squeeze(-1)
    if discrete: return label
    else:
        centers = discretise_Normal(distrib.mean, distrib.stddev, num_bins=num_categories)
        return np.digitize(label, centers)


class SynthVec(DisentanglementDataset):
    """
    Synthetic vectors dataset
    ----------------------------
    This dataset is generated in 2 steps:
    1) the 'causal'/'parent' variables are obtained by ancestral sampling from a random acyclic graph
    2) the 'observations' are computed as a non-linear function (parametrised by a mlp) of the causal variables
    """

    data_file = 'synth_vec.h5'
    _FACTORS_IN_ORDER = []
    _NUM_VALUES_PER_FACTOR = {} #property of the dataset -not of the generation process itself (obtained in factorisation step)
    NUM_TRAIN = 100000
    NUM_TEST = 2000
    MAPPING_DEPTH = 5
    NUM_CATEG_CONTINUOUS = 20

    def __init__(self,
                 root: str,
                 name: str,
                 num_factors:int,
                 dim_observations:int,
                 allow_continuous:bool=False, #whether to include continuous variables in the causal graph
                 allow_discrete:bool=True, #whether to include discrete variables   //          //
                 generate: bool = False,
                 overwrite:bool = False,
                 test:bool = False,
                 noise:bool=False, #whether to use noises as labels or the causal variables (for disentanglement scoring)
                 verbose:bool=False) -> None:

        super(SynthVec, self).__init__()

        self.name = name
        self.root = root
        self.dim_in = num_factors
        self.dim_out = dim_observations
        self.allow_continuous = allow_continuous
        self.allow_discrete = allow_discrete
        self.test = test
        self.noise = noise

        if generate:
            X1,Y1,N1,X2,Y2,N2,metadata = self.generate(store=True, overwrite=overwrite)
        else:
            if not self._check_generated():
                raise RuntimeError('Dataset not found.' +
                                   ' You can use generate=True to synthetise it')
            # --------- load it
            X1,Y1,N1,X2,Y2,N2,metadata = self.read_source_files()
        if self.test:
            self.data = X2
            self.original_labels = N2 if noise else Y2
        else:
            self.data = X1
            self.original_labels = N1 if noise else Y1
        self.labels = self.categorise_labels(self.original_labels[()]) # used for disentanglement purposes
        self.metadata = metadata
        self.elaborate_info() # extract relevant disentanglement-testing info from metadata
        print("Dataset loaded.")

        if verbose: print(self)

    def switch_to_noises(self):
        """Function has an effect ONLY IF self.noise is False.
        It recovers the noises from file and sets them as new labels.
        Useful for evaluation purposes."""
        X1,Y1,N1,X2,Y2,N2,metadata = self.read_source_files()
        if self.test:
            self.original_labels = N2
        else:
            self.original_labels = N1
        self.labels = self.categorise_labels(self.original_labels[()])

    def get_graph_matrix(self):
        return nx.to_numpy_matrix(self.metadata["graph"])

    def __repr__(self):
        """ Str representation of the dataset """
        head = "Dataset {0} info".format(self.__class__.__name__)
        body = ["Size = {0}".format(len(self)), "Factors of variation : "]
        for n,v_num in self._NUM_VALUES_PER_FACTOR.items():
            line = n+" with "+str(v_num)+" values"
            body.append(line)
        lines = [head] + [" " * 2 + line for line in body]
        return '\n'.join(lines)

    def categorise_labels(self, labels):
        """Turn labels into categorical variables, and store them as integers"""
        return _histogram_discretize(labels.T, self.NUM_CATEG_CONTINUOUS).T

    def elaborate_info(self):
        """Extracts dataset information from the metadata

        metadata = {
            "graph":graph,
            "node_order":node_order,
            "node_ancestors":node_ancestors,
            "samplers":samplers,
            "discrete":discrete,
            "obs_mapping":obs_mapping
        }"""
        self._FACTORS_IN_ORDER = [""]*self.dim_in
        for node in self.metadata["node_order"]:
            distrib = self.metadata["samplers"][node]
            discrete = self.metadata["discrete"][node]
            name = f"factor{node}_discrete" if discrete else f"factor{node}_continuous"
            self._FACTORS_IN_ORDER[node] = name
            self._NUM_VALUES_PER_FACTOR[name] = len(distrib.probs) if discrete else self.NUM_CATEG_CONTINUOUS
        self.graph = self.metadata["graph"]

    def read_source_files(self):
        """ Reads the .h5 file into training and test arrays"""
        print("Loading generated data.")
        with h5py.File(Path(self.raw_folder)/self.data_file, "r") as hf:
            X1 = hf["x_train"][()]
            Y1 = hf["y_train"][()]
            N1 = hf["noises_train"][()]
            X2 = hf["x_test"][()]
            Y2 = hf["y_test"][()]
            N2 = hf["noises_test"][()]
        print("Loading metadata")
        with open(Path(self.raw_folder)/"metadata.pkl","rb") as mf:
            metadata = pickle.load(mf)

        return X1,Y1,N1,X2,Y2,N2,metadata

    def check_and_substitute(self, factors:np.ndarray, other_factors:np.ndarray, index:int):
        """Checks if all the factors in the factors array exists in the dataset
        - overrides the implementation given in DisentanglementDataset superclass"""
        return factors

    def sample(self, batch_size, node_order, node_ancestors, samplers, equations, noises:Tensor=None):
        """
        Sampling batch_size samples from the generative model by passing through the variables
        in ancestral orders.
        Returns both the sampled data and the noise values.
        """
        sample_noises = noises is None
        updater = gen_bar_updater()
        x = torch.zeros(batch_size, self.dim_in)
        if sample_noises: noises = [None]*self.dim_in
        for i,node in enumerate(node_order):
            if sample_noises:
                noise = samplers[node].sample((batch_size,))
                noise = noise.reshape(batch_size, -1)
                noises[node] = noise.view(batch_size)
            else: noise= noises[:,node]
            if len(node_ancestors[node]):
                noise = torch.cat([noise, x[:,node_ancestors[node]]], -1)
            x[:,node] = equations[node](noise.float()).squeeze(-1)
            updater(i+1, 1, self.dim_in) #visual feedback of the progress
        if batch_size==1: x = x.squeeze(0)
        return x, torch.stack(noises, axis=1) #batch size x dim_in

    def sample_observations_from_causes(self, X):
        observations = self.metadata["obs_mapping"](X)
        return observations

    def sample_pairs_observations(self, num):
        """ Samples a batch of pairs of observations as used in BetaVAE disentanglement metric.
            -> only one factor index fixed for every pair"""
        X,U = self.sample(num, self.metadata["node_order"],
                          self.metadata["node_ancestors"],
                          self.metadata["samplers"],
                          self.metadata["equations"])
        X2,U2 = self.sample(num, self.metadata["node_order"],
                            self.metadata["node_ancestors"],
                            self.metadata["samplers"],
                            self.metadata["equations"])

        index = np.random.randint(0,self.dim_in)

        if self.noise:
            first_factors = U
            second_factors = U2
        else:
            first_factors = X
            second_factors = X2

        second_factors[:,index] = first_factors[:,index]
        obs1 = self.sample_observations_from_causes(X)
        obs2 = self.sample_observations_from_causes(X2)
        return index, obs1, obs2

    def generate(self, store:bool=True, overwrite:bool=False):
        """ Downloading source files (.h5)"""
        os.makedirs(self.raw_folder, exist_ok=True)
        # --------- download source file
        if self._check_generated() and not overwrite:
            print("Files already there. Proceed to reading.")
            return self.read_source_files()
        # first generate the generating structure
        graph, node_order, node_ancestors = generate_random_acyclic_graph(self.dim_in, seed=11)
        samplers, discrete = get_samplers(self.dim_in, not self.allow_continuous, not self.allow_discrete)
        equations = nn.ModuleList(build_equations(node_order, node_ancestors))
        obs_mapping = get_random_MLP(num_layers=self.MAPPING_DEPTH, din=self.dim_in, dout=self.dim_out)
        #then generate causal variables and observations
        print("Generate training samples")
        y_train, noises_train = self.sample(self.NUM_TRAIN, node_order, node_ancestors, samplers, equations)
        x_train = obs_mapping(y_train.float())
        print("Generate testing samples")
        y_test, noises_test = self.sample(self.NUM_TEST, node_order, node_ancestors, samplers, equations)
        x_test = obs_mapping(y_test.float())
        # grouping all generative information
        metadata = {
            "graph":graph,
            "node_order":node_order,
            "node_ancestors":node_ancestors,
            "samplers":samplers,
            "discrete":discrete,
            "obs_mapping":obs_mapping
        }

        if store:
            print("Storing generated data.")
            with h5py.File(Path(self.raw_folder)/self.data_file, "w") as hf:
                #labels have to be numpy arrays
                y_train = y_train.cpu().numpy()
                noises_train = noises_train.cpu().numpy()
                y_test = y_test.cpu().numpy()
                noises_test = noises_test.cpu().numpy()
                #TODO: maybe zip?
                X1 = hf.create_dataset("x_train", data=x_train)[()]
                Y1 = hf.create_dataset("y_train", data=y_train)[()]
                N1 = hf.create_dataset("noises_train", data=noises_train)[()]
                X2 = hf.create_dataset("x_test", data=x_test)[()]
                Y2 = hf.create_dataset("y_test", data=y_test)[()]
                N2 = hf.create_dataset("noises_test", data=noises_test)[()]

            print("Storing metadata")
            with open(Path(self.raw_folder)/"metadata.pkl","wb") as mf:
                pickle.dump(metadata, mf)
        print("Done!")
        return X1,Y1,N1,X2,Y2,N2,metadata

    def _check_generated(self):
        return os.path.exists(Path(self.raw_folder)/self.data_file) and \
               os.path.exists(Path(self.raw_folder)/"metadata.pkl")

    def __getitem__(self, index: int) -> Any:
        x = self.data[index]
        y = torch.tensor(self.labels[index])
        return x,y

    def __len__(self) -> int:
        return self.data.shape[0]

    @property
    def raw_folder(self) -> str:
        # raw folder should be ("./datasets/SynthVec/SynthVec/raw
        return os.path.join(self.root, self.__class__.__name__, 'raw' , self.name)

    @property
    def processed_folder(self) -> str:
        # raw folder should be ("./datasets/SynthVec/SynthVec/processed
        return os.path.join(self.root, self.__class__.__name__, 'processed', self.name)

    @property
    def num_factors(self):
        return self.dim_in

    @property
    def factors_names(self):
        return self._FACTORS_IN_ORDER

    @property
    def factors_num_values(self):
        return self._NUM_VALUES_PER_FACTOR




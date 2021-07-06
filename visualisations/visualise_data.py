"""Script offering a series of tools for the visualisation of datasets"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import networkx as nx
import numpy as np


class SynthVecDataVisualiser(object):

    APPROX_LEVEL = 10000

    def __init__(self, dataloader:DataLoader, **kwargs):
        super(SynthVecDataVisualiser, self).__init__()
        self.loader = dataloader
        self.dataset = self.loader.dataset.dataset
        self.meta = self.dataset.metadata
        # Fix the random seed for reproducibility.
        self.random_state = np.random.RandomState(0)

    def plot_graph(self, figsize=(7,4)):
        figure = plt.figure(figsize=figsize)
        nx.draw(self.meta["graph"], with_labels=True, node_size=300, node_color="#99dddd",
                edge_color="#55ff11", pos=nx.planar_layout(self.meta["graph"]))
        return figure

    def plot_noises_distributions(self, figsize=(10,20)):
        samplers = self.meta["samplers"]
        discrete = self.meta["discrete"]
        figure, axs = plt.subplots(nrows=len(samplers), figsize=figsize)
        for i in range(len(samplers)):
            if discrete[i]: #TODO: make this one look nicer
                sns.distplot(samplers[i].sample([self.APPROX_LEVEL]), ax = axs[i], kde=False)
            else:
                axs[i].set_xlim(-40,40)
                sns.distplot(samplers[i].sample([self.APPROX_LEVEL]), ax = axs[i])
        return figure


    def plot_causes2noises(self, figsize=(10,20)):
        X1,Y1,N1,X2,Y2,N2,metadata = self.dataset.read_source_files()
        discrete = self.meta["discrete"]
        figure = plt.figure(figsize=figsize)
        for i in range(N1.shape[1]):
            kind = "scatter" if discrete[i] else "kde"
            sns.jointplot(N1[:,i], Y1[:,i], kind=kind)
        return figure


    
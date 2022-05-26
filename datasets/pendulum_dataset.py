"""Script responsible for the creation and management of the PENDULUM dataset - a synthetic dataset used in
the CausalVAE paper for interventional experiments.
Code from https://github.com/huawei-noah/trustworthyAI/tree/934a924a7849e9f6abf0f561f59a1211dca435ad/Causal_Disentangled_Representation_Learning/causal_data"""


#TODO
# use this code for generation: https://github.com/huawei-noah/trustworthyAI/blob/934a924a7849e9f6abf0f561f59a1211dca435ad/Causal_Disentangled_Representation_Learning/causal_data/pendulum.py
# use this code for handling: https://github.com/huawei-noah/trustworthyAI/blob/934a924a7849e9f6abf0f561f59a1211dca435ad/Causal_Disentangled_Representation_Learning/run_pendulum.py
# build corresponding experiment configuration

from pathlib import Path
from typing import Any, Optional, Callable

import torchvision.datasets

from .disentanglement_datasets import DisentanglementDataset
import numpy as np
import h5py
import torch
import urllib
import pickle
import os
import matplotlib.image as mpimg
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import VisionDataset
from .utils import gen_bar_updater, transform_discrete_labels
import torchvision.transforms.functional as fn

def projection(phi, x, y, base = -0.5):
    """Utility method to compute projection of x onto y given angle"""
    b = y-x*math.tan(phi)
    shade = (base - b)/math.tan(phi)
    return shade

class Pendulum(torchvision.datasets.VisionDataset,DisentanglementDataset):
    """ Pundulum dataset utilised in CausalVAE experiments.
    Description:
        Each image contains 3 entities (PENDULUM, LIGHT, SHADOW), and 4 concepts
        ((PENDULUM ANGLE, LIGHT ANGLE) → (SHADOW LOCATION, SHADOW LENGTH)).
        In Pendulum generator, the image size is set to be 96 × 96 with 4 channels.
        We generate about 7k images (6k for training and 1k for inference),
        ϕ1 and ϕ2 are ranged in around [−π/4 , π/4 ], and they are generated independently.
        For each image, we provide 4 labels, which include light position,
        pendulum angle, shadow position and shadow length.
        For light position, we use the value of center of semicircle
        as supervision signal. For the pendulum angle, we use the value of φ2
        as supervision signal. For shadow position and shadow length, we use the
        length of 3/4 as supervision signal respectively.
    """


    shape = (4, 96, 96)
    _FACTORS_IN_ORDER = ['Pendulum angle', 'Light position', 'Shadow length', 'Shadow position']
    _FACTORS_KEYS = ['p','l','s','m']
    key_pad_len = 5

    PENDULUM_RANGE = (-40,44)
    LIGHT_RANGE = (60,148)
    SCALES = np.array([[0,44],
                      [100,40],
                      [7,7.5],
                      [10,10]])

    def __init__(
            self,
            root: str,
            generate: bool = False,
            overwrite: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None) -> None:

        super(Pendulum, self).__init__(root,
                                       transform=transform,
                                       target_transform=target_transform)

        self.root = root
        if generate: self.generate(overwrite=overwrite)

        if not self._check_generated():
            raise RuntimeError('Dataset not found.' +
                               ' You can use generate=True to synthetise it')
        # --------- load it
        self.paths, self.labels = self.read_source_file()

        # --------- load factors dictionary
        self.factors = self.read_labels_csv()
        self._NUM_FACTORS_PER_VALUE = self.count_num_factors_per_value(self.factors)
        print("Dataset loaded.")
        print(self)

    @staticmethod
    def get_factors_given_angles(pendulum, light):
        """Function hiding the math behind the scene generation.
        It first determines the position of the sun ball, then projects the light on the
        pendulum and finally computes the shade position and length."""

        theta = pendulum*math.pi/200.0
        phi = light*math.pi/200.0
        x = 10 + 8*math.sin(theta)
        y = 10.5 - 8*math.cos(theta)

        light_len = projection(phi, 10, 10.5, 20.5)

        #calculate the mid index of
        ball_x = 10+9.5*math.sin(theta)
        ball_y = 10.5-9.5*math. cos(theta)
        mid = (projection(phi, 10.0, 10.5)+projection(phi, ball_x, ball_y))/2
        shade = max(3,abs(projection(phi, 10.0, 10.5)-projection(phi, ball_x, ball_y)))

        return x,y,light_len,mid,shade

    def generate(self, overwrite=True):
        """Generates the Pendulum dataset
        The function will not generate a new dataset if the overwrite flag is off and the
        dataset already exists."""

        if self._check_generated() and not overwrite: return

        print("Fabricating Pendulum dataset")

        if not os.path.exists(Path(self.raw_folder)) or overwrite:
            os.makedirs(Path(self.raw_folder))

        data = pd.DataFrame(columns=['p', 'l', 'shade','mid'])
        #TODO: show progress bar while creating dataset
        for p in range(self.PENDULUM_RANGE[0], self.PENDULUM_RANGE[1]):
            for l in range(self.LIGHT_RANGE[0], self.LIGHT_RANGE[1]):
                if l==100: continue

                # computing the factors
                x,y,light,mid,shade = self.get_factors_given_angles(p,l)

                #drawing the image
                ball = plt.Circle((x,y), 1.5, color = 'firebrick')
                gun = plt.Polygon(([10,10.5],[x,y]), color = 'black', linewidth = 3)
                sun = plt.Circle((light,20.5), 3, color = 'orange')
                shadow = plt.Polygon(([mid - shade/2.0, -0.5],[mid + shade/2.0, -0.5]), color = 'black', linewidth = 3)
                ax = plt.gca()
                ax.add_artist(gun)
                ax.add_artist(ball)
                ax.add_artist(sun)
                ax.add_artist(shadow)
                ax.set_xlim((0, 20))
                ax.set_ylim((-1, 21))
                plt.axis('off')
                # saving the image
                file_name = 'a_' + str(int(p)) + '_' + str(int(l)) + '_' + str(int(shade)) + '_' + str(int(mid)) +'.png'
                plt.savefig(Path(self.raw_folder)/file_name,dpi=96)
                plt.clf()

                #creating new entry in the dataframe
                new=pd.DataFrame({'p':p, 'l':l,'s':shade,'m':mid},index=[1])
                data=pd.concat([data,new],ignore_index=True, sort=False)

        #saving labels dataframe
        filepath = Path(self.processed_folder)/"labels.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath)

        return

    def __repr__(self):
        """String description of the dataset"""
        head = "Pendulum dataset info"
        body = ["Size = {0}".format(len(self)), "Factors of variation : "]
        for i,n in enumerate(self.factors_names):
            line = n
            body.append(line)
        last_line = "For more info see original paper: https://arxiv.org/abs/2004.08697"
        lines = [head] + [" " * 2 + line for line in body] + [last_line]
        return '\n'.join(lines)

    def _check_generated(self):
        try:
            return len(os.listdir(Path(self.raw_folder)))!=0 and \
                   len(os.listdir(Path(self.processed_folder)))!=0
        except FileNotFoundError: return False

    def read_source_file(self):
        """ Reads the directory content of the raw data into a list of filenames and a numpy array"""
        print("Loading pendulum dataset files...")
        dirpath = Path(self.raw_folder)
        imgs_names = os.listdir(dirpath)
        imgs_paths = [os.path.join(dirpath, img) for img in imgs_names]
        imgs_labels = [list(map(int,img[:-4].split("_")[1:]))  for img in imgs_names]
        self.size = len(imgs_paths)
        return imgs_paths, imgs_labels

    def read_labels_csv(self):
        """ Reads the .csv file containing all the labels"""
        filepath = Path(self.processed_folder)/"labels.csv"
        data = pd.read_csv(filepath)
        return data

    def count_num_factors_per_value(self, labels_df):
        name2num = {}
        for n in self._FACTORS_KEYS:
            name2num[n]=labels_df[n].nunique()
        return name2num

    def __getitem__(self, index: int) -> Any:

        img_path = self.paths[index]
        label = torch.from_numpy(np.asarray(self.labels[index])).float()
        img = Image.open(img_path)
        img = fn.resize(img, list(self.shape[1:]))
        img = np.asarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


    def __len__(self) -> int:
        return self.size


    @property
    def raw_folder(self) -> str:
        # raw folder should be ("./datasets/Pendulum/Pendulum/raw
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        # raw folder should be ("./datasets/Pendulum/Pendulum/processed
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def num_factors(self):
        return len(self._FACTORS_IN_ORDER)

    @property
    def factors_names(self):
        return self._FACTORS_IN_ORDER

    @property
    def factors_num_values(self):
        return self._NUM_FACTORS_PER_VALUE


    def categorise_labels(self, labels):
        pass

    def sample_pairs_observations(self, num):
        """ Samples a batch of pairs of observations as used in BetaVAE disentanglement metric.
        -> only one factor index fixed for every pair
        Overriding implementation in Disentnglement Dataset"""
        obs1 = []
        obs2 = []
        count=0
        factor_idx = np.random.randint(0,self.num_factors)
        fixed_property = self._FACTORS_KEYS[factor_idx]
        grouped = self.factors.groupby(fixed_property).apply(lambda g: g.sample(n=num, replace=True))
        #sampling couples from each group randomly
        for idx in grouped.sample(frac=1).index: #applying shuffling to get random factors values
            group = grouped.loc[idx[0]].drop_duplicates()
            num_pairs = group.shape[0]//2
            obs1.append(group[:num_pairs].to_numpy())
            obs2.append(group[num_pairs:num_pairs*2].to_numpy())
            count+=num_pairs
            if count==num:
                return factor_idx, np.asarray(obs1), np.asarray(obs2)
        raise RuntimeError("Not enough pairs for the sampling on property "+fixed_property)





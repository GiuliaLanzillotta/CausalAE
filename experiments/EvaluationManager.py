"""Script managing evaluation and exploration of models"""
import pickle

import pandas as pd
import torch
from experiments import pick_model_manager, get_causal_block_graph
from pathlib import Path
from configs import get_config
from experiments.data import DatasetLoader
import os
import glob
import time
import copy

from experiments.utils import temperature_exponential_annealing
from visualisations import ModelVisualiser, SynthVecDataVisualiser, vis_responses, vis_xnets, vis_latents
from metrics import FIDScorer, ModelDisentanglementEvaluator, LatentOrthogonalityEvaluator, LatentConsistencyEvaluator
import pytorch_lightning as pl
from metrics.responses import compute_response_matrix



""" Evaluation toolbox for GenerativeAE models"""

class ModelHandler(object):
    """Offers a series of tools to inspect a given model."""
    def __init__(self, experiment:pl.LightningModule, **kwargs):
        self.config = experiment.params
        self.experiment = experiment
        self.model = self.experiment.model
        self.dataloader = self.experiment.loader
        self.model.eval()
        model_name = self.config["model_params"]["name"]
        self.model_name = model_name
        print(model_name+ " model hanlder loaded.")
        self.visualiser = None
        self.fidscorer = None
        self._orthogonalityScorer = None
        self._disentanglementScorer = None
        self._consistencyScorer = None
        self.device = next(self.model.parameters()).device
        self.send_model_to(self.device)
        # update Xnet tau
        if "X" in self.model_name: self.reset_tau()

    @classmethod
    def from_experiment(cls, experiment:pl.LightningModule, **kwargs):
        return cls(experiment, **kwargs)

    @classmethod
    def from_config(cls, model_name:str, model_version:str, data:str, data_version="", **kwargs):
        config = get_config(tuning=False, model_name=model_name, data=data,
                                 version=model_version, data_version=data_version)
        experiment = pick_model_manager(model_name)(params = config, verbose=kwargs.get("verbose"))
        return cls(experiment, **kwargs)

    def reset_tau(self, step_num=None):
        if step_num is None: step_num = self.experiment.global_step
        tau = temperature_exponential_annealing(step_num)
        self.model.tau = tau
        print(f"Tau set to {tau}")

    def score_disentanglement(self, **kwargs):
        """Scoring model on disentanglement metrics"""
        scores = {}
        full = kwargs.get("full",False) # full report
        if self._disentanglementScorer is None:
            self._disentanglementScorer = ModelDisentanglementEvaluator(self.model, self.dataloader.val)
        disentanglement_scores, complete_scores = self._disentanglementScorer.score_model(device = self.device)
        scores.update(disentanglement_scores)
        if full: scores["extra_disentanglement"] = complete_scores
        return scores

    def score_orthogonality(self, **kwargs):
        """Scoring the model against orthogonality measures """
        if self._orthogonalityScorer is None:
            try: unit_dim = self.model.unit_dim
            except AttributeError: unit_dim=1
            self._orthogonalityScorer = LatentOrthogonalityEvaluator(self.model, self.dataloader.val,
                                                                     self.model.latent_size, unit_dim)

        self.device = next(self.model.parameters()).device
        ortho_scores = self._orthogonalityScorer.score_latent(device=self.device,
                                                              strict=kwargs.get("strict",False),
                                                              hierarchy=kwargs.get("hierarchy",False))
        return ortho_scores

    def score_FID(self, **kwargs):
        """Scoring the model reconstraction quality with FID"""
        num_FID_steps = kwargs.get("num_FID_steps",10)
        generation = kwargs.get('generation',False)
        kwargs = copy.deepcopy(kwargs) # need this to make modifications to the dictionary possible without hurting the rest of the program

        if self.fidscorer is None:
            self.fidscorer = FIDScorer()

        for idx in range(num_FID_steps):
            if idx==0:
                self.fidscorer.start_new_scoring(self.config['data_params']['batch_size']*num_FID_steps, device=self.device)
            inputs, labels = next(iter(self.dataloader.val))
            if generation:
                kwargs.pop('num_samples', None)
                fake_input = self.model.generate(num_samples=inputs.shape[0], activate=True, device=self.device, **kwargs)
            else: fake_input = self.model.reconstruct(inputs.to(self.device), activate=True)
            try: self.fidscorer.get_activations(inputs.to(self.device), fake_input) #store activations for current batch
            except Exception as e:
                print("Reached the end of FID scorer buffer")
                raise(e)

        FID_score = self.fidscorer.calculate_fid()

        return FID_score

    def initialise_consistencyScorer(self, **kwargs):
        """ Looks in the kwargs for the following keys:
            - random_seed
            - num_batches
            - mode
            - verbose
        """

        if self._consistencyScorer is not None:
            return

        random_seed = kwargs.get("random_seed",11)
        num_batches = kwargs.get("num_batches",10)
        mode = kwargs.get("mode", "X")
        verbose = kwargs.get("verbose",True)


        self._consistencyScorer = LatentConsistencyEvaluator(self.model, self.dataloader.val, mode = mode,
                                                             device = self.device, random_seed=random_seed, verbose=verbose)
        # initialising the latent posterior distribution
        self._consistencyScorer.sample_codes_pool(num_batches, self.device)

    def evaluate_invariance(self, **kwargs):
        """ Evaluate invariance of respose map to intervention of the kind specified in the kwargs
        kwargs expected keys:
        - intervention_type: ['noise', ...] - unused for now
        - hard: whether to perform hard or soft intervention (i.e. resample from the dimension or not)
        - num_interventions: number of interventions to base the statistics on
        - num_samples: number of samples for each intervention
        - store_it: SE (self evident)
        - load_it: SE
        - normalise: whether to normalise the score by each unit std dev or not
        + all kwarks for 'initialise_invarianceScorer' function
        """
        intervt_type = kwargs.get("intervention_type", "noise") #TODO: include in the code
        hard = kwargs.get("hard_intervention", False)
        num_interventions = kwargs.get("num_interventions", 50)
        samples_per_intervention = kwargs.get("num_samples",50)
        store_it = kwargs.get("store",False)
        load_it = kwargs.get("load", False)
        normalise = kwargs.get('normalise',True)
        print(f"Scoring model's response map invariance to {intervt_type} interventions.")

        if load_it:
            print("Loading invariances matrix")
            path = Path(self.config['logging_params']['save_dir']) / \
                   self.config['logging_params']['name'] / \
                   self.config['logging_params']['version'] / ("invariances"+".pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return pickle.load(f)
            print("No matrix found at "+str(path))
            print("Calculating invariance from scratch.")

        self.initialise_consistencyScorer(**kwargs)
        D = self.model.latent_size
        U = self.model.unit_dim
        num_units = D//U
        invariances = torch.zeros((num_units,num_units), requires_grad=False).to(self.device)
        std_devs = torch.zeros((2, num_units), requires_grad=False).to(self.device)
        for u in range(num_units):
            print(f"Intervening on {u} ...")
            with torch.no_grad():
                # interventions on unit u
                errors, std_dev = self._consistencyScorer.noise_invariance(unit=u, unit_dim=U, num_units=num_units,
                                                                           num_samples=samples_per_intervention,
                                                                           num_interventions=num_interventions,
                                                                           device=self.device)
                invariances[u,:] = errors
                std_devs += std_dev

        std_devs/=num_units
        if normalise: invariances = invariances/std_devs[1]
        invariances = 1.0 - invariances

        if store_it:
            print("Storing invariance evaluation results.")
            path = Path(self.config['logging_params']['save_dir']) / \
                   self.config['logging_params']['name'] / \
                   self.config['logging_params']['version']
            with open(path/ ("invariances"+".pkl"), 'wb') as o:
                pickle.dump(invariances, o)

        return invariances, std_devs

    def evaluate_self_consistency(self, **kwargs):
        """ Evaluate self_consistency of respose map
        kwargs expected keys:
        - num_samples: number of samples for each intervention
        - normalise: whether to normalise the score by each unit std dev or not
        + all kwarks for 'initialise_consistencyScorer' function
        """

        level = kwargs.get('level', 1)
        print(f"Scoring model's self consistency at level {level}.")
        normalise = kwargs.get('normalise',False)

        self.initialise_consistencyScorer(**kwargs)
        D = self.model.latent_size
        U = self.model.unit_dim
        num_units = D//U
        with torch.no_grad(): errors, std_devs = self._consistencyScorer.self_consistency(U, num_units,
                                                                            device=self.device, **kwargs)

        if normalise: errors = errors/std_devs[1]
        consistency = 1.0 - errors
        return consistency, std_devs

    def evaluate_equivariance(self, **kwargs):
        """ Evaluate equivariance of respose map to intervention of the kind specified in the kwargs
        kwargs expected keys:
        - intervention_type: ['noise', ...] - unused for now
        - hard: whether to perform hard or soft intervention (i.e. resample from the dimension or not)
        - num_interventions: number of interventions to base the statistics on
        - num_samples: number of samples for each intervention
        - store_it: SE (self evident)
        - load_it: SE
        - normalise: whether to normalise the score by each unit std dev or not
        + all kwarks for 'initialise_consistencyScorer' function
        """
        intervt_type = kwargs.get("intervention_type", "noise") #TODO: include in the code
        hard = kwargs.get("hard_intervention", False)
        store_it = kwargs.get("store",False)
        load_it = kwargs.get("load", False)
        print(f"Scoring model's response map equivariance to {intervt_type} interventions.")

        if load_it:
            print("Loading equivariance matrix")
            path = Path(self.config['logging_params']['save_dir']) / \
                   self.config['logging_params']['name'] / \
                   self.config['logging_params']['version'] / ("equivariance"+".pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return pickle.load(f)
            print("No matrix found at "+str(path))
            print("Calculating equivariance from scratch.")

        self.initialise_consistencyScorer(**kwargs)
        D = self.model.latent_size
        U = self.model.xunit_dim
        equivariances = torch.zeros((D,D), requires_grad=False, device=self.device)
        for u in range(D):
            print(f"Intervening on {u} ...")
            with torch.no_grad():
                # interventions on unit u
                errors = self._consistencyScorer.noise_equivariance(unit=u, xunit_dim=U, device=self.device, **kwargs)
                equivariances[u,:] = errors

        equivariances = 1.0 - equivariances

        if store_it:
            print("Storing equivariance evaluation results.")
            path = Path(self.config['logging_params']['save_dir']) / \
                   self.config['logging_params']['name'] / \
                   self.config['logging_params']['version']
            with open(path/ ("equivariance"+".pkl"), 'wb') as o:
                pickle.dump(equivariances, o)

        return equivariances

    def latent_responses(self, **kwargs):
        """Computes latent response matrix for the given model plus handles storage
        (saving and loading) of the same."""

        num_batches = kwargs.get("num_batches",10)
        num_samples = kwargs.get("num_samples",100)
        store_it = kwargs.get("store",False)
        load_it = kwargs.get("load", False)

        if load_it:
            print("Loading latent response matrix")
            path = Path(self.config['logging_params']['save_dir']) / \
                   self.config['logging_params']['name'] / \
                   self.config['logging_params']['version'] / ("responses"+".pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return pickle.load(f)
            print("No matrix found at "+str(path))

        print("Computing latent response matrix")
        #self.send_model_to("cpu")
        matrix = compute_response_matrix(self.dataloader.val, self.model, device=self.device, #we need memory
                                         num_batches=num_batches, num_samples=num_samples)
        self.send_model_to(self.device)

        if store_it:
            print("Storing latent response matrix")
            path = Path(self.config['logging_params']['save_dir']) / \
                   self.config['logging_params']['name'] / \
                   self.config['logging_params']['version'] / ("responses"+".pkl")
            with open(path, 'wb') as o:
                pickle.dump(matrix, o)

        return matrix

    def send_model_to(self, device:str):
        if device=="cpu": self.model.cpu()
        else: self.model.cuda()

    def load_batch(self, train=True, valid=False, test=False):
        if train: loader = self.dataloader.train
        elif valid: loader = self.dataloader.val
        elif test: loader = self.dataloader.test
        else: raise NotImplementedError #TODO: include multiple sets from RFD
        new_batch = next(iter(loader))
        return new_batch

    def list_available_checkpoints(self):
        base_path = Path(self.config['logging_params']['save_dir']) / \
                    self.config['logging_params']['name'] / \
                    self.config['logging_params']['version']
        checkpoint_path =  base_path / "checkpoints/"
        checkpoints = glob.glob(str(checkpoint_path) + "/*ckpt")
        print("Available checkpoints at "+ str(checkpoint_path)+" :")
        print(checkpoints)
        return checkpoints

    def score_causal_vars_entropy(self, **kwargs):
        """Scores entropy on each causal dimension
        Note: only applicable to Xnets"""
        entropy = vis_xnets.compute_renyis_entropy_X(self.model, iter(self.dataloader.test), self.device, **kwargs)
        return entropy

    def load_checkpoint(self, name=None):
        #TODO: together with checkpoint load some training status file, old results etc
        base_path = Path(self.config['logging_params']['save_dir']) / \
                    self.config['logging_params']['name'] / \
                    self.config['logging_params']['version']
        checkpoint_path =  base_path / "checkpoints/"
        hparams_path = str(base_path)+"/hparams.yaml"
        actual_checkpoint_path = ""

        try:
            if name is None:
                actual_checkpoint_path = max(glob.glob(str(checkpoint_path) + "/*ckpt"), key=os.path.getctime)
                print("Loading latest checkpoint at "+actual_checkpoint_path+" .")
            else:
                actual_checkpoint_path = str(checkpoint_path) +"/"+ name + ".ckpt"
                print("Loading selected checkpoint at "+ actual_checkpoint_path)

            self.experiment = self.experiment.load_from_checkpoint(actual_checkpoint_path,
                                                                   hparams_file=hparams_path,
                                                                   device=self.device,
                                                                   strict=False,
                                                                   params=self.config)
            self.model = self.experiment.model
            self.model.eval()
            self.device = next(self.model.parameters()).device
            self.send_model_to(self.device)
        except ValueError:
            print(f"No checkpoint available at "+str(checkpoint_path)+". Cannot load trained weights.")

    def score_model(self, FID=True, disentanglement=True, orthogonality=True, invariance=True,
                    save_scores=True, update_general_scores=True, equivariance=True, self_consistency=True, **kwargs):
        """Scores the model on the test set in loss and other terms selected
        #TODO: add here the saving of the scores to the 'scores.csv' file that has the scores of all the models"""
        self.device = next(self.model.parameters()).device
        self.send_model_to(self.device)


        if update_general_scores:
            #initialising the update
            df = pd.DataFrame()
            df['model_name'] = [self.config['logging_params']['name']]
            df['model_version'] = [self.config['logging_params']['version']]
            df['dataset'] = [self.config['data_params']['dataset_name']]
            df['random_seed'] = [self.config['model_params']['random_seed']]

        # 1. collect all the scores available
        start=time.time()
        scores = {}
        if orthogonality:
            ortho_scores = self.score_orthogonality(**kwargs)
            scores.update(ortho_scores)
        if disentanglement:
            disentanglement_scores = self.score_disentanglement(**kwargs)
            scores.update(disentanglement_scores)
        if FID:
            scores["FID_rec"] = self.score_FID(generation=False, **kwargs)
            scores["FID_gen"] = self.score_FID(generation=True, **kwargs)
        if invariance:
            invariances, _ = self.evaluate_invariance(**kwargs)
            scores["INV"] = torch.sum(invariances*(1-torch.eye(self.model.latent_size, device=self.device))).cpu().numpy() #TODO: correct to take into account ignored dimensions (maybe divide by posterior sd?)
        if equivariance:
            equivariances = self.evaluate_equivariance(**kwargs)
            scores["EQV"] = torch.sum(equivariances).cpu().numpy()
        if self_consistency:
            consistencies, _ = self.evaluate_self_consistency(**kwargs)
            scores["SCN"] = torch.sum(consistencies).cpu().numpy()
        end = time.time()

        print("Time elapsed for scoring {:.0f}".format(end-start))

        # save complete scores to a binary file
        if save_scores:
            name = kwargs.get("name","scoring")
            path = Path(self.config['logging_params']['save_dir']) / \
                        self.config['logging_params']['name'] / \
                        self.config['logging_params']['version'] / (name+".pkl")
            with open(path, 'wb') as o:
                pickle.dump(scores, o)

        #update general scores (contained in .csv file)
        if update_general_scores:
            for k,v in scores.items():
                df.loc[0,k] = v
            full_df_path = self.config['logging_params']['save_dir'] + "/" + "all_scores.csv"  #read csv into pd.dataframe
            try: full_df = pd.read_csv(full_df_path, index_col=0)
            except FileNotFoundError:
                print("Full scores dataframe not found. Starting a new one")
                full_df = pd.DataFrame() # empty initialisation
            full_df = pd.concat([full_df, df], ignore_index=True) #add one row to the csv
            full_df.drop_duplicates(subset = ["model_name","model_version","dataset","random_seed"],
                                    keep="last", inplace=True, ignore_index=True)
            full_df.to_csv(full_df_path) #save it again

        return scores

    def load_scores(self, **kwargs):
        """Return saved scores dictionary if any"""

        name = kwargs.get("name","scoring")
        path = Path(self.config['logging_params']['save_dir']) / \
                    self.config['logging_params']['name'] / \
                    self.config['logging_params']['version'] / (name+".pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)


        print("No scores file found at "+str(path))


class VisualModelHandler(ModelHandler):
    """Offers a series of tools to inspect a given model."""
    def plot_model(self, do_originals=False, do_reconstructions=False,
                   do_random_samples=False, do_traversals=False, do_hybrisation=False,
                   do_loss2distortion=False, do_marginal=False, do_loss2marginal=False,
                   do_invariance=False, do_latent_block=False, do_traversal_responses=False,
                   do_latent_response_field=False, do_equivariance=False,
                   do_jointXmarginals=False, do_hybridsX=False, do_unitMarginal=False,
                   do_marginalX=False, do_latent_response_fieldX=False, do_N2X=False,
                   do_double_hybridsXN=False, do_interpolationN=False, **kwargs):

        plots = {}
        if self.visualiser is None:
            self.visualiser = ModelVisualiser(self.model, self.dataloader.test, **self.config["vis_params"])
        if do_reconstructions:
            plots["reconstructions"] = self.visualiser.plot_reconstructions(device=self.device, **kwargs)
        if do_random_samples:
            try: plots["random_samples"] = self.visualiser.plot_samples_from_prior(device=self.device, **kwargs)
            except ValueError:pass #no prior samples stored yet
        if do_traversals:
            plots["traversals"] = self.visualiser.plot_latent_traversals(device=self.device, tailored=True, **kwargs)
        if do_originals: # print the originals
            plots["originals"] = self.visualiser.plot_originals()
        if do_hybrisation: # print the originals
            print(f"Plotting hybridisation on the noise variables")
            hybrids, _, _  = vis_latents.hybridiseN(self.model, self.device, **kwargs)
            plots["hybridsN"] = self.visualiser.plot_grid(hybrids, nrow=3, **kwargs)

        if do_loss2distortion:
            plots["distortion"] = self.visualiser.plot_loss2distortion(device=self.device, **kwargs)
        if do_marginal:
            codes = vis_latents.get_posterior(self.model, iter(self.dataloader.test), self.device, **kwargs)
            plots["marginal"] = self.visualiser.plot_marginal(codes, device=self.device, **kwargs)
        if do_loss2marginal:
            plots["marginal_distortion"] = self.visualiser.plot_loss2marginaldistortion(device=self.device, **kwargs)
        if do_invariance:
            invariances, std_devs = self.evaluate_invariance(**kwargs)
            plots["invariances"] = self.visualiser.plot_heatmap(invariances.cpu().numpy(),
                                                                title="Invariances", threshold=0., **kwargs)
            plots["std_devs"] = self.visualiser.plot_heatmap(std_devs.cpu().numpy(),
                                                             title="Standard Deviation of marginals (original and responses)",
                                                             threshold=0., **kwargs)
        if do_equivariance:
            equivariances = self.evaluate_equivariance(**kwargs)
            plots["equivariances"] = self.visualiser.plot_heatmap(equivariances.cpu().numpy(), title="Equivariances",
                                                                  threshold=0., **kwargs)

        if do_traversal_responses:
            all_traversal_latents, all_traversals_responses, traversals_steps = \
                vis_responses.traversal_responses(self.model, self.device, **kwargs)
            plots["trvs_responses"] = []
            D = self.model.latent_size
            for d in range(D):
                fig = self.visualiser.plot_traversal_responses(d, all_traversal_latents[d], all_traversals_responses[d],
                                                               traversals_steps=traversals_steps, **kwargs)
                plots["trvs_responses"].append(fig)


        if do_latent_block:
            # plotting latent block adjacency matrix
            A = get_causal_block_graph(self.model, self.model_name, self.device, **kwargs)
            plots['causal_block_graph'] = self.visualiser.plot_heatmap(A.cpu().numpy(),
                                                                       title="Causal block adjacency matrix",
                                                                       threshold=10e-1, **kwargs)

        if do_latent_response_field:
            i = kwargs.pop("i",0)
            j = kwargs.pop("j", 1)
            response_field, hybrid_grid = vis_responses.response_field(i, j, self.model, self.device, **kwargs)
            X, Y = hybrid_grid
            plots["latent_response_field"] = self.visualiser.plot_vector_field(response_field, X, Y, i=i, j=j, **kwargs)

        if do_jointXmarginals:
            i = kwargs.pop("i",0)
            j = kwargs.pop("j", 1)
            print(f"Plotting X{i}-X{j} joint marginal")
            Xij, hue = vis_xnets.compute_joint_ij(i, j, self.model, self.device, **kwargs)
            Xij = Xij.detach().cpu().numpy()
            plots["Xij"] = self.visualiser.scatterplot_with_line(Xij[:,0], Xij[:,1], hue,
                                                                 x_name=f"X{j}", y_name=f"X{i}", **kwargs)

        if do_hybridsX:
            print(f"Plotting hybridisation on the causal variables")
            hybrids, _, _  = vis_xnets.hybridiseX(self.model, self.device, **kwargs)
            plots["hybridsX"] = self.visualiser.plot_grid(hybrids, nrow=3, **kwargs)

        if do_double_hybridsXN:
            print(f"Plotting double hybridisation on the noise and causal variables")
            hybrids_N, _samples  = vis_latents.double_hybridiseN(self.model, self.device, **kwargs)
            hybrids_X  = vis_xnets.double_hybridiseX(self.model, self.device, prior_samples=_samples, **kwargs)
            plots["hybridsN"] = self.visualiser.plot_grid(hybrids_N, nrow=self.model.latent_size+1, **kwargs)
            plots["hybridsX"] = self.visualiser.plot_grid(hybrids_X, nrow=self.model.latent_size+1, **kwargs)


        if do_interpolationN:
            print(f"Plotting interpolation between random sample from the posterior on the noises")
            interpolation  = vis_latents.interpolate(self.model, iter(self.dataloader.test), self.device, **kwargs)
            plots["interpolationN"] = self.visualiser.plot_grid(interpolation, nrow=interpolation.shape[0], **kwargs)

        if do_unitMarginal:
            u = kwargs.get("unit",0)
            print(f"Plotting marginal of multidim unit {u}")
            samples_ux2D = vis_xnets.multidimUnitMarginal(self.model, device=self.device, **kwargs)
            plots[f"unit{u}"] = self.visualiser.kdeplot(samples_ux2D[:,0], samples_ux2D[:,1], **kwargs)

        if do_marginalX: # note: only works for unidimensional Xnets
            Xs = vis_xnets.get_posterior(self.model, iter(self.dataloader.test), self.device, **kwargs)
            plots["marginal"] = self.visualiser.plot_marginal(Xs, device=self.device, **kwargs)


        if do_latent_response_fieldX:
            i = kwargs.pop("i",0)
            j = kwargs.pop("j", 1)
            response_field, hybrid_grid = vis_responses.response_fieldX(i, j, self.model, self.device, **kwargs)
            X, Y = hybrid_grid
            plots["latent_response_fieldX"] = self.visualiser.plot_vector_field(response_field, X, Y, i=i, j=j, **kwargs)

        if do_N2X:
            dim = kwargs.pop("dim",0)
            print(f"Plotting N{dim}-X{dim} joint marginal")
            NX, hue = vis_xnets.compute_N2X(dim, self.model, self.device, **kwargs)
            NX = NX.detach().cpu().numpy()  # shape (MxN) x (1 + xunit_dim) - the first is the noise dimension, the others are Xs
            ndims = NX.shape[1] - 1
            for xd in range(ndims):
                plots["Xij"] = self.visualiser.scatterplot_with_line(NX[:,0], NX[:,xd], hue,
                                        x_name=f"N{dim}", y_name=f"X{dim}-{xd}", **kwargs)



        return plots


class VectorModelHandler(ModelHandler):
    """Offers a series of tools to inspect a given model."""
    def __init__(self, experiment:pl.LightningModule, **kwargs):
        super().__init__(experiment, **kwargs)
        self.model_visualiser = None
        self.data_visualiser = None

    def switch_labels_to_noises(self):
        print("Loading noises as labels.")
        self.config["data_params"]["noise"] = True
        self.dataloader = DatasetLoader(self.config["data_params"])

    def plot_data(self):
        plots = {}
        if self.data_visualiser is None:
            self.data_visualiser = SynthVecDataVisualiser(self.dataloader.test)

        plots["graph"] = self.data_visualiser.plot_graph()
        plots["noises"] = self.data_visualiser.plot_noises_distributions()
        plots["causes2noises"] = self.data_visualiser.plot_causes2noises()
        return plots

    def plot_model(self, **kwargs):

        plots = {}
        if self.model_visualiser is None:
            self.model_visualiser = ModelVisualiser(self.model,
                                                    self.dataloader.test)

        plots["distortion"] = self.visualiser.plot_loss2distortion(device=self.device, **kwargs)
        plots["marginal"] = self.visualiser.plot_marginal(device=self.device, **kwargs)
        plots["marginal_distortion"] = self.visualiser.plot_loss2marginaldistortion(device=self.device, **kwargs)
        invariances = self.evaluate_invariance(load_it = True, store_it=True, **kwargs)
        plots["invariances"] = self.visualiser.plot_heatmap(invariances.cpu().numpy(), title="Invariances", **kwargs)

        return plots



if __name__ == '__main__':

    params = {"model_name":"XCAE",
              "model_version":"standardS",
              "data" : "MNIST"}

    # load handler
    handler = VisualModelHandler.from_config(**params)
    handler.config["logging_params"]["save_dir"] = "./logs"
    handler.load_checkpoint()
    handler.reset_tau(100000)
    figure_params = {"figsize":(9,7)}
    for w in range(1): res = handler.plot_model(do_reconstructions=True, **figure_params)
    for i in range(6):
        other_args = {"dim":i, "num_samples":100, "marginal_samples":100,
                      "prior_mode":"uniform", "with_line":True}
        res = handler.plot_model(do_reconstructions=False, do_N2X=True,
                             **figure_params, **other_args)
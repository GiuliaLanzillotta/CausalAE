"""Script managing evaluation and exploration of models"""
import pickle

import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.optim import lr_scheduler

from experiments import pick_model_manager, get_causal_block_graph
from pathlib import Path
from configs import get_config
from experiments.data import DatasetLoader
import os
import glob
import time
import copy
import math



from experiments.utils import temperature_exponential_annealing
from visualisations import ModelVisualiser, SynthVecDataVisualiser, vis_responses, vis_xnets, vis_latents
from metrics import FIDScorer, ModelDisentanglementEvaluator, LatentOrthogonalityEvaluator, LatentConsistencyEvaluator, inference
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
        print(model_name+ " model handler loaded.")
        self.visualiser = None
        self.fidscorer = None
        self._orthogonalityScorer = None
        self._disentanglementScorer = None
        self._consistencyScorer = None
        self._inet = None
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
        """Scoring the model reconstraction quality with FID
        kwargs accepted keys:
        -num_FID_steps
        -kind: str \in ['reconstruction', 'generation', 'traversals']
        """
        num_FID_steps = kwargs.get("num_FID_steps",10)
        kind = kwargs.get('kind','reconstruction')
        # adjusting FID steps based on input size
        factor = math.ceil(self.config['data_params']['size']/32)
        num_FID_steps = math.ceil(num_FID_steps/factor)
        print(f"FID scoring set with {num_FID_steps} steps")

        kwargs = copy.deepcopy(kwargs) # need this to make modifications to the dictionary possible without hurting the rest of the program

        if self.fidscorer is None:
            self.fidscorer = FIDScorer()

        for idx in range(num_FID_steps):
            if idx==0:
                self.fidscorer.start_new_scoring(self.config['data_params']['batch_size']*num_FID_steps, device=self.device)
            inputs, labels = next(iter(self.dataloader.val))
            if kind == 'generation':
                kwargs.pop('num_samples', None)
                with torch.no_grad():
                    fake_input = self.model.generate(num_samples=inputs.shape[0], activate=True, device=self.device, **kwargs)
            elif kind == 'traversals':
                num_samples = inputs.shape[0]//10 + 1
                with torch.no_grad():
                    fake_input = vis_latents.traversals(self.model, self.device, inputs=inputs, num_samples=num_samples, num_steps=10)
                    fake_input = torch.cat(fake_input, dim=0).to(self.device)
                assert fake_input.shape[0] >= inputs.shape[0], "Not enough fake inputs collected"
                fake_input = fake_input[:inputs.shape[0]]
            else:
                with torch.no_grad():
                    fake_input = self.model.reconstruct(inputs.to(self.device), activate=True)
            try: self.fidscorer.get_activations(inputs.to(self.device), fake_input) #store activations for current batch
            except RuntimeError as e:
                raise e

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

    def score_SCN_under_perturbation(self, **kwargs):
        """Evaluates the model's self consistency under multiple modes of perturbation"""
        consistencies = {}
        for prior_mode in ["posterior","hybrid","uniform"]:
            params = {'num_samples': kwargs.get('num_samples',1000),
                      'random_seed': kwargs.get('random_seed',23),
                      'num_batches': kwargs.get('num_batches',10),
                      'prior_mode': prior_mode,
                      'normalise': True,
                      'verbose': kwargs.get("verbose",False),
                      'level': 1}
            with torch.no_grad():
                consistency, std_dev = self.evaluate_self_consistency(**params)
                consistencies["SCN_"+prior_mode] = torch.clamp(consistency, 0.0, 1).mean().cpu().numpy().item() # average across dimensions
        return consistencies


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
        normalise = kwargs.get('normalise',False)
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
                errors, std_dev = self._consistencyScorer.noise_equivariance(unit=u, xunit_dim=U, device=self.device, **kwargs)
                equivariances[u,:] = errors

        if normalise: equivariances /= std_dev
        equivariances = 1.0 - equivariances

        if store_it:
            print("Storing equivariance evaluation results.")
            path = Path(self.config['logging_params']['save_dir']) / \
                   self.config['logging_params']['name'] / \
                   self.config['logging_params']['version']
            with open(path/ ("equivariance"+".pkl"), 'wb') as o:
                pickle.dump(equivariances, o)

        return equivariances, std_dev

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

    def evaluate_inference(self, **kwargs):
        """ Evaluates trained representation on inference task using gradient boosting"""
        causal = kwargs.get('causal',False)
        def representation(inputs): # representation function to be used
            return self.model.get_representation(inputs, causal=causal).detach().clone()
        importance_matrix, train_acc, test_acc = inference.do_inference(self.dataloader.train,
                                                                        self.dataloader.test,
                                                                        representation,
                                                                        self.device,
                                                                        num_train=kwargs.get('num_train',20000),
                                                                        num_test=kwargs.get('num_test',10000),
                                                                        batch_size=self.config['data_params']['batch_size'])
        return importance_matrix, train_acc, test_acc

    def load_batch(self, train=True, valid=False, test=False):
        if train: loader = self.dataloader.train
        elif valid: loader = self.dataloader.val
        elif test: loader = self.dataloader.test
        else: raise NotImplementedError #TODO: include multiple sets from RFD
        new_batch = next(iter(loader))
        return new_batch

    @property
    def root(self):
        base_path = Path(self.config['logging_params']['save_dir']) / \
                    self.config['logging_params']['name'] / \
                    self.config['logging_params']['version']
        return base_path

    def list_available_checkpoints(self):
        checkpoint_path =  self.root / "checkpoints/"
        checkpoints = glob.glob(str(checkpoint_path) + "/*ckpt")
        print("Available checkpoints at "+ str(checkpoint_path)+" :")
        print(checkpoints)
        return checkpoints

    def score_causal_vars_entropy(self, **kwargs):
        """Scores entropy on each causal dimension
        Note: only applicable to Xnets"""
        entropy = vis_xnets.compute_renyis_entropy_X(self.model, iter(self.dataloader.test), self.device, **kwargs)
        return entropy

    def score_classification_on_responses(self, **kwargs):
        """Obtains classification score on responses"""
        scores, outs, Ys, Y_hats, preds =  vis_responses.classification_on_responses(self.model, self.device, **kwargs)
        scores = np.asarray(scores).mean(axis=0)
        return scores

    def score_DCI_on_response(self, **kwargs):
        """Computes DCI score on traversal responses."""
        return vis_responses.DCI_on_responses(self.model, self.device, **kwargs)


    def load_checkpoint(self, name=None):
        #TODO: together with checkpoint load some training status file, old results etc
        checkpoint_path =  self.root / "checkpoints/"
        hparams_path = str(self.root)+"/hparams.yaml"
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
                    save_scores=True, update_general_scores=True, equivariance=True,
                    self_consistency=True, inference=True, sparsity=True,
                    response_classification=False, DCI_response=True, SCN_perturbed=True, **kwargs):
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
            scores["FID_rec"] = self.score_FID(kind="reconstruction", **kwargs)
            scores["FID_gen"] = self.score_FID(kind="generation", **kwargs)
            scores["FID_trv"] = self.score_FID(kind="traversals", **kwargs)

        if invariance:
            invariances, std_devs = self.evaluate_invariance(**kwargs)
            weighted = kwargs.get('weighted',True) # wheter to use weights depending on the standard deviations in the sum
            W = 1-torch.eye(self.model.latent_size, device=self.device); Z = W.sum()
            scores["INV"] = (torch.sum(invariances*W)/Z).cpu().numpy()
            if weighted:
                W = W*(std_devs[1,:].view(-1,1))
                Z = W.sum().cpu().numpy()
                scores["INV_w"] = torch.sum(invariances*W).cpu().numpy()/Z

        if equivariance and 'X' in self.model_name:
            weighted = kwargs.get('weighted',True) # wheter to use weights depending on the standard deviations in the sum
            equivariances, std_devs = self.evaluate_equivariance(**kwargs)
            W = torch.ones(self.model.latent_size, self.model.latent_size, device=self.device); Z = W.sum()
            scores["EQV"] = (torch.sum(equivariances*W)/Z).cpu().numpy()
            if weighted:
                W = W*(std_devs.view(-1,1))
                Z = W.sum().cpu().numpy()
                scores["EQV_w"] = torch.sum(equivariances*W).cpu().numpy()/Z

        if sparsity and 'X' in self.model_name:
            # plotting latent block adjacency matrix
            A = get_causal_block_graph(self.model, self.model_name, self.device, tau=100000)
            scores['sparsity'] = (A.sum()/(self.model.latent_size**2)).cpu().numpy()

        if self_consistency:
            weighted = kwargs.get('weighted',True) # wheter to use weights depending on the standard deviations in the sum
            multi_level = kwargs.get('multi_level',True)
            if multi_level:
                kwargs["level"] = 10
            consistencies, std_devs = self.evaluate_self_consistency(**kwargs)
            W = torch.ones(self.model.latent_size, device=self.device); Z = W.sum()
            scores["SCN"] = (torch.sum(consistencies[0,:]*W)/Z).cpu().numpy() #first level
            if multi_level:
                scores["SCN+"] = (torch.sum(consistencies[4,:]*W)/Z).cpu().numpy() #last level
                scores["SCN++"] = (torch.sum(consistencies[-1,:]*W)/Z).cpu().numpy() #last level
            if weighted:
                W = W*(std_devs[1,:].view(-1))
                Z = W.sum().cpu().numpy()
                scores["SCN_w"] = torch.sum(consistencies[0,:]*W).cpu().numpy()/Z

        if inference:
            _, _, inference = self.evaluate_inference(**kwargs)
            scores["inference"] = np.mean(inference)
            if 'X' in self.model_name:
                _, _, inferenceX = self.evaluate_inference(causal=True, **kwargs)
                scores["inferenceX"] = np.mean(inferenceX)

        if response_classification:
            class_scores = self.score_classification_on_responses(**kwargs)
            # maybe we can do a weighted average and exclude collapsed dimensions
            scores["PLC"] = class_scores[0] #prior latent classification
            scores["RLC"] = class_scores[1] #response latent classification

        if DCI_response:
            d, c, importance_matrix = self.score_DCI_on_response(**kwargs)
            scores["DCIR_dis"] = d
            scores["DCIR_cmplt"] = c

        if SCN_perturbed:
            SCN_perturbed_scores = self.score_SCN_under_perturbation(**kwargs)
            scores.update(SCN_perturbed_scores)


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
            print("Updated general scores.")

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
                   do_double_hybridsXN=False, do_interpolationN=False, do_traversalsX=False,
                   do_traversalsNX=False, do_inference=False, do_multiple_traversals=False,
                   do_CausalVAE_interventions=False, do_intervention_res=False,
                   do_classification_on_responses=False, **kwargs):

        plots = {}
        if self.visualiser is None:
            self.visualiser = ModelVisualiser(self.model, self.dataloader.test, **self.config["vis_params"])
        if do_reconstructions:
            plots["reconstructions"] = self.visualiser.plot_reconstructions(device=self.device, **kwargs)
        if do_random_samples:
            try: plots["random_samples"] = self.visualiser.plot_samples_from_prior(device=self.device, **kwargs)
            except ValueError:pass #no prior samples stored yet
        if do_traversals:
            traversals_rec = vis_latents.traversals(self.model, self.device, **kwargs)
            plots["traversals"] = self.visualiser.plot_grid(torch.cat(traversals_rec, dim=0).to(self.device),
                                                            nrow=kwargs.get('steps'), **kwargs)
        if do_intervention_res:
            plots["intervention_res"] = self.visualiser.plot_results_intervention(self.device, **kwargs)

        if do_traversalsX:
            #notice: only works if argument 'dim' is provided to kwargs
            dims = kwargs.get('dims')
            all = []
            for d in dims:
                traversals_rec = vis_xnets.traversalsX(self.model, self.device, dim=d, **kwargs)
                all.append(traversals_rec)
            plots["traversalsX"] = self.visualiser.plot_grid(torch.cat(all, dim=0).to(self.device),
                                                                 nrow=kwargs.get('steps',20), **kwargs)
        if do_traversalsNX:
            # plot traversals on N alongside traversals on X for a given dimension (which has to be provided in kwargs)
            dim = kwargs.get('dim')
            traversalsN = vis_latents.traversals(self.model, self.device, **kwargs)[dim]
            traversalsX = vis_xnets.traversalsX(self.model, self.device, **kwargs)
            plots["traversalsNX"] = self.visualiser.plot_grid(torch.cat([traversalsN, traversalsX], dim=0).to(self.device),
                                                             nrow=kwargs.get('steps', 20), **kwargs)


        if do_originals: # print the originals
            plots["originals"] = self.visualiser.plot_originals()
        if do_hybrisation: # print the originals
            print(f"Plotting hybridisation on the noise variables")
            hybrids, _, _  = vis_latents.hybridiseN(self.model, self.device, **kwargs)
            plots["hybridsN"] = self.visualiser.plot_grid(hybrids, nrow=3, **kwargs)
        if do_CausalVAE_interventions:
            assert self.config['logging_params']['name']=="CausalVAE", "CausalVAE interventions only apply to CausalVAE models"
            print(f"Plotting effect of random interventions on the causal variables")
            self.visualiser.plot_causalVAE_interventions(device=self.device, **kwargs)

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
            equivariances, _ = self.evaluate_equivariance(**kwargs)
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

        if do_multiple_traversals:
            # list of reconstructions for each dimension
            all_traversals_recs = vis_latents.traversals(self.model, self.device, **kwargs)
            plots["multi_traversals"] = []
            D = self.model.latent_size
            for d in range(D):
                fig = self.visualiser.plot_grid(all_traversals_recs[d], nrow=kwargs.get('steps'), **kwargs)
                plots["multi_traversals"].append(fig)

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
            dimN = kwargs.pop("dimN",0)
            dimX = kwargs.pop("dimX",0)
            print(f"Plotting N{dimN}-X{dimX} joint marginal")
            NX, hue = vis_xnets.compute_N2X(dimN, dimX,  self.model, self.device, **kwargs)
            NX = NX.detach().cpu().numpy()  # shape (MxN) x (1 + xunit_dim) - the first is the noise dimension, the others are Xs
            ndims = NX.shape[1] - 1
            res = []
            for xd in range(ndims):
                res.append(self.visualiser.scatterplot_with_line(NX[:,0], NX[:,1+xd], hue,
                                        x_name=f"N{dimN}", y_name=f"X{dimX}-{xd}", legend=False, **kwargs))
            plots["Xij"] = res

        if do_inference:
            importance_matrix, train_acc, test_acc = self.evaluate_inference(**kwargs)
            plots["inference_imp"] = self.visualiser.plot_heatmap(torch.Tensor(importance_matrix), title="Importance matrix for inference", threshold=0., **kwargs)
            plots["inference_acc"] = self.visualiser.plot_heatmap(torch.Tensor(test_acc), title="Average accuracies for inference", threshold=0., **kwargs)

        if do_classification_on_responses:
            scores, outs, Ys, Y_hats = vis_responses.classification_on_responses(self.model, self.device, **kwargs)
            #TODO: complete here


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
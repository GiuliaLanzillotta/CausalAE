""" All functions needed to compute the material to visualise the latent responses"""
import numpy as np
import torch
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from metrics import DCI
from models import GenerativeAE, Xnet
from . import utils


def traversal_responses(model:GenerativeAE, device, **kwargs):
    """Computes the amount of distortion recorded on each latent dimension while traversing a single dimension
    kwargs accepted keywords:
        - num_samples: number of samples from the latent space to use in the computation
        - steps: number of steps to take in the traversal
        - all arguments accepted in 'sample_noise_from_prior'

    Returns 2 lists containing the latents and corresponding responses for each latent unit
    """
    print("Computing traversal responses ... ")
    if not kwargs.get('num_samples'):
        kwargs['num_samples'] = 50
    steps = kwargs.get('steps',20)
    relative = kwargs.get('relative',False)
    unit_dim = 1

    all_traversal_latents = []
    all_traversals_responses = []
    # sample N vectors from prior
    prior_samples = model.sample_noise_from_prior(device=device, **kwargs).detach()
    ranges = model.get_prior_range()
    # for each latent unit we start traversal
    # 1. obtain traversals values
    traversals_steps = utils.get_traversals_steps(steps, ranges, relative=relative).to(device).detach() #torch Tensor
    for d in range(model.latent_size):
        with torch.no_grad():
            # 2. do traversals
            traversals = utils.do_latent_traversals_multi_vec(prior_samples, unit_dim=unit_dim,
                                                              unit=d, values=traversals_steps[d],
                                                              device=device, relative=relative) # shape steps x N x D
            traversals_latents = traversals.view(-1, model.latent_size) # reshaping to fit into batch
            # 3. obtain responses
            trvs_response = model.encode_mu(model.decode(traversals_latents, activate=True))
            all_traversal_latents.append(traversals)
            all_traversals_responses.append(trvs_response.view(steps, -1, model.latent_size))

    print("...done")

    return all_traversal_latents, all_traversals_responses, traversals_steps

def hybrid_responses():
    #TODO
    pass

def response_field(i:int, j:int, model:GenerativeAE, device, **kwargs):
    """Evaluates the response field over the selected latent dimensions (i and j are the indices)
    kwargs accepted keys:
    - grid_size
    - num_samples
    - all kwargs accepted in 'sample noise from prior'
    #TODO: add support for multidimensional units

    Returns tensor of size (grid_size**2, 2) with the mean responses over the grid and the grid used
    """

    print(f"Computing response field for {i} and {j} ... ")
    grid_size = kwargs.get('grid_size', 20) #400 points by default
    num_samples = kwargs.get('num_samples',50)
    unit_dim = 1
    num_units = model.latent_size//unit_dim
    # sample N vectors from prior
    prior_samples = model.sample_noise_from_prior(device=device, **kwargs).detach()
    ranges = model.get_prior_range()
    with torch.no_grad():
        hybrid_grid = torch.meshgrid([torch.linspace(ranges[i][0],ranges[i][1], steps=grid_size),
                                      torch.linspace(ranges[j][0],ranges[j][1], steps=grid_size)])
        i_values = hybrid_grid[0].contiguous().view(-1, unit_dim) # M x u
        j_values = hybrid_grid[1].contiguous().view(-1, unit_dim) # M x u
        assert i_values.shape[0] == grid_size**2, "Something wrong detected with meshgrid"
        # now for each of the prior samples we want to evaluate the full grid in order to then average the results  (mean field approximation)
        all_samples = torch.tile(prior_samples, (grid_size**2,1,1)) #shape = M x N x D (M is grid_size**2)
        all_samples[:,:,i] = i_values.repeat(1,num_samples)
        all_samples[:,:,j] = j_values.repeat(1,num_samples)
        responses = model.encode_mu(model.decode(all_samples.view(-1,num_units), activate=True))
        response_field = torch.hstack([responses[:,i], responses[:,j]]).view(grid_size**2, num_samples, 2).mean(dim=1) # M x 2

    return response_field, hybrid_grid


def response_fieldX(i:int, j:int, model:Xnet, device, **kwargs):
    """Evaluates the response field over the selected causal units (i and j are the indices)
    kwargs accepted keys:
    - grid_size
    - num_samples
    - all kwargs accepted in 'sample noise from prior'
    Returns tensor of size (grid_size**2, 2) with the mean responses over the grid and the grid used
    """

    print(f"Computing response field for X{i} and X{j} ... ")
    grid_size = kwargs.get('grid_size', 20) #400 points by default
    num_samples = kwargs.get('num_samples',50)
    range_limit = kwargs.get('range_limit',3)
    unit_dim = model.xunit_dim
    assert unit_dim==1, "Only one-dimensional units are supported"
    num_units = model.latent_size
    # sample N vectors from prior and compute causal variables
    prior_samples = model.sample_noise_from_prior(device=device, **kwargs).detach()
    Xs = model.get_causal_variables(prior_samples, **kwargs).detach()
    with torch.no_grad():
        hybrid_grid = torch.meshgrid([torch.linspace(-range_limit,range_limit, steps=grid_size),
                                      torch.linspace(-range_limit,range_limit, steps=grid_size)])
        i_values = hybrid_grid[0].contiguous().view(-1, unit_dim) # M x u
        j_values = hybrid_grid[1].contiguous().view(-1, unit_dim) # M x u
        assert i_values.shape[0] == grid_size**2, "Something wrong detected with meshgrid"
        # now for each of the prior samples we want to evaluate the full grid in order to then average the results  (mean field approximation)
        all_samples = torch.tile(Xs, (grid_size**2,1,1)) #shape = M x N x D (M is grid_size**2)
        all_samples[:,:,i] = i_values.repeat(1,num_samples)
        all_samples[:,:,j] = j_values.repeat(1,num_samples)
        responses = model.encode_mu(model.decode_from_X(all_samples.view(-1,num_units), activate=True).detach()).detach()
        X_responses = model.get_causal_variables(responses, **kwargs).detach()
        response_field = torch.hstack([X_responses[:,i], X_responses[:,j]]).view(grid_size**2, num_samples, 2).mean(dim=1) # M x 2

    return response_field, hybrid_grid


def get_latent_classification_examples(model:GenerativeAE, device, **kwargs):
    """Collects examples for classification problem on the latent dimensions.
    An example consists of a pair (images, label), where """
    #TRY https://gist.github.com/s7ev3n/5717df957e61ce2fa3600368cd7724c9

def DCI_on_responses(model:GenerativeAE, device, **kwargs):
    """Implements DCI metric test on response map

    kwargs accepted keywords:
        - num_samples: number of samples from the latent space to use in the computation
        - steps: number of steps to take in the traversal
        - bins: number of bins to use for categorisation of the latent variables
        - random_state: random seed used for classifier training
        - all arguments accepted in 'sample_noise_from_prior'
    """
    # we need to obtain bins for all variables first
    print("Computing DCI test on responses ... ")
    if not kwargs.get('num_samples'):
        kwargs['num_samples'] = 50
    steps = kwargs.get('steps', 40)
    bins = kwargs.get('bins', 5)
    verbose = kwargs.get('verbose', False)
    if hasattr(model, "unit_dim"):
        unit_dim = model.unit_dim
    else: unit_dim = 1 #TODO: make computation work for multi-dim models

    # sample N vectors from prior
    prior_samples = model.sample_noise_from_prior(device=device, **kwargs).detach()
    ranges = model.get_prior_range()
    # for each latent unit we start traversal
    # 1. obtain traversals values (for all the dimensions)
    traversals_steps = utils.get_traversals_steps(steps, ranges, relative=False).to(device).detach()  # torch Tensor
    # 2. obtain categories for each latent dimension
    class_boundaries = utils.get_class_boundaries(bins, ranges).detach().cpu().numpy()  # D x bins tensor
    # now we need to collect our dataset by applying traversals to each dimension for all of our samples
    # and then extracting the corresponding label
    Xs = []; Ys = []
    for d in range(model.latent_size):
        if verbose: print(f"Collecting dataset samples from dimension {d}")
        with torch.no_grad():
            # 3. do traversals on dimension d
            traversals = utils.do_latent_traversals_multi_vec(prior_samples, unit_dim=unit_dim,
                                                              unit=d, values=traversals_steps[d],
                                                              device=device, relative=False) # shape steps x N x D
            X = traversals.reshape(-1,model.latent_size).cpu().numpy() # now we have (stepsxN) latent vectors
            np.random.shuffle(X)
            out = model.decode(torch.Tensor(X).to(device), activate=True)
            X_hat = model.encode_mu(out).cpu().numpy()
            Xs.append(X_hat)
            # 4. obtain the labels associated to each vector using the bins: # i <-> bins[i-1] < x <= bins[i]
            # Y is latent_size x batch size
            Y = np.vstack([np.digitize(X[:, i], class_boundaries[i, :], right=False) for i in range(model.latent_size)])
            Ys.append(Y)
    # put everything together
    X = np.vstack(Xs) # should be (batch x D) x D
    Y = np.hstack(Ys) # should be D x (BxD)
    # now we train a classifier and compute the importance matrix
    importance_matrix = np.zeros(shape=[model.latent_size, model.latent_size], dtype=np.float64)
    for i in range(model.latent_size):
        if verbose: print(f"Training linear model for dimension {i}")
        classifier = GradientBoostingClassifier()
        classifier.fit(X, Y[i,:])
        importance_matrix[:, i] = np.abs(classifier.feature_importances_)  # why abs?
    disentanglement = DCI.disentanglement(importance_matrix)
    completeness = DCI.completeness(importance_matrix)
    print("...done")

    return disentanglement, completeness, importance_matrix



def classification_on_responses(model:GenerativeAE, device, **kwargs):
    """Implements classification test on response map

    kwargs accepted keywords:
        - num_samples: number of samples from the latent space to use in the computation
        - steps: number of steps to take in the traversal
        - bins: number of bins to use for categorisation of the latent variables
        - random_state: random seed used for classifier training
        - all arguments accepted in 'sample_noise_from_prior'

    """
    print("Computing classification test responses ... ")
    if not kwargs.get('num_samples'):
        kwargs['num_samples'] = 50
    steps = kwargs.get('steps',40)
    bins = kwargs.get('bins',5)
    random_state = kwargs.get('random_state',11)
    verbose = kwargs.get('verbose',False)
    unit_dim = 1 #TODO check and fix

    outs = []; Ys = []; Y_hats = []; preds=[]; scores = []
    # sample N vectors from prior
    prior_samples = model.sample_noise_from_prior(device=device, **kwargs).detach()
    ranges = model.get_prior_range()
    # for each latent unit we start traversal
    # 1. obtain traversals values (for all the dimensions)
    traversals_steps = utils.get_traversals_steps(steps, ranges, relative=False).to(device).detach() #torch Tensor
    # 2. obtain categories for each latent dimension
    class_boundaries = utils.get_class_boundaries(bins, ranges).detach().cpu().numpy() #D x bins tensor
    for d in range(model.latent_size):
        with torch.no_grad():
            # 3. do traversals on dimension d
            traversals = utils.do_latent_traversals_multi_vec(prior_samples, unit_dim=unit_dim,
                                                              unit=d, values=traversals_steps[d],
                                                              device=device, relative=False) # shape steps x N x D
            X = traversals.reshape(-1,model.latent_size).cpu().numpy() # now we have (stepsxN) latent vectors
            np.random.shuffle(X) #shuffling before training
            # 4. obtain the labels associated to each vector
            Y = np.digitize(X[:,d],class_boundaries[d,:], right=False) # i <-> bins[i-1] < x <= bins[i]
            Ys.append(Y)
            # 5. now we train a linear classifier to predict the class labels from X and we score it on the training set
            if verbose: print(f"Training linear model for dimension {d}")
            #shuffle before training
            classifier = linear_model.LogisticRegression(random_state=random_state)
            classifier.fit(X, Y)
            pred = classifier.predict(X)
            preds.append(pred)
            prior_accuracy = np.mean(pred == Y)
            if verbose: print(f"Prior accuracy: {prior_accuracy:.2g}")
            # 6. obtain response vectors
            out = model.decode(torch.Tensor(X).to(device), activate=True)
            outs.append(out)
            X_hat = model.encode_mu(out).cpu().numpy()
            # 7. score accuracy in the response space
            Y_hat = classifier.predict(X_hat)
            Y_hats.append(Y_hat)
            response_accuracy = np.mean(Y_hat==Y)
            if verbose: print(f"Response accuracy: {response_accuracy:.2g}")
            scores.append(np.asarray([prior_accuracy, response_accuracy]))


    print("...done")

    return scores, outs, Ys, Y_hats, preds

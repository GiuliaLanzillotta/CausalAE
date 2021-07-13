""" Script containing methods to compute latent responses """

from torch.utils.data import DataLoader
from models.BASE import GenerativeAE
from torch import Tensor
import torch

def compute_response_matrix(dataloader:DataLoader,
                            model:GenerativeAE,
                            device:str,
                            num_batches=10,
                            num_samples=100):
    """Computes latent response matrix as described in https://arxiv.org/abs/2106.16091
    Note: the response matrix will be approximated here as only a part of the dataset will be used
    - num_samples= number of samples from the aggregate posterior (to obtain v_i)"""
    originals = []
    for i in range(num_batches):
        observations, _ = next(iter(dataloader))
        with torch.no_grad():
            codes = model.encode(observations.to(device)) # samples from posterior
            originals.append(codes)
    originals = torch.vstack(originals); N, D = originals.shape # this will constitute the base for the aggregate posterior
    Rz = compute_response(originals, model, device)

    # to obtain a perfect estimate we would need N to be the size of the dataset and
    # num samples to be comparable to it
    matrix = torch.zeros(D, D).to(device)
    for d in range(D):
        # for each new dimension obtain num_samplesxN new latent vectors
        new_codes = torch.stack(num_samples*[originals]).to(device)
        idxs = torch.multinomial(torch.ones(N), N*num_samples, replacement=True).to(device)
        # resample only the given dimension for each latent vector
        new_codes[:, :, d] = originals[idxs, d].view(num_samples, N)
        # compute the response to the resampled vector
        responses = compute_response(new_codes.view(-1,D), model, device).view(num_samples, N, D)
        #compute the distance between the response and the original latent vector
        delta = responses - Rz
        # square and average
        matrix[d, :] = torch.sum(delta**2, axis=[0,1])/(N*num_samples)

    return matrix



def compute_response(latent_vectors:Tensor, model:GenerativeAE, device:str):
    """ computes R(z) = E(D(z))
    Note: for now all our models have deterministic decoders."""
    with torch.no_grad():
        z_hat = model.encode(model.decode(latent_vectors.to(device), activate=True))
    return z_hat


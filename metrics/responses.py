""" Script containing methods to compute latent responses """

from torch.utils.data import DataLoader
from models.BASE import GenerativeAE
from torch import Tensor
import torch
from datasets.utils import gen_bar_updater

def compute_response_matrix(dataloader:DataLoader,
                            model:GenerativeAE,
                            device:str,
                            num_batches=10,
                            num_samples=100,
                            scaling=True):
    """Computes latent response matrix as described in https://arxiv.org/abs/2106.16091
    Note: the response matrix will be approximated here as only a part of the dataset will be used
    - num_samples= number of samples from the aggregate posterior (to obtain v_i)"""
    originals = []
    original_responses = []

    with torch.no_grad():

        print("Sampling from aggregate posterior...")
        updater = gen_bar_updater()

        for i in range(num_batches):
            observations, _ = next(iter(dataloader))
            #TODO: support for stochastic decoders
            codes = model.encode_mu(observations.to(device)) # samples from posterior
            originals.append(codes)
            original_responses.append(compute_response(codes, model, device))
            updater(i+1, 1, num_batches)
        all_originals = torch.vstack(originals); N, D = all_originals.shape # this will constitute the base for the aggregate posterior
        B = originals[0].shape[0]

        # to obtain a perfect estimate we would need N to be the size of the dataset and
        # num samples to be comparable to it
        print("Computing responses across latent dimensions...")
        updater = gen_bar_updater()
        matrix = torch.zeros(D, D).to(device)
        for d in range(D):
            # for each new dimension obtain num_samplesxN new latent vectors
            for b in range(num_batches):
                batch = originals[b]
                batch_responses = original_responses[b]
                new_codes = batch.clone().detach() # B x D
                for i in range(num_samples):
                    idxs = torch.multinomial(torch.ones(N), B, replacement=True).to(device)
                    new_codes[:, d] = all_originals[idxs, d] # still B x D
                    new_responses = compute_response(new_codes, model, device)
                    delta = new_responses - batch_responses # B x D
                    matrix[d, :] = matrix[d, :] + torch.sum(delta**2, axis=0) # 1 x D
            updater(d+1, 1, D)
            matrix[d,:]/=float(N*num_samples) # taking the average at the end
            if scaling: # normalise the dimension responses by its standard deviation
                # if the response is higher than 1 the dimension affects others
                # if it's less than 1 the dimension is not used
                matrix[d,:]/=torch.pow(torch.std(matrix[d,:]),2)

        matrix = torch.sqrt(matrix)

        """ 

            
            Rz = torch.vstack(original_responses)
            new_codes = torch.stack(num_samples*[all_originals]).to(device)
            idxs = torch.multinomial(torch.ones(N), N*num_samples, replacement=True).to(device)
            # resample only the given dimension for each latent vector
            new_codes[:, :, d] = all_originals[idxs, d].view(num_samples, N)
            # compute the response to the resampled vector
            new_responses = compute_response(new_codes.view(-1,D), model, device).view(num_samples, N, D)
            #compute the distance between the response and the original latent vector
            delta = new_responses - Rz
            # square and average
            matrix[d, :] = torch.sum(delta**2, axis=[0,1])/(N*num_samples)
            updater(d+1, 1, D)
        """


    return matrix



def compute_response(latent_vectors:Tensor, model:GenerativeAE, device:str):
    """ computes R(z) = E(D(z))
    Note: for now all our models have deterministic decoders."""
    with torch.no_grad():
        z_hat = model.encode_mu(model.decode(latent_vectors.to(device), activate=True))
    return z_hat


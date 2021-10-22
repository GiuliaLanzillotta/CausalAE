"""Inference net to solve inference tasks"""
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models import FCBlock, GenerativeAE


class INET(nn.Module):
    """Inference net"""

    def __init__(self, trained_net:GenerativeAE, input_size:int, output_size:int, **kwargs):
        super().__init__()
        self.causal = kwargs.get('causal',False)
        self.multi_class = kwargs.get('multi_class')
        self.trained_net = trained_net
        self.trained_net.requires_grad_(False)
        self.input_size = input_size
        self.output_size = output_size
        self.softmax = nn.Softmax()
        self.FC = FCBlock(input_size, [64, 64, output_size], nn.LeakyReLU)
        if self.multi_class:
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, X):
        """X: Tensor containing input images
        Output: vector of probabilities of size n
        The i-th element of the vector is a real number between 0
        (attribute i is not present) and 1 (attribute i is present)."""
        with torch.no_grad():
            representation = self.trained_net.get_representation(X, causal=self.causal).detach().clone()
        logits = self.FC(representation)
        return logits


    def compute_loss(self, predictions, Y):
        """Computes cross-entropy loss on the input
        @one_hot: whether the output dimensions encode multiple classes  """
        if self.multi_class:
            _loss = torch.sum(self.loss(predictions, Y), dim=1).mean()
        else: _loss = self.loss(predictions, Y).mean()
        return _loss

    def compute_accuracy(self, predictions, Y):
        """Computes accuracy score given the predictions as logits and the true labels
        Note: Y and predictions must be on the same device """
        if not self.multi_class: # categorical case
            labels = torch.argmax(predictions, dim=1)
            accuracy = torch.sum((labels == Y).float())
        else:
            labels = (self.softmax(predictions)>0.5).float().to(self.device)
            accuracy = torch.sum(torch.sum(labels == Y, dim=1)/Y.shape[1])
        return accuracy









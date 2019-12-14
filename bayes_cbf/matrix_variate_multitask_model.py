import numpy as np
import torch

from gpytorch.means import MultitaskMean

from bayes_cbf.matrix_variate_multitask_kernel import prod


class HetergeneousMatrixVariateMean(MultitaskMean):
    """
    Computes a mean depending on the input.

    Our mean can be the mean of either of the two related GaussianProcesses

        Xdot = F(X)ᵀU

    or

        Y = F(X)ᵀ

    We take input in the form

        M, X, U = MXU

    where M is the mask, where 1 value means we want Xdot = F(X)ᵀU, while 0
    means that we want Y = F(X)ᵀ
    """
    def __init__(self, mean_module, decoder, matshape, **kwargs):
        num_tasks = prod(matshape)
        super().__init__(mean_module, num_tasks, **kwargs)
        self.decoder = decoder
        self.matshape = matshape

    def mean1(self, UH, mu):
        # TODO: Make this a separate module
        XdotMean = UH.unsqueeze(-2) @ mu # D x n
        output = XdotMean.reshape(-1)
        return output

    def mean2(self, mu):
        # TODO: Make this a separate module
        return mu.reshape(-1)

    def forward(self, MXU):
        assert not torch.isnan(MXU).any()
        B = MXU.shape[:-1]

        Ms, _, UH = self.decoder.decode(MXU)
        assert Ms.size(-1) == 1
        Ms = Ms[..., 0]
        idxs = torch.nonzero(Ms - Ms.new_ones(Ms.size()))
        idxend = torch.min(idxs) if idxs.numel() else Ms.size(-1)
        mu = torch.cat([sub_mean(MXU).unsqueeze(-1)
                        for sub_mean in self.base_means], dim=-1)
        assert not torch.isnan(mu).any()
        mu  = mu.reshape(-1, *self.matshape)
        output = None
        if idxend != 0:
            # assume sorted
            assert (Ms[..., idxend:] == 0).all()
            output = self.mean1(UH[..., :idxend, :], mu[:idxend, ...])

        if Ms.size(-1) != idxend:
            Fmean = self.mean2(mu[idxend:, ...])
            output = torch.cat([output, Fmean]) if output is not None else Fmean
        return output

    def state_dict(self):
        return dict(
            matshape=self.matshape,
            decoder=self.decoder.state_dict()
        )

    def load_state_dict(self, state_dict):
        self.matshape = state_dict.pop('matshape')
        self.decoder.load_state_dict(state_dict['decoder'])



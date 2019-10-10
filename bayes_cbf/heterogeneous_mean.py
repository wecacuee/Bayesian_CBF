import torch
from gpytorch.means import MultitaskMean, ConstantMean


class HetergeneousMatrixVariateMean(MultitaskMean):
    def __init__(self, *args, decoder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = decoder

    def forward(self, MXU):
        #
        B = MXU.shape[:-1]
        Ms, X, UH = self.decoder.decode(MXU)
        assert Ms.size(-1) == 1
        Ms = Ms[..., 0]
        idxs = torch.nonzero(Ms - Ms.new_ones(Ms.size()))
        idxend = torch.min(idxs) if idxs.numel() else Ms.size(-1)
        # assume sorted
        assert (Ms[..., idxend:] == 0).all()
        UH = UH[..., :idxend, :]
        X1 = X[..., :idxend, :]
        X2 = X[..., idxend:, :]
        mu = torch.cat([sub_mean(MXU).unsqueeze(-1)
                        for sub_mean in self.base_means], dim=-1)
        mp1 = UH.size(-1)
        n   = X.size(-1)
        mu  = mu.reshape(-1, mp1, n)
        XdotMean = UH.unsqueeze(-2) @ mu[:idxend, ...] # D x n
        output = XdotMean.reshape(idxend, -1)
        if Ms.size(-1) != idxend:
            Fmean = mu[idxend:, ...].reshape(Ms.size(-1) - idxend, -1)
            output = torch.cat([output, Fmean])
        return output

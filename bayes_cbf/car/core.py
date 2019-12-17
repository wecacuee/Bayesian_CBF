from HyundaiGenesis import HyundaiGenesisDynamicsModel


class HyundaiGenesisControlAffine:
    def __init__(self):
        self.hgdm = HyundaiGenesisDynamicsModel()

    @property
    def state_size(self):
        pass

    @property
    def ctrl_size(self):
        pass

    def f_func(self, X):
        pass

    def g_func(self, X):
        pass

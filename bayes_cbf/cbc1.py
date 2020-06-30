from abc import ABC, abstractmethod, abstractproperty
from functools import partial
import math

from scipy.special import erfinv

from .misc import to_numpy
from .gp_algebra import DeterministicGP

def cbc1_safety_factor(δ):
    assert δ < 0.5 # Ask for at least more than 50% safety
    factor = math.sqrt(2)*erfinv(1 - 2*δ)
    assert factor > 1
    return factor


class RelDeg1Safety(ABC):
    @abstractproperty
    def gamma(self):
        pass

    @abstractproperty
    def model(self):
        pass

    @abstractmethod
    def cbf(self, x):
        pass

    @abstractmethod
    def grad_cbf(self, x):
        pass

    @abstractproperty
    def max_unsafe_prob(self):
        pass

    def cbc(self, u0):
        h_gp = DeterministicGP(lambda x: self.gamma * self.cbf(x),
                               shape=(1,), name="h(x)")
        grad_h_gp = DeterministicGP(self.grad_cbf,
                                    shape=(self.model.state_size,),
                                    name="∇ h(x)")
        fu_gp = self.model.fu_func_gp(u0)
        cbc = grad_h_gp.t() @ fu_gp + h_gp
        return cbc

    def safety_factor(self):
        return cbc1_safety_factor(self.max_unsafe_prob)

from abc import ABC, abstractmethod, abstractproperty
from functools import partial
import math

from scipy.special import erfinv

from .cbc2 import cbc2_safety_socp
from .misc import to_numpy
from .gp_algebra import DeterministicGP

def cbc1_safety_factor(δ):
    assert δ < 0.5 # Ask for at least more than 50% safety
    factor = math.sqrt(2)*erfinv(1 - 2*δ)
    assert factor > 1
    return factor

cbc1_safety_socp = partial(cbc2_safety_socp,
                           safety_factor=cbc1_safety_factor)


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

    def cbc1(self, u0):
        h_gp = DeterministicGP(lambda x: self.gamma * self.cbf(x),
                               shape=(1,), name="h(x)")
        grad_h_gp = DeterministicGP(self.grad_cbf,
                                    shape=(self.model.state_size,),
                                    name="∇ h(x)")
        fu_gp = self.model.fu_func_gp(u0)
        cbc = grad_h_gp.t() @ fu_gp + h_gp
        return cbc

    def as_socp(self, i, xi, u0, convert_out=to_numpy):
        return list(map(convert_out,
                        cbc1_safety_socp(self.cbc1, xi, u0,
                                         max_unsafe_prob=self.max_unsafe_prob)))

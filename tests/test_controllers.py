import pytest
import torch
from bayes_cbf.controllers import convert_cbc_terms_to_socp_terms
from bayes_cbf.misc import to_numpy, random_psd

def test_convert_cbc_terms_to_socp_term(m = 2,
                                        extravars = 2):
    bfe = torch.rand((m,))
    e = torch.rand(1)
    V_hom = random_psd(m+1)
    V = V_hom[1:, 1:]
    bfv = V_hom[1:, 0] * 2
    v = V_hom[0, 0]
    u = torch.rand((m,))
    A, bfb, bfc, d = convert_cbc_terms_to_socp_terms(
        bfe, e, V, bfv, v, extravars, testing=True)
    y_u  = torch.cat((torch.zeros(extravars), u))
    std_rhs = (A @ y_u + bfb).norm()
    mean_rhs = bfc @ y_u + d
    std_lhs = torch.sqrt(u.T @ V @ u + bfv @ u + v)
    mean_lhs = - bfe @ u - e
    assert to_numpy(mean_lhs ) == pytest.approx(to_numpy(mean_rhs ), abs=1e-4, rel=1e-2)
    assert to_numpy(std_lhs ) == pytest.approx(to_numpy(std_rhs ), abs=1e-4, rel=1e-2)
    assert to_numpy(mean_lhs + std_lhs) == pytest.approx(to_numpy(mean_rhs + std_rhs), abs=1e-4, rel=1e-2)

if __name__ == '__main__':
    test_convert_cbc_terms_to_socp_term()

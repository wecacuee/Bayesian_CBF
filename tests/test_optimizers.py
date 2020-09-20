import pytest
import numpy as np
from bayes_cbf.optimizers import convert_socp_to_cvxopt_format, optimizer_socp_cvxopt


def test_cvxopt_example():
    from cvxopt import matrix, solvers
    c = matrix([-2., 1., 5.])
    G = [ matrix( [[12., 13., 12.], [6., -3., -12.], [-5., -5., 6.]] ) ]
    G += [ matrix( [[3., 3., -1., 1.], [-6., -6., -9., 19.], [10., -2., -2., -3.]] ) ]
    h = [ matrix( [-12., -3., -2.] ),  matrix( [27., 0., 3., -42.] ) ]
    sol = solvers.socp(c, Gq = G, hq = h)
    assert sol['status'] == 'optimal'
    assert np.asarray(sol['x']) == pytest.approx(np.array([
        [-5.02e+00],
        [-5.77e+00],
        [-8.52e+00]]), abs=1e-3, rel=1e-2)
    assert np.asarray(sol['zq'][0]) == pytest.approx(np.array([
        [ 1.34e+00],
        [-7.63e-02],
        [-1.34e+00]]), abs=1e-3, rel=1e-2)
    assert np.asarray(sol['zq'][1]) == pytest.approx(np.array([
        [ 1.02e+00],
        [ 4.02e-01],
        [ 7.80e-01],
        [-5.17e-01]]), abs=1e-3, rel=1e-2)

def test_controller_socp_cvxopt():
    """
    From cvxopt socp example:

    >>> from cvxopt import matrix, solvers
    >>> c = matrix([-2., 1., 5.])
    >>> G = [ matrix( [[12., 13., 12.], [6., -3., -12.], [-5., -5., 6.]] ) ]
    >>> G += [ matrix( [[3., 3., -1., 1.], [-6., -6., -9., 19.], [10., -2., -2., -3.]] ) ]
    >>> h = [ matrix( [-12., -3., -2.] ),  matrix( [27., 0., 3., -42.] ) ]
    >>> sol = solvers.socp(c, Gq = G, hq = h)
    >>> sol['status']
    optimal
    >>> print(sol['x'])
    [-5.02e+00]
    [-5.77e+00]
    [-8.52e+00]
    >>> print(sol['zq'][0])
    [ 1.34e+00]
    [-7.63e-02]
    [-1.34e+00]
    >>> print(sol['zq'][1])
    [ 1.02e+00]
    [ 4.02e-01]
    [ 7.80e-01]
    [-5.17e-01]
    """
    linear_objective = np.array([-2., 1., 5.])
    # | A x + b |₂ ≺ cᵀ x + d
    A = list(map(np.array, [
        [[-13., 3., 5.],
         [-12., 12., -6.]],

        [[-3., 6., 2.],
         [ 1., 9., 2.],
         [-1., -19., 3.]]
    ]))
    b = list(map(np.array, [
        [-3, -2],

        [0., 3., -42]
    ]))

    c = list(map(np.array, [
        [-12., -6., 5,],

        [-3., 6., -10]
    ]))

    d = list(map(np.array, [
        -12,

        27
    ]))

    exp_uopt = np.array([
        [-5.02e+00],
        [-5.77e+00],
        [-8.52e+00]])

    names = ("1", "2")

    socp_constraints = zip(A, b, c, d)
    named_socp_constraints = list(zip(names, socp_constraints))


    cvx_c, cvx_Gqs, cvx_hqs = convert_socp_to_cvxopt_format(
        linear_objective,
        named_socp_constraints)

    exp_Gqs = [ np.array( [[12., 13., 12.], [6., -3., -12.], [-5., -5., 6.]] ) ]
    exp_Gqs += [ np.array( [[3., 3., -1., 1.], [-6., -6., -9., 19.], [10., -2., -2., -3.]] ) ]
    exp_hqs = [ np.array( [-12., -3., -2.] ),  np.array( [27., 0., 3., -42.] ) ]

    for Gq, hq, exp_Gq, exp_hq in zip(cvx_Gqs, cvx_hqs, exp_Gqs, exp_hqs):
        assert Gq.T == pytest.approx(exp_Gq)
        assert hq.flatten() == pytest.approx(exp_hq)


    from cvxopt import solvers, matrix
    inputs_socp = dict(c = matrix(cvx_c),
                   Gq = list(map(matrix, cvx_Gqs)),
                   hq = list(map(matrix, cvx_hqs)))
    #print("test_optimizers.py:72", inputs_socp)
    sol = solvers.socp(**inputs_socp)
    uopt = sol['x']

    assert np.asarray(uopt) == pytest.approx(np.asarray(exp_uopt), rel=1e-2)

    uopt = optimizer_socp_cvxopt(np.random.rand(3),
                                 linear_objective,
                                 named_socp_constraints)
    assert np.asarray(uopt) == pytest.approx(np.asarray(exp_uopt).flatten(), rel=1e-2)

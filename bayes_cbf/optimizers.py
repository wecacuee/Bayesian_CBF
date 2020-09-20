import numpy as np

class InfeasibleProblemError(ValueError):
    pass

def convert_socp_to_cvxopt_format(c, socp_constraints):
    """
    socp_constraints = [(Nameₖ, (Aₖ, bfbₖ, bfcₖ, dₖ), ...]

    min_u   cᵀu
       s.t. dₖ + bfcₖ u ≻ |Aₖ u + bfbₖ|₂

    min cᵀu
    s.t Gₖ x + sₖ = hₖ,    k = 0, ..., M
        A x = b
        h₀ - G₀ x ⪰ 0     # Component wise inequalities
        hₖ[0] - Gₖ[0, :] x ⪰ | hₖ[1:] - Gₖ[1:, :] x |₂,    k = 1, ..., M

    """
    m = np.asarray(c).shape[-1]
    Gqs = []
    hqs = []
    for name, (A, bfb, bfc, d) in socp_constraints:
        # (name, (A, b, c, d))
        # |Au + bfb| < bfc' u + d
        # s.t. ||Ax + b||_2 <= c'x + d
        # But || h₁ - G₁x ||₂ ≺ h₀ - g₀' x
        # G = [g₀'; G₁] = [-c'; -A]
        # h = [h₀; h₁] = [d; b]
        Gqi = np.zeros((A.shape[0]+1, m))
        Gqi[0, :] = -bfc
        Gqi[1:, :] = -A
        Gqs.append(Gqi)
        hqi = np.zeros((A.shape[0]+1, 1))
        hqi[0, 0] = d.reshape(1, 1)
        hqi[1:, 0] = bfb
        hqs.append(hqi)

    return c, Gqs, hqs

def optimizer_socp_cvxopt(u0, linear_objective, socp_constraints):
    """
    Solve the optimization problem

    min_u   A u + b
       s.t. h₀ - (G u)₀ ≻ |h₁ - (Gu)₁ |₂

    u0: reference control signal
    linear_objective: (

    convert to cvxopt format and pass to cvxopt

    min cᵀu
    s.t Gₖ x + sₖ = hₖ,    k = 0, ..., M
        A x = b
        s₀ ⪰ 0            # Component wise inequalities
        sₖ₀ ⪰ | sₖ₁ |₂,    k = 1, ..., M

    min cᵀu
    s.t Gₖ x + sₖ = hₖ,    k = 0, ..., M
        A x = b
        h₀ - G₀ x ⪰ 0     # Component wise inequalities
        hₖ[0] - Gₖ[0, :] x ⪰ | hₖ[1:] - Gₖ[1:, :] x |₂,    k = 1, ..., M

    """
    from cvxopt import solvers, matrix
    c, Gqs, hqs = convert_socp_to_cvxopt_format(linear_objective,
                                                socp_constraints)
    inputs_socp = dict(c = matrix(c),
                       Gq = list(map(matrix, Gqs)),
                       hq = list(map(matrix, hqs)))
    #print("optimizers.py:72", inputs_socp)
    sol = solvers.socp(**inputs_socp)
    if sol['status'] != 'optimal':
        if sol['status'] == 'primal infeasible':
            y_uopt = sol.get('z', u0)
        else:
            y_uopt = u0

        print("{c}.T [y, u]\n".format(c=c)
              + "s.t. "
              + "".join((" sq = {hq} - {Gq} [{y_uopt}]\n".format(hq=np.asarray(hq),
                                                             Gq=np.asarray(Gq),
                                                             y_uopt=np.asarray(y_uopt))
                     for Gq, hq in zip(Gqs, hqs))))
        raise InfeasibleProblemError("Infeasible problem: %s" % sol['status'])

    return np.asarray(sol['x']).astype(u0.dtype).reshape(-1)


def optimizer_socp_cvxpy(u0, linear_objective, socp_constraints, solver='GUROBI'):
    import cvxpy as cp
    x = cp.Variable(u0.shape)
    x.value = u0
    constraints = []
    for name, (A, bfb, bfc, d) in socp_constraints:
        constraints.append(cp.norm(A @ x + bfb) <= bfc @ x + d)

    obj = cp.Minimize(linear_objective @ x)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver)
    return x.value


def optimizer_qp_cvxpy(u0, quadratic_objective, linear_constraints, solver='GUROBI'):
    import cvxpy as cp
    x = cp.Variable(u0.shape)
    x.value = u0
    constraints = []
    for name, (bfc, d) in linear_constraints:
        constraints.append(0 <= bfc @ x + d)
    A, bfb = quadratic_objective
    obj = cp.Minimize(cp.sum_squares(A @ x + bfb))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver)
    return x.value

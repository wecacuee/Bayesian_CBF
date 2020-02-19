import numpy as np
import torch

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, solver=None,
                    initvals=None, **kwargs):
    import cvxopt
    from cvxopt import matrix
    #P = (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
    else:
        args.extend([None, None])
    if A is not None:
        args.extend([matrix(A), matrix(b)])
    else:
        args.extend([None, None])
    args.extend([initvals, solver])
    solvers = cvxopt.solvers
    old_options = solvers.options.copy()
    solvers.options.update(kwargs)
    try:
        sol = cvxopt.solvers.qp(*args)
    except ValueError:
        return None
    finally:
        solvers.options.update(old_options)
    if 'optimal' not in sol['status']:
        return None
    return np.asarray(sol['x']).reshape((P.shape[1],))


def control_QP_cbf_clf(x,
                       ctrl_aff_constraints,
                       constraint_margin_weights=[]):
    """
    Args:
          A_cbfs: A tuple of CBF functions
          b_cbfs: A tuple of CBF functions
          constraint_margin_weights: Add a margin constant to the constraint
                                     that is maximized.

    """
    #import ipdb; ipdb.set_trace()
    clf_idx = 0
    A_total = np.vstack([af.A(x).detach().numpy()
                         for af in ctrl_aff_constraints])
    b_total = np.vstack([af.b(x).detach().numpy()
                         for af in ctrl_aff_constraints]).flatten()
    D_u = A_total.shape[1]
    N_const = A_total.shape[0]

    # u0 = l*g*sin(theta)
    # uopt = 0.1*g
    # contraints = A_total.dot(uopt) - b_total
    # assert contraints[0] <= 0
    # assert contraints[1] <= 0
    # assert contraints[2] <= 0


    # [A, I][ u ]
    #       [ ρ ] ≤ b for all constraints
    #
    # minimize
    #         [ u, ρ1, ρ2 ] [ 1,     0] [  u ]
    #                       [ 0,   100] [ ρ2 ]
    #         [A_cbf, 1] [ u, -ρ ] ≤ b_cbf
    #         [A_clf, 1] [ u, -ρ ] ≤ b_clf
    N_slack = len(constraint_margin_weights)
    A_total_rho = np.hstack(
        (A_total,
         np.vstack((-np.eye(N_slack),
                    np.zeros((N_const - N_slack, N_slack))))
        ))
    A = A_total
    P_rho = np.eye(D_u + N_slack)
    P_rho[D_u:, D_u:] = np.diag(constraint_margin_weights)
    q_rho = np.zeros(P_rho.shape[0])
    #u_rho_init = np.linalg.lstsq(A_total_rho, b_total - 1e-1, rcond=-1)[0]
    u_rho = cvxopt_solve_qp(P_rho.astype(np.float64),
                            q_rho.astype(np.float64),
                            G=A_total_rho.astype(np.float64),
                            h=b_total.astype(np.float64),
                            show_progress=False,
                            maxiters=1000)
    if u_rho is None:
        raise RuntimeError("""QP is infeasible
        minimize
        u_rhoᵀ {P_rho} u_rho
        s.t.
        {A_total_rho} u_rho ≤ {b_total}""".format(
            P_rho=P_rho,
            A_total_rho=A_total_rho, b_total=b_total))
    # Constraints should be satisfied
    constraint = A_total_rho @ u_rho - b_total
    assert np.all((constraint <= 1e-2) | (constraint / np.abs(b_total) <= 1e-2))
    return torch.from_numpy(u_rho[:D_u]).to(dtype=x.dtype)


def to_numpy(x):
    return x.detach().double().cpu().numpy() if isinstance(x, torch.Tensor) else x


def controller_qcqp(u0, objective, quadratic_constraints):
    import cvxpy as cp
    import gurobipy
    u = cp.Variable(u0.shape)
    cp_obj = objective(u)
    cp_qc = [qc(u) for qc in quadratic_constraints]
    prob = cp.Problem(cp.Minimize(cp_obj), cp_qc)
    prob.solve(solver=cp.GUROBI)
    return u.value

class InfeasibleOptimization(Exception):
    pass

def controller_qcqp_gurobi(u0, quad_objective, quadratic_constraints,
                           DisplayInterval=120,
                           OutputFlag=0,
                           NonConvex=2,
                           **kwargs):
    import gurobipy as gp
    from gurobipy import GRB
    # Create a new model
    m = gp.Model("controller_qcqp_gurobi")
    m.Params.OutputFlag = OutputFlag
    m.Params.DisplayInterval = DisplayInterval
    m.Params.NonConvex = NonConvex
    for k, v in kwargs.items():
        m.setParam(k, v)

    # Create variables
    u = m.addMVar(shape=u0.shape, vtype=GRB.CONTINUOUS, name="u")


    m.setMObjective(*quad_objective, sense=GRB.MINIMIZE)
    for i, (name, (Q, c, const)) in enumerate(quadratic_constraints):
        m.addMQConstr(Q, c, "<", -const, xQ_L=u, xQ_R=u, xc=u, name=name)
    m.optimize()
    if m.getAttr(GRB.Attr.Status) == GRB.OPTIMAL:
        return u.X
    elif m.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
        raise InfeasibleOptimization("Optimal value not found. Problem is infeasible.")
    elif m.getAttr(GRB.Attr.Status) == GRB.UNBOUNDED:
        raise InfeasibleOptimization("Optimal value not found. Problem is unbounded.")
    return u.X

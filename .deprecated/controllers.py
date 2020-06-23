
class ControlCBFLearned(Controller):
    def quad_objective(self, i, x, u0, convert_out=to_numpy):
        # TODO: Too complex
        # TODO: Make it (u - π(x))ᵀR(u - π(x))
        x_g = self.x_goal
        P = self.x_quad_goal_cost
        R = torch.eye(self.u_dim)
        λ = self.egreedy(i)
        fx = (x + self.dt * self.model.f_func(x)
              + self.dt * self.mean_dynamics_model.f_func(x))
        Gx = (self.dt * self.model.g_func(x.unsqueeze(0)).squeeze(0)
              + self.dt * self.mean_dynamics_model.g_func(x))
        # xp = fx + Gx @ u
        # (1-λ)(xp - x_g)ᵀ P (xp - x_g) + λ (u - u₀)ᵀ R (u - u₀)
        # Quadratic term: uᵀ (λ R + (1-λ)GₓᵀPGₓ) u
        # Linear term   : - (2λRu₀ + 2(1-λ)GₓP(x_g - fx)  )ᵀ u
        # Constant term : + (1-λ)(x_g-fx)ᵀP(x_g-fx) + λ u₀ᵀRu₀


        # Quadratic term λ R + (1-λ)Gₓᵀ P Gₓ
        Q = λ * R + (1-λ) * Gx.T @ P @ Gx
        # Linear term - (2λRu₀ + 2(1-λ)Gₓ P(x_g - fx)  )ᵀ u
        c = -2.0*(λ * R @ u0 + (1-λ) * Gx.T @ P @ (x_g - fx))
        # Constant term + (1-λ)(x_g-fx)ᵀ P (x_g-fx) + λ u₀ᵀRu₀
        const = (1-λ) * (x_g-fx).T @ P @ (x_g-fx) + λ * u0.T @ R @ u0

        Q_ = add_diag_const(Q)
        c_ = torch.cat((c, torch.tensor([0.])), dim=0)
        return list(map(convert_out, (Q_, c_, const)))
        # return list(map(convert_out, (R, torch.tensor([0]), torch.tensor(0))))


    def _stochastic_cbf2(self, i, x, u0, convert_out=to_numpy):
        (E_mean_A, E_mean_b), (var_k_Q, var_k_p, var_k_r), mean, var = \
            cbc2_quadratic_terms(
                self.cbf2.h2_col, self.cbf2.grad_h2_col, self.model, x, u0,
                k_α=self.cbf2.cbf_col_K_alpha)
        with torch.no_grad():
            δ = self.max_unsafe_prob
            ratio = (1-δ)/δ
            mean_Q = (E_mean_A.T @ E_mean_A).reshape(self.u_dim,self.u_dim)
            assert (torch.eig(var_k_Q)[0][:, 0] > 0).all()
            assert (torch.eig(mean_Q)[0][:, 0] > 0).all()
            A = var_k_Q * ratio # - mean_Q
            E_mean_A_ = torch.cat((E_mean_A, torch.tensor([-1.])), dim=0)
            A_ = add_diag_const(A, -1.)

            mean_p = ((2 * E_mean_A @ E_mean_b)
                      if E_mean_b.ndim
                      else (2 * E_mean_A * E_mean_b))
            b = var_k_p * ratio # - mean_p
            b_ = torch.cat((b, torch.tensor([0.])), dim=0)
            mean_r = (E_mean_b @ E_mean_b) if E_mean_b.ndim else (E_mean_b * E_mean_b)
            c = var_k_r * ratio # - mean_r
            constraints = [(r"$-E[CBC2] \le \alpha$",
                            list(map(convert_out,
                                     (torch.zeros(u0.shape[0]+1, u0.shape[0]+1),
                                     - E_mean_A_, -E_mean_b))))]
            constraints.append(
                (r"$\frac{1-\delta}{\delta} V[CBC2] \le \alpha$",
                 list(map(convert_out, (A_, b_, c)))))
            return constraints


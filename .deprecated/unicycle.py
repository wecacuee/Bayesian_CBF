from bayes_cbf.gp_algebra import GaussianProcess

class UnicyclePolarGoalFrame:
    """
    ż = ∑ gᵢ(z) uᵢ

    This system is controllable by continuous control law only if gᵢ(z*) are
    independent, differentiable, and m < n, where z* is the goal state.

    z = [x, y, ϕ]
    ctrl = [u, ω]

    ẋ = u cos ϕ
    ẏ = u sin ϕ
    ϕ̇ = ω

    So Aicardi et al (1995) suggested an alternative parameterization in polar
    coordinates in goal frame that makes gᵢ(z*) dependent. There many such
    parameterizations, but we use the one suggested by Aicardi et al.

    e = √x²+y²
    θ = tan⁻¹(y/x)
    α = θ - ϕ


    ė = - u cos α
    α̇ =   u sin α / e - ω
    θ̇ =   u sin α / e

    Aicardi et al (1995): Aicardi, Casalino, Bicchi, Balestrino. Closed loop
    Steering of Unicycle like Vehicles via Lyapunov Techniques. IEEE Robotics
    and Automation magazine. April 1995
    """
    @staticmethod
    def from_cartesian(Z_rel):
        x, y, ϕ = Z_rel[..., 0], Z_rel[..., 1], Z_rel[..., 2]
        e = Z_rel[..., :2].norm(p=2, dim=-1)
        if e.abs() > 1e-6:
            θ = y.atan2(x)
        else:
            θ = ϕ
        α = θ - ϕ
        return tensor.cat([e, α, θ], dim=-1)

    @staticmethod
    def to_cartesian(polar):
        e, α, θ = polar[..., 0], polar[..., 1], polar[..., 2]
        ϕ = θ - α
        x = e * θ.cos()
        y = e * θ.sin()
        return torch.cat([x, y, θ], dim=-1)

    @staticmethod
    def from_dot_cartesian(Z_rel, dot_Z_rel):
        x, y, ϕ = Z_rel[..., 0], Z_rel[..., 1], Z_rel[..., 2]
        dot_x, dot_y, dot_ϕ = dot_Z_rel[..., 0], dot_Z_rel[..., 1], dot_Z_rel[..., 2]

        polar = self.from_cartesian(Z_rel)
        e, α, θ = polar[..., 0], polar[..., 1], polar[..., 2]

        dot_e = - (dot_x + dot_y) / e
        dot_θ = (x * dot_y - y * dot_x)/ (e*e)
        dot_α = dot_θ - dot_ϕ
        return torch.cat([dot_e, dot_θ, dot_α], dim=-1)

    @staticmethod

    def to_dot_cartesian(polar, dot_polar):
        e, α, θ = polar[..., 0], polar[..., 1], polar[..., 2]
        dot_e, dot_θ, dot_α = dot_polar[..., 0], dot_polar[..., 1], dot_polar[..., 2]

        Z_rel = self.to_cartesian(polar)
        x, y, ϕ = Z_rel[..., 0], Z_rel[..., 1], Z_rel[..., 2]

        dot_x = dot_e * θ.cos() - e * θ.sin()
        dot_y = dot_e * θ.sin() + e * θ.cos()
        dot_ϕ = dot_θ - dot_α
        return torch.cat([dot_x, dot_y, dot_ϕ], dim=-1)

class UncycleDynamicsPolarGoalFrame(DynamicsModel):
    """
    ż = ∑ gᵢ(z) uᵢ

    This system is controllable by continuous control law only if gᵢ(z*) are
    independent, differentiable, and m < n, where z* is the goal state.

    z = [x, y, ϕ]
    ctrl = [u, ω]

    ẋ = u cos ϕ
    ẏ = u sin ϕ
    ϕ̇ = ω

    So Aicardi et al (1995) suggested an alternative parameterization in polar
    coordinates in goal frame that makes gᵢ(z*) dependent. There many such
    parameterizations, but we use the one suggested by Aicardi et al.

    e = √x²+y²
    θ = tan⁻¹(y/x)
    α = θ - ϕ


    ė = - u cos α
    α̇ =   u sin α / e - ω
    θ̇ =   u sin α / e

    Aicardi et al (1995): Aicardi, Casalino, Bicchi, Balestrino. Closed loop
    Steering of Unicycle like Vehicles via Lyapunov Techniques. IEEE Robotics
    and Automation magazine. April 1995
    """
    def __init__(self):
        super().__init__()
        self._ctrl_size = 2
        self._state_size = 3

    @property
    def ctrl_size(self):
        return self._ctrl_size

    @property
    def state_size(self):
        return self._state_size

    def f_func(self, X_in):
        """
                [ 0   ]
         f(x) = [ 0   ]
                [ 0   ]
        """
        return X_in.new_zeros(X_in.shape) * X_in

    def g_func(self, Polar_in, epsilon=1e-6):
        """
                [    -cos(α), 0  ]
         g(x) = [ sin(α) / e, -1 ]
                [ sin(α) / e, 0  ]

        if e == 0
        α = 0
        """
        Polar = Polar_in.unsqueeze(0) if Polar_in.dim() <= 1 else Polar_in
        gX = torch.zeros((*Polar_in.shape, self.m))
        e = Polar[..., 0:1].unsqueeze(-1)
        α = Polar[..., 2:3]
        α = α.unsqueeze(-1)
        zero = α.new_zeros((*X.shape[:-1], 1, 1))
        ones = α.new_ones((*X.shape[:-1], 1, 1))
        theta_dot = (α.sin() / e).where(e.abs() > epsilon, 1)
        e_dot = (-α.cos()).where(e.abs() > epsilon, -1)
        gX = torch.cat([torch.cat([e_dot, zero], dim=-1),
                        torch.cat([theta_dot, -ones], dim=-1),
                        torch.cat([theta_dot, zero], dim=-1)],
                       dim=-2)

        return gX.squeeze(0) if Polar_in.dim() <= 1 else gX

    def normalize_state(self, X_in):
        X_in[..., 2] = X_in[..., 2] % math.pi
        return X_in

class UnicycleDynamicsModelPolarWrapper(DynamicsModel):
    def __init__(self, cartesian_model, x_goal, converters=UnicyclePolarGoalFrame()):
        super().__init__()
        self._cartesian_model = cartesian_model
        self._x_goal = x_goal
        self._converters = converters

    @property
    def ctrl_size(self):
        return self._cartesian_model.ctrl_size

    @property
    def state_size(self):
        return self.cartesian_model.state_size


    def f_func(self, Polar_in):
        """
        T : p -> x
        f : x -> ẋ
         fₚ(p) = T⁻¹(fₓ(T(p)))
        """
        Polar = Polar_in.unsqueeze(0) if Polar_in.dim() <= 1 else Polar_in
        X_rel = self._converters.to_cartesian(Polar)
        Xdot = self.cartesian_model.f_func(X_rel + self._x_goal)
        dot_polar = self._converters.from_dot_cartesian(X_rel, Xdot)
        return dot_polar.squeeze(0) if Polar_in.dim() <= 1 else dot_polar

    def g_func(self, Polar_in):
        """
        T : p -> x
        g : x -> ẋ
         gₚ(p) = T⁻¹(gₓ(T(p)))
        """
        Polar = Polar_in.unsqueeze(0) if Polar_in.dim() <= 1 else Polar_in
        X_rel = self._converters.to_cartesian(Polar)
        gX = self.cartesian_model.g_func(X_rel + self._x_goal)
        gX_polar = self._converters.from_dot_cartesian(
            X_rel, gX.transpose(-1, -2)).transpose(-1, 2)
        return gX_polar.squeeze(0) if Polar_in.dim() <= 1 else gX_polar

    def fu_func_gp(self, u0):
        """
        f(x) ~ N(mu(x), knl(x, x'))
        T⁻¹(f(T(p)) ~ N( T⁻¹(mu(T(p))), knl(T(p), T(p')) )
        """
        cartesian_gp = self.cartesian_model.fu_func_gp(u0)
        T = self._converters.to_cartesian
        Tinv = self._converters.from_dot_cartesian
        return GaussianProcess(
            lambda p: Tinv(T(p), cartesian_gp.mean(T(p) + self._x_goal)),
            lambda p, pp: cartesian_gp.knl(T(p) + self._x_goal, T(pp) + self._x_goal),
            cartesian_gp.shape,
            name="f_polar(p)")


class OMPLPlanner(Planner):
    def __init__(self, x0, x_goal, numSteps, dt):
        self.x0 = x0
        self.x_goal = x_goal
        self.numSteps = numSteps
        assert self.numSteps >= 3
        self._solution_path = self._make_plan()
        self.dt = dt

    def _make_plan(self):
        from ompl import base as ob
        from ompl import geometric as og

        # create an SE2 state space
        space = ob.SE2StateSpace()

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-10)
        bounds.setHigh(10)
        space.setBounds(bounds)

        # create a simple setup object
        ss = og.SimpleSetup(space)
        # ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

        start = ob.State(space)
        # ... or set specific values
        start().setX(self.x0[0])
        start().setY(self.x0[1])
        start().setYaw(self.x0[2])

        goal = ob.State(space)
        # ... or set specific values
        goal().setX(self.x_goal[0])
        goal().setY(self.x_goal[1])
        goal().setYaw(self.x_goal[2])

        ss.setStartAndGoalStates(start, goal)

        # this will automatically choose a default planner with
        # default parameters
        solved = ss.solve(1.0)

        assert solved
        # try to shorten the path
        ss.simplifySolution()
        # print the simplified path
        return ss.getSolutionPath()


    def plan(self, t_step):
        pass

    def dot_plan(self, t_step):
        pass


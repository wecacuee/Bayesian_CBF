from abc import ABC, abstractmethod

import torch
import numpy as np
from scipy.interpolate import splrep, splev, spalde

from bayes_cbf.misc import normalize_radians, to_numpy


class Planner(ABC):
    @abstractmethod
    def plan(self, x, t):
        pass

    @abstractmethod
    def dot_plan(self, x, t):
        pass

class PiecewiseLinearPlanner(Planner):
    def __init__(self, x0, x_goal, numSteps, dt, frac_time_to_reach_goal=0.7):
        self.x0 = x0
        self.x_goal = x_goal
        self.numSteps = numSteps
        assert self.numSteps >= 3
        self.frac_time_to_reach_goal = frac_time_to_reach_goal
        self.dt = dt
        self._checkpoint_list = self._checkpoints()

    def _checkpoints(self):
        numSteps = self.numSteps
        x0 = self.x0
        x_goal = self.x_goal
        xdiff = (x_goal[:2] - x0[:2])
        xdiff_norm = xdiff / xdiff.norm()
        t_second_stage = min(int(numSteps*self.frac_time_to_reach_goal), numSteps-1)
        return [(t_second_stage,
                 torch.cat([x_goal[:2], xdiff_norm])),
                (numSteps,
                 torch.cat([x_goal[:2], x_goal[2:].cos(), x_goal[2:].sin()]))]

    def _get_checkpoint_interval(self, t_step):
        assert t_step <= self.numSteps
        prev_t, prev_x = 0, torch.cat([self.x0[:2], self.x0[2:].cos(), self.x0[2:].sin()])
        for checkpoint_t, checkpoint_x in self._checkpoint_list:
            if t_step <= checkpoint_t:
                break
            prev_t, prev_x = checkpoint_t, checkpoint_x
        assert prev_t != checkpoint_t
        return [(checkpoint_t, checkpoint_x), (prev_t, prev_x)]

    def _target_step(self, t_step):
        return min(t_step + max(int(0.1*self.numSteps), 1), self.numSteps)

    def plan(self, t_step):
        t_step = self._target_step(t_step)
        [(checkpoint_t, checkpoint_x), (prev_t, prev_x)] = self._get_checkpoint_interval(t_step)
        x_p =  (checkpoint_x - prev_x) * (t_step - prev_t) / (checkpoint_t - prev_t) + prev_x
        return torch.cat([x_p[:2], x_p[3:4].atan2(x_p[2:3])])

    def dot_plan(self, t_step):
        t_step = self._target_step(t_step)
        [(checkpoint_t, checkpoint_x), (prev_t, prev_x)] = self._get_checkpoint_interval(t_step)
        xdiff = (checkpoint_x - prev_x) / ((checkpoint_t - prev_t) * self.dt)
        return torch.cat([xdiff[:2], (xdiff[2:3] - xdiff[3:4])/(xdiff[2:4]**2).sum()])

class SplinePlanner(Planner):
    def __init__(self, x0, x_goal, numSteps, dt):
        self.x0 = x0
        self.x_goal = x_goal
        self.numSteps = numSteps
        assert self.numSteps >= 3
        knots = self._knots()
        self._x_spl = splrep(knots[:, 0], knots[:, 1])
        self._y_spl = splrep(knots[:, 0], knots[:, 2])
        self._yaw_spl = splrep(knots[:, 0], knots[:, 3])
        self.dt = dt

    def _knots(self):
        numSteps = self.numSteps
        x0 = to_numpy(self.x0)
        x_goal = to_numpy(self.x_goal)
        xdiff = (x_goal[:2] - x0[:2])
        desired_theta = np.arctan2(xdiff[1], xdiff[0])
        t_first_step = max(int(numSteps*0.1), 1)
        t_second_stage = min(int(numSteps*0.9), numSteps-1)
        dx = (x_goal - x0)/(t_second_stage - t_first_step)
        t_mid = (t_second_stage + t_first_step)/2
        x_mid = (x0 + x_goal)/2
        return np.array([
            [0, x0[0], x0[1], x0[2]],
            [t_first_step, x0[0], x0[1], desired_theta],
            [t_first_step + 1, x0[0]+dx[0], x0[1]+dx[1], desired_theta],
            [t_mid, x_mid[0], x_mid[1], desired_theta],
            [t_second_stage - 1, x_goal[0]-dx[0], x_goal[1]-dx[1], desired_theta],
            [t_second_stage, x_goal[0], x_goal[1], desired_theta],
            [numSteps, x_goal[0], x_goal[1], x_goal[2]]])

    def plan(self, t_step):
        return torch.from_numpy(np.hstack([splev(t_step, self._x_spl),
                                           splev(t_step, self._y_spl),
                                           splev(t_step, self._yaw_spl)])).to(
                                               device=getattr(self.x0, 'device', None),
                                               dtype=self.x0.dtype)

    def dot_plan(self, t_step):
        return torch.from_numpy(np.hstack([spalde(t_step, self._x_spl)[0],
                                           spalde(t_step, self._y_spl)[0],
                                           spalde(t_step, self._yaw_spl)[0]])).to(
                                               device=getattr(self.x0, 'device', None),
                                               dtype=self.x0.dtype)

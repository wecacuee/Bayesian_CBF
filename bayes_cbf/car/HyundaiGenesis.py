#import numpy as np
import torch
import math

class StateSE3(object):
  __slots__ = ['pose','twist']

class PoseSE3(object):
  __slots__ = ['position','orientation']

class TwistSE3(object):
  __slots__ = ['linear','angular']

def rotz(a):
  ''' a rotation of a about the Z axis'''
  ca, sa = torch.cos(a), torch.sin(a)
  zz, ee = torch.zeros_like(a), torch.ones_like(a)
  R = torch.empty(list(a.shape)+[3,3]) if isinstance(a, torch.Tensor) else torch.empty((3,3))
  R[...,0,0], R[...,0,1], R[...,0,2] = ca,-sa, zz
  R[...,1,0], R[...,1,1], R[...,1,2] = sa, ca, zz
  R[...,2,0], R[...,2,1], R[...,2,2] = zz, zz, ee
  return R

class AckermannParameters(object):
  __slots__ = ['lf', 'lr', 'half_width', 'mass', 'Iz', 'C_alpha_f', 'C_alpha_r',
               'acceleration_time_constant', 'steering_angle_time_constant']
  # [CoG to front axle (m), CoG to rear axle (m), half-width (m), mass (kg), 
  #  inertia (kg*m2), front axle cornering stiffness (N/rad), rear axle  cornering stiffness (N/rad),
  #  control delay (s), control delay (s)]

class AckermannInput(object):
  __slots__ = ['acceleration','steering_angle'] # m/s^2, rad


class HyundaiGenesisParameters(AckermannParameters):
  def __init__(self):
    self.lf = 1.5213 # m (CoG to front axle)
    self.lr = 1.4987 # m (CoG to rear axle)
    self.half_width = 0.945	 # m (half-width)
    self.mass = 2303.1 # kg (vehicle mass)
    self.Iz = 5520.1 # kg*m2 (vehicle inertia)
    self.C_alpha_f = 7.6419e4*2  # N/rad	(front axle cornering stiffness) # 200k
    self.C_alpha_r = 13.4851e4*2	# N/rad	(rear axle cornering stiffness)	 # 250k
    self.acceleration_time_constant = 0.4 # s
    self.steering_angle_time_constant  = 0.1 # s


def rotmat_to_z(R):
  """
  Assuming R is rotation around +Z axis, returns the angle of rotation.

  >>> a = torch.rand(1) * 2 * math.pi
  >>> agot = rotmat_to_z(rotz(a))
  >>> a == agot
  True
  """
  return torch.atan2(R[..., 1,0], R[..., 0,0])


class StateAsArray:
  def serialize(self, state, inp):
    pos = state.pose.position[:, :2]
    ori = rotmat_to_z(state.pose.orientation)
    v = state.twist.linear[:, :2]
    w = state.twist.angular[:, 2:3]
    iacc = inp.acceleration
    i_st_angle = inp.steering_angle
    return torch.cat((pos, ori.unsqueeze(-1), v, w, iacc, i_st_angle), dim=-1)

  def deserialize(self, X):
    (pos, ori, v, w, iacc, i_st_angle) = (
      X[:, :2], X[:, 2], X[:, 3:5], X[:, 5], X[:, 6], X[:, 7])

    state = StateSE3()
    state.pose = PoseSE3()
    state.pose.position = torch.zeros(pos.shape[0], 3)
    state.pose.position[:, :2] = pos
    state.pose.orientation = rotz(ori)
    state.twist = TwistSE3()
    state.twist.linear = torch.zeros(pos.shape[0], 3)
    state.twist.linear[:, :2] = v
    state.twist.angular = torch.zeros(pos.shape[0], 3)
    state.twist.angular[:, 2] = w

    inp = AckermannInput()
    inp.acceleration = iacc
    inp.steering_angle = i_st_angle
    return state, inp


class HyundaiGenesisDynamicsModel(object):
  '''
  A vehicle dynamics simulator using a linear tire model.
  Modified Code from:
    https://github.com/MPC-Berkeley/genesis_path_follower/blob/master/scripts/vehicle_simulator.py
    https://github.com/urosolia/RacingLMPC/blob/master/src/fnc/SysModel.py
  '''
  def __init__(self):
    # Hyundai Genesis Parameters from HCE:
    self.param = HyundaiGenesisParameters()

    self.state = StateSE3()
    self.state.pose = PoseSE3()
    self.state.pose.position = torch.tensor([[0.0,0.0,0.0]])
    self.state.pose.orientation = torch.eye(3).unsqueeze(0)
    self.state.twist = TwistSE3()
    self.state.twist.linear = torch.tensor([[0.0,0.0,0.0]])
    self.state.twist.angular = torch.tensor([[0.0,0.0,0.0]])

    self.input = AckermannInput()
    self.input.acceleration = torch.tensor([[0.0]])
    self.input.steering_angle = torch.tensor([[0.0]])

    self.dt = 0.01 # vehicle model update period (s) and frequency (Hz)
    self.hz = int(1.0/self.dt)

    self.desired_acceleration = 0.0 # m/s^2
    self.desired_steering_angle = 0.0 # rad

  @property
  def ctrl_size(self):
    return 2

  @property
  def state_size(self):
    return 6

  def setInput(self, acc, steer):
    self.desired_acceleration = acc
    self.desired_steering_angle = steer

  def fu_func(self, X, U):
    state, inp = StateAsArray().deserialize(X)
    da, ds = self.controlDelay(self.dt, inp, U[:, 0], U[:, 1])
    inp.acceleration += da
    inp.steering_angle += ds

    a, s = inp.acceleration, inp.steering_angle
    m, Iz, lf, lr = self.param.mass, self.param.Iz, self.param.lf, self.param.lr
    vx, vy, w = (state.twist.linear[:, 0], state.twist.linear[:, 1], state.twist.angular[:, 2])

    # Compute tire slip angle and lateral force at front and rear tire (linear model)
    Fyf, Fyr = self.tireLateralForce(state, inp)
    dX = torch.zeros_like(X)
    dX[:, :2] = dpos = state.pose.orientation.bmm(state.twist.linear.unsqueeze(-1)).squeeze(-1)[:, :2]
    dX[:, 2]  = dori = state.twist.angular[:, 2]
    dX[:, 3]  = dvx  = (a - 1.0/m*Fyf*torch.sin(s) + w*vy)
    dX[:, 4]  = dvy  = (1.0/m*(Fyf*torch.cos(s) + Fyr) - w*vx)
    dX[:, 5]  = dw   = 1.0/Iz*(lf*Fyf*torch.cos(s) - lr*Fyr)
    dX[:, 6]  = da
    dX[:, 7]  = ds
    return dX


  def updateModel(self, disc_steps = 10):
    deltaT = self.dt/disc_steps
    U = torch.tensor([[self.desired_acceleration, self.desired_steering_angle]])
    da, ds = self.controlDelay(self.dt, self.input, U[:, 0], U[:, 1])
    self.input.acceleration += da
    self.input.steering_angle += ds
    for i in range(disc_steps):
      R = self.state.pose.orientation
      linear_twist = self.state.twist.linear
      X = StateAsArray().serialize(self.state, self.input)
      dX = self.fu_func(X, U)
      # discretized SE(3) dynamics
      dstate, dinp = StateAsArray().deserialize(dX * deltaT)
      # self.state.pose.orientation @ self.state.twist.linear
      self.state.pose.position += dstate.pose.position
      self.state.pose.orientation = self.state.pose.orientation @ dstate.pose.orientation
      self.state.twist.linear += dstate.twist.linear
      self.state.twist.angular +=  dstate.twist.angular


  def tireLateralForce(self, state, inp):
    ''' compute tire slip angle and lateral force '''
    alpha_f, alpha_r = 0.0, 0.0
    vx, vy, w = state.twist.linear[:, 0], state.twist.linear[:, 1], state.twist.angular[:, 2]
    if torch.abs(vx) > 1.0: # np.fabs
      alpha_f = inp.steering_angle - torch.atan2(vy + self.param.lf*w, vx)
      alpha_r = -torch.atan2(vy - self.param.lr*w, vx)
    return self.param.C_alpha_f * alpha_f, self.param.C_alpha_r * alpha_r


  def controlDelay(self, dt, inp, acc, steer):
    ''' Simulate first order control delay in acceleration/steering '''
    # e_<n> = self.<n> - self.<n>_des
    # d/dt e_<n> = - kp * e_<n>
    # or kp = alpha, low pass filter gain
    # kp = alpha = discretization time/(time constant + discretization time)
    #ad = self.desired_acceleration
    #sd = self.desired_steering_angle
    ad = acc
    sd = steer
    atc = self.param.acceleration_time_constant
    stc = self.param.steering_angle_time_constant
    da = dt/(dt + atc) * (ad - inp.acceleration)
    ds = dt/(dt + stc) * (sd - inp.steering_angle)
    return da, ds


if __name__=='__main__':
  print("Hello World!")
  HGDM = HyundaiGenesisDynamicsModel()
  HGDM.setInput(1.0,0.1)
  for k in range(5):
    HGDM.updateModel()
    print(HGDM.state.pose.position)
  print("That's all folks.")







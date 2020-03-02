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

  def inc_control(self, dU):
    self.acceleration += dU[:, 0]
    self.steering_angle = torch.atan2(
      torch.sin(self.steering_angle) + dU[:, 2],
      torch.cos(self.steering_angle) + dU[:, 1])

  def set_control(self, U):
    self.acceleration = U[:, 0]
    self.steering_angle = torch.atan2(U[:, 2], U[:, 1])

  def control(self):
    sta = self.steering_angle.unsqueeze(-1)
    return torch.cat([
      self.acceleration.unsqueeze(-1), torch.cos(sta), torch.sin(sta)
    ], dim=-1)


class HyundaiGenesisParameters(AckermannParameters):
  def __init__(self):
    self.lf = 1.5213 # m (CoG to front axle)
    self.lr = 1.4987 # m (CoG to rear axle)
    self.half_width = 0.945	 # m (half-width)
    self.mass = 2303.1 # kg (vehicle mass)
    self.Iz = 5520.1 # kg*m2 (vehicle inertia)
    self.C_alpha_f = 7.6419e4*2  # N/rad	(front axle cornering stiffness) # 200k
    self.C_alpha_r = 13.4851e4*2	# N/rad	(rear axle cornering stiffness)	 # 250k
    self.acceleration_time_constant = atc = 0.4 # s
    self.steering_angle_time_constant  = sac = 0.1 # s
    self.control_time_constant = torch.tensor([[atc, math.cos(sac), math.sin(sac)]])


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
    U = inp.control()
    return torch.cat((pos, ori.unsqueeze(-1), v, w, U), dim=-1)

  def deserialize(self, X):
    (pos, ori, v, w, U) = (
      X[:, :2], X[:, 2], X[:, 3:5], X[:, 5], X[:, 6:9])

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
    inp.set_control(U)
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
    self.input.acceleration = torch.tensor([0.0])
    self.input.steering_angle = torch.tensor([0.0])

    self.dt = 0.01 # vehicle model update period (s) and frequency (Hz)
    self.hz = int(1.0/self.dt)

    self.desired_acceleration = 0.0 # m/s^2
    self.desired_steering_angle = 0.0 # rad

  @property
  def ctrl_size(self):
    return 3

  @property
  def state_size(self):
    return 9

  def setInput(self, acc, steer):
    self.desired_acceleration = acc
    self.desired_steering_angle = steer

  def _fg_func(self, X_in):
    if X_in.ndim == 1:
      X = X_in.unsqueeze(0)
    else:
      X = X_in

    assert X.shape[-1] == self.state_size
    state, inp = StateAsArray().deserialize(X)
    m, Iz, lf, lr = self.param.mass, self.param.Iz, self.param.lf, self.param.lr
    vx, vy, w = (state.twist.linear[:, 0], state.twist.linear[:, 1],
                 state.twist.angular[:, 2])

    # Compute tire slip angle and lateral force at front and rear tire (linear model)
    Fyf, Fyr = self.tireLateralForce(state, inp)
    fX = torch.zeros_like(X)
    gX = X.new_zeros(X.shape[0], X.shape[-1], self.ctrl_size)
    fX[:, :2] = dpos = state.pose.orientation.bmm(
      state.twist.linear.unsqueeze(-1)).squeeze(-1)[:, :2]
    fX[:, 2]  = dori = state.twist.angular[:, 2]

    #dX[:, 3]  = dvx  = (a - 1.0/m*Fyf*sins + w*vy)
    #dX[:, 4]  = dvy  = (1.0/m*(Fyf*coss + Fyr) - w*vx)
    #dX[:, 5]  = dw   = 1.0/Iz*(lf*Fyf*coss - lr*Fyr)

    gX[:, 3, :], fX[:, 3] = torch.tensor([1, 0, - 1.0/m*Fyf]), w*vy
    gX[:, 4, :], fX[:, 4] = torch.tensor([0, 1.0/m*Fyf, 0]), 1.0/m*Fyr - w*vx
    gX[:, 5, :], fX[:, 5] = torch.tensor([0, 1.0/Iz*lf*Fyf, 0]), - 1.0/Iz*lr*Fyr
    gX[:, 6:9, :] = torch.eye(self.ctrl_size)
    if X_in.ndim == 1:
      fX = fX.squeeze(0)
      gX = gX.squeeze(0)
    return fX, gX

  def fu_func(self, X, U):
    assert U.shape[-1] == self.ctrl_size
    _, inp = StateAsArray().deserialize(X)
    dU = self.controlDelay(self.dt, inp, U)
    inp.inc_control(dU)

    Ut = inp.control()
    fX, gX = self._fg_func(X)
    return fX + gX.bmm(Ut.unsqueeze(-1)).squeeze(-1)

  def f_func(self, X):
    return self._fg_func(X)[0]

  def g_func(self, X):
    return self._fg_func(X)[1]

  def updateModel(self, disc_steps = 10):
    deltaT = self.dt / disc_steps
    U = torch.tensor([[
      self.desired_acceleration,
      math.cos(self.desired_steering_angle),
      math.sin(self.desired_steering_angle)]])
    #U = torch.tensor([[self.desired_acceleration, self.desired_steering_angle]])
    dU = self.controlDelay(self.dt, self.input, U)
    self.input.inc_control(dU)
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


  def controlDelay(self, dt, inp, dctrl):
    ''' Simulate first order control delay in acceleration/steering '''
    # e_<n> = self.<n> - self.<n>_des
    # d/dt e_<n> = - kp * e_<n>
    # or kp = alpha, low pass filter gain
    # kp = alpha = discretization time/(time constant + discretization time)
    #ad = self.desired_acceleration
    #sd = self.desired_steering_angle
    ad = dctrl[:, 0]
    sd = torch.atan2(dctrl[:, 2], dctrl[:, 1])
    atc = self.param.acceleration_time_constant
    stc = self.param.steering_angle_time_constant
    ctc = self.param.control_time_constant
    da = dt/(dt + atc) * (ad - inp.acceleration)
    ds = dt/(dt + stc) * (sd - inp.steering_angle)
    #du = dt/(dt + ctc) * (dctrl  - inp.control())
    inp = AckermannInput()
    inp.acceleration = da
    inp.steering_angle = ds
    return inp.control()


if __name__=='__main__':
  print("Hello World!")
  HGDM = HyundaiGenesisDynamicsModel()
  HGDM.setInput(1.0,0.1)
  for k in range(5):
    HGDM.updateModel()
    print(HGDM.state.pose.position)
  print("That's all folks.")







import os
import os.path as osp
from functools import partial
import math
import torch

import pybullet
from pybullet_envs.bullet.racecar import Racecar
from pybullet_utils import bullet_client
import pybullet_data
from pkg_resources import resource_filename
import em # pip install empy
from contextlib import contextmanager
import tempfile

from bayes_cbf.sampling import DynamicsModel

@contextmanager
def empy_expanded_file(template, **kw):
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(
            em.expand(open(template).read(), **kw)
            .encode('utf-8')
        )
    yield tf.name
    os.unlink(tf.name)


class BulletBody:
    def __init__(self, bullet_client, bullet_id):
        self.bullet_client = bullet_client
        self.bullet_id = bullet_id

    def __getattr__(self, name):
        return partial(
            getattr(self.bullet_client, name),
            self.bullet_id)

class RacecarEnv(Racecar, DynamicsModel):
    def __init__(self):
        self.bullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        self._urdfRoot = pybullet_data.getDataPath()
        self._timestep = 0.01

        self.bullet_client.resetSimulation()
        Racecar.__init__(self,
          self.bullet_client,
          urdfRootPath=self._urdfRoot,
          timeStep=self._timestep)
        self.racecar = BulletBody(self.bullet_client,
                                  self.racecarUniqueId)

        self._state = None

        self.bullet_client.setTimeStep(self._timestep)

        self._initalize_stadium()
        self._adjust_car()
        self.bullet_client.setGravity(0,0,-10)

    @property
    def ctrl_size(self):
        return 2

    @property
    def state_size(self):
        return 3

    def _initalize_stadium(self):
        self._stadiumobjects = map(
            partial(BulletBody, self.bullet_client),
            self._p.loadSDF(
                osp.join(pybullet_data.getDataPath(),
                         "plane_stadium.sdf")))

    def _adjust_car(self):
        pos, orn = self.racecar.getBasePositionAndOrientation()
        newpos = [pos[0], pos[1], pos[2]+0.1]
        self.racecar.resetBasePositionAndOrientation(newpos, orn)

    def _initialize_obstacles(self):
        self._obstacles = []
        for i, (oc, orad) in enumerate(
                zip(self.obstacle_centers,
                    self.obstacle_radii)):
            self._obstacles.append(
                self.create_new_obstacle('obstacle_%d' % i, oc, orad)
                )

    def create_new_obstacle(self, name, center, radius):
        obstacle_model = resource_filename("bayes_cbf", "bulletcar/data/obstacle/model.sdf.empy")
        bc = self.bullet_client
        with empy_expanded_file(obstacle_model,
                                MODEL_NAME=name,
                                RADIUS=float(radius)) as ef:
            oid, = bc.loadSDF(ef)
            obstacle = BulletBody(self.bullet_client, oid)

        pos, orn = obstacle.getBasePositionAndOrientation()
        newpos = [center[0], center[1], pos[2]]
        obstacle.resetBasePositionAndOrientation(newpos, orn)
        return obstacle

    @staticmethod
    def _3D_to_2D(state3D):
        qx, qy, qz, qw = state3D[3:]
        sin_half_theta = math.sqrt(qx*qx + qy*qy+ qz*qz)
        theta = 2*math.atan2(sin_half_theta, qw)
        return torch.tensor(state3D[:2] + [theta])

    @staticmethod
    def _2D_to_3D(ref3D, x0):
        pos, orn = ref3D
        newpos = [x0[0], x0[1], pos[2]+0.1]
        orn = [0, 0, math.sin(x0[2]/2), math.cos(x0[2]/2)]
        return [newpos, orn]


    def get2DObs(self):
        return self._3D_to_2D(self.getObservation())

    def step(self, action, dt):
        #action = [max(min(a, 1), -1) for a in action]
        old_state = self.get2DObs()
        self.bullet_client.setTimeStep(dt)
        self.applyAction(action)
        self.bullet_client.stepSimulation()
        state = self.get2DObs()
        return dict(x=state,
                    xdot=((state - old_state) / dt))

    def set_init_state(self, x0):
        ref3D = self.racecar.getBasePositionAndOrientation()
        newpos, orn = self._2D_to_3D(ref3D, x0)
        self.racecar.resetBasePositionAndOrientation(newpos, orn)
        for _ in range(20):
            self.bullet_client.stepSimulation()


def main():

    environment = RacecarEnv()

    targetVelocitySlider = environment._p.addUserDebugParameter("wheelVelocity", -1, 1, 0)
    steeringSlider = environment._p.addUserDebugParameter("steering", -1, 1, 0)

    while (True):
        targetVelocity = environment._p.readUserDebugParameter(targetVelocitySlider)
        steeringAngle = environment._p.readUserDebugParameter(steeringSlider)
        action = [targetVelocity, steeringAngle]
        environment.step(action, 0.01)


if __name__ == "__main__":
    main()

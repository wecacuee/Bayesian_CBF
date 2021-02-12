import os

import pybullet
from pybullet_envs.bullet.racecar import Racecar as _Racecar
from pybullet_utils import bullet_client as bc
import pybullet_data
from pkg_resources import parse_version

class Racecar(_Racecar):
    def __init__(self):
        self.bullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        self._urdfRoot = pybullet_data.getDataPath()
        self._timestep = 0.01

        self.bullet_client.resetSimulation()
        super().__init__(
          self.bullet_client,
          urdfRootPath=self._urdfRoot,
          timeStep=self._timestep)

        self.bullet_client.setTimeStep(self._timestep)

        self.stadiumobjects = self._p.loadSDF(os.path.join(self._urdfRoot, "plane_stadium.sdf"))
        self.bullet_client.setGravity(0, 0, -9.8)
        pos, orn = self.bullet_client.getBasePositionAndOrientation(
          self.racecarUniqueId)
        newpos = [pos[0], pos[1], pos[2]+0.1]
        self.bullet_client.resetBasePositionAndOrientation(
          self.racecarUniqueId, newpos, orn)


def main():

    environment = Racecar()

    targetVelocitySlider = environment._p.addUserDebugParameter("wheelVelocity", -1, 1, 0)
    steeringSlider = environment._p.addUserDebugParameter("steering", -1, 1, 0)

    while (True):
        targetVelocity = environment._p.readUserDebugParameter(targetVelocitySlider)
        steeringAngle = environment._p.readUserDebugParameter(steeringSlider)
        action = [targetVelocity, steeringAngle]
        environment.applyAction(action)
        environment.bullet_client.stepSimulation()
        obs = environment.getObservation()
        print("obs")
        print(obs)


if __name__ == "__main__":
    main()

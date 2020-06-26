import numpy as np
import vtkplotter as vplt; vplt.embedWindow(False)

import pkg_resources

def absfilepath(relpath):
  return pkg_resources.resource_filename(__package__ or 'bayes_cbf.car', relpath)


class CarWorld():
  def __init__(self):
    self.wall = vplt.load(absfilepath("data/walls.stl")).c("green").alpha(0.2).addShadow(z=-0.5)
    self.obs = vplt.load(absfilepath("data/obs.stl")).c("red").addShadow(z=-0.5)
    self.lexus = vplt.load(absfilepath("data/lexus_hs.obj")
    ).scale([0.01,0.01,0.01]
    ).texture(absfilepath("data/lexus_hs_diffuse.jpg")).rotateZ(180)
    self.plotter = vplt.Plotter(bg='white', axes={'xyGrid':True, 'zxGrid2':True,'showTicks':True})

  def setCarPose(self,x,y,theta):
    self.lexus.pos(x=x,y=y,z=0.0).orientation(newaxis=[0,0,1.0],rotation=theta, rad=True)

  def show(self):
    vplt.show( self.wall, self.obs, self.lexus, interactive=0,
               camera={'pos':[-0.114, -21.332, 35.687],
                       'focalPoint':[9.611, 2.363, 0.07],
                       'viewup':[0.267, 0.767, 0.583],
                       'distance':43.871,
                       'clippingRange':[33.465, 57.074]} )

  def close(self):
    vplt.closeWindow()


class CarWithObstacles():
  def __init__(self):
    self.lexus = vplt.load(absfilepath("data/lexus_hs.obj")
    ).scale([0.01,0.01,0.01]
    ).texture(absfilepath("data/lexus_hs_diffuse.jpg")).rotateZ(180)
    self.plotter = vplt.Plotter(bg='white', axes={'xyGrid':True, 'zxGrid2':True,'showTicks':True})
    self.goal = vplt.Sphere(pos=(0,0,0.2), r=0.2, c="gold", alpha=0.3)
    self.obstacles = []

  def setGoal(self, x, y):
      self.goal = vplt.Sphere(pos=(x,y,0.2), r=0.2, c="gold", alpha=0.3)

  def addObstacle(self, x, y, radius):
      self.obstacles.append(
          vplt.Cylinder(pos=(x,y,0.5),
                        height=1,
                        axis=vplt.vector(0, 0, 1),
                        r=0.8, c="dg"))


  def setCarPose(self,x,y,theta):
    self.lexus.pos(x=x,y=y,z=0.0).orientation(newaxis=[0,0,1.0],rotation=theta, rad=True)

  def show(self):
    vplt.show( self.goal, *self.obstacles, self.lexus, interactive=0)
               # camera={'pos':[0, 0, 10],
               #         'focalPoint':[0, 0, 10],
               #         'viewup':[0, 0, -1],
               #         'distance':43.871,
               #         'clippingRange':[33.465, 57.074]} )

  def close(self):
    vplt.closeWindow()


if __name__ == '__main__':
    viewer = CarWithObstacles()
    for x, y in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
        viewer.addObstacle(x, y, 0.8)
    N = 500
    pose = np.array([2, 0, -np.pi])
    step = (np.array([0, 0, 0]) - pose) / N
    step[2] = 0
    for k in range(N):
        viewer.setCarPose(*(pose + k*step))
        viewer.show()
    vplt.interactive()
    viewer.close()

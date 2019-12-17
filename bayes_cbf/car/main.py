import vtkplotter as vplt; vplt.embedWindow(False)
#import numpy as np
import torch

if __name__ == '__main__':
  from HyundaiGenesis import HyundaiGenesisDynamicsModel
else:
  from .HyundaiGenesis import HyundaiGenesisDynamicsModel
#import importlib
#import time
import pkg_resources


def absfilepath(relpath):
  return pkg_resources.resource_filename(__package__ or 'bayes_cbf.car', relpath)


class CarWorld():
  def __init__(self):
    self.wall = vplt.load(absfilepath("data/walls.stl")).c("grey").alpha(0.2).addShadow(z=-0.5)
    self.obs = vplt.load(absfilepath("data/obs.stl")).c("grey").addShadow(z=-0.5)
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


if __name__=='__main__':
  print("Hello World!")

  myViewer = CarWorld()
  HGDM = HyundaiGenesisDynamicsModel()
  HGDM.state.pose.position = torch.tensor([[1.9,2.5,0.0]])
  theta = torch.atan2(HGDM.state.pose.orientation[0, 1,0], HGDM.state.pose.orientation[0, 0,0])
  myViewer.setCarPose(HGDM.state.pose.position[0, 0], HGDM.state.pose.position[0, 1], theta)
  myViewer.show()

  accel, steer = 2, 0.2
  for k in range(500):
    HGDM.setInput(accel,steer)
    HGDM.updateModel()
    theta = torch.atan2(HGDM.state.pose.orientation[0, 1, 0],
                        HGDM.state.pose.orientation[0, 0, 0])
    car_pose = (HGDM.state.pose.position[0, 0].item(),
                HGDM.state.pose.position[0, 1].item(),
                theta.item())
    print(car_pose)
    myViewer.setCarPose(*car_pose)
    myViewer.show()
    if k > 200:
      steer = -0.1
    if k > 400:
      steer = 0.1
    #print('k = %d'%k)
    #time.sleep(HGDM.dt)

  myViewer.close()

  print("That's all folks.")





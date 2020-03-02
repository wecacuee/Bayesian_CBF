import torch
if __name__ == '__main__':
  from HyundaiGenesis import HyundaiGenesisDynamicsModel
  from vis import CarWorld
else:
  from .HyundaiGenesis import HyundaiGenesisDynamicsModel
  from .vis import CarWorld



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





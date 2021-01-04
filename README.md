# Control Barriers in Bayesian Learning of System Dynamics
![Python package](https://github.com/wecacuee/Bayesian_CBF/workflows/Python%20package/badge.svg)

[Website](https://vikasdhiman.info/Bayesian_CBF/)

## Demo

![](./saved-runs/unicycle_move_to_pose_fixed_mean_cbf_collides_1209-1257/animation.gif)

## Installation

1. If you are using a Python virtual environment, edit setup.bash to point to the
  activate script of the virtual environment. One way to do that is:

  ``` shellsession
  python3_ver () 
  { 
      python3 -V 2>&1 | sed -e 's/.*3.\([0-9]\).*/py3\1/'
  }
  mkdir -p .tox/
  virtualenv --python=python3 .tox/$(python3_ver)
  ```


2. Install gurobi. Edit setup.bash to set GUROBI_LIB_PATH

3. Activate environment and install current package in edit mode

  ``` shellsession
  source setup.bash
  pip install -e .
  ```

## Run tests

``` shellsession
pytest
```

## Unicycle demos

1. To run experiment where unicycle with mean CBF collides with the obstacle run

   ```shellsession
   python -c 'from bayes_cbf.unicycle_move_to_pose import unicycle_mean_cbf_collides_obstacle; unicycle_mean_cbf_collides_obstacle()'
   ```

   ![](./saved-runs/unicycle_move_to_pose_fixed_mean_cbf_collides_v1.2.3/animation.gif)
   
2. To run experiment where unicycle with Bayes CBF drives safely between the obstacles

   ```shellsession
   python -c 'from bayes_cbf.unicycle_move_to_pose import unicycle_bayes_cbf_safe_obstacle; unicycle_bayes_cbf_safe_obstacle()'
   ```

   ![](./saved-runs/unicycle_move_to_pose_fixed_mean_cbf_collides_1209-1257/animation.gif)
   
3. To run experiment where unicycle gets stuck without learning run

   ```shellsession
   python -c 'from bayes_cbf.unicycle_move_to_pose import unicycle_no_learning_gets_stuck; unicycle_no_learning_gets_stuck()'
   ```

   ![](./saved-runs/unicycle_move_to_pose_fixed_no_learning_gets_stuck_v1.2.3/animation.gif)
   
4. To run experiment where unicycle passes safely through obstacles due to learning run

   ```shellsession
   python -c 'from bayes_cbf.unicycle_move_to_pose import unicycle_learning_helps_avoid_getting_stuck; unicycle_learning_helps_avoid_getting_stuck()'
   ```

   ![](./saved-runs/unicycle_move_to_pose_fixed_learning_helps_avoid_getting_stuck_v1.2.3/animation.gif)

## Pendulum demos

1. A script to run pendulum example with random controller (No learning)
  ``` shellsession
  run_pendulum_control_trivial
  ```

2. A script to run pendulum example with CBF-CLF-controller (No learning)
  ``` shellsession
  run_pendulum_control_cbf_clf
  ```

3. A script to run Bayesian learning on pendulum example with random controller.
  ``` shellsession
  pendulum_learn_dynamics
  ```

4. A script to run Bayesian learning on pendulum example with CBF-CLF
   controller.
  ``` shellsession
  pendulum_control_online_learning
  ```

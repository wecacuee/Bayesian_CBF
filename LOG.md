# Dec 02

## 13:57
Ran bayes_cbf.pendulum:learn_dynamics()
Logs data/runs/learn_dynamics_v1.0.0-7-g718f8a5/events.out.tfevents.1606945888.dwarf.25367.0

Plots are okay, but need to plot from the logs.
Also need to compare with other learning methods.

## 14:07
Saved logs were bad because fx and gx data were saved as same name; overridden.

Re-ran bayes_cbf.pendulum:learn_dynamics()
Too much spread
Logs: data/runs/learn_dynamics_v1.0.1-1-g97c87d8/events.out.tfevents.1606946860.dwarf.27863.0

Did not save Xtrain. Working around that.

## 15:12

Moved from plotting online to logging and then plotting.

Successful log
data/runs/learn_dynamics_v1.0.2/events.out.tfevents.1606951767.dwarf.13002.0
Move to saved-runs/learn_dynamics_v1.0.2/events.out.tfevents.1606951767.dwarf.13002.0
Copied to overleaf with script.

# Dec 03

## 11:43

1. Compute ControlAffineRegressor with a different kernel.

# Dec 07

## 11:45

1. Decided not to further simplify Independent GPs



# Dec 16

## 20:38

1. Why is decoupled GP slower than MVGP

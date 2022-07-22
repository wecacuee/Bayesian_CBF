try:
    from bayes_cbf.pendulum import learn_dynamics_matrix_vector_independent_vis
except ImportError:
    import subprocess
    subprocess.run("pip install git+ssh://git@github.com/wecacuee/BayesCBF.git@v1.1.1".split())
    from bayes_cbf.pendulum import learn_dynamics_matrix_vector_independent_vis

learn_dynamics_matrix_vector_independent_vis(events_file='events.out.tfevents.1607382261.dwarf.5274.7')

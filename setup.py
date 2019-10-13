import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class Pytest(TestCommand):
    def run_tests(self):
        import pytest
        errno = pytest.main(["tests"])
        sys.exit(errno)


setup(name="bayes_cbf",
      packages=find_packages(),
      tests_require=['pytest', 'scipy'],
      cmdclass = {'test': Pytest},
      install_requires=['matplotlib', 'cvxopt',
                        #'gpytorch==fractional-outputs-per-inputs',
                        'gpytorch @ git+https://github.com/wecacuee/gpytorch.git@fractional-outputs-per-input',
                        'torch',
                        'torch-vision', 'pyro-ppl'],
      entry_points={
          'console_scripts': [
              'run_pendulum_control_trival = bayes_cbf.pendulum:run_pendulum_control_trival',
              'run_pendulum_control_cbf_clf = bayes_cbf.pendulum:run_pendulum_control_cbf_clf',
              'pendulum_learn_dynamics = bayes_cbf.pendulum:learn_dynamics',
              'pendulum_control_online_learning = bayes_cbf.pendulum:run_pendulum_control_online_learning'

          ]}
)

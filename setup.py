import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class Pytest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]
    def run_tests(self):
        import pytest
        errno = pytest.main(["tests"])
        sys.exit(errno)


setup(name="bayes_cbf",
      packages=find_packages(),
      tests_require=['pytest', 'scipy'],
      cmdclass = {'test': Pytest},
      install_requires=['matplotlib', 'cvxopt', 'gpytorch', 'torch',
                        'torch-vision', 'pyro-ppl'])

from setuptools import setup, find_packages

setup(name="bayes_cbf",
      packages=find_packages(),
      install_requires=['matplotlib', 'cvxopt', 'gpytorch', 'torch',
                        'torch-vision', 'pyro-ppl'])

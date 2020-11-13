from setuptools import find_packages
from setuptools import setup

setup(name='mb_modelbase',
      version='0.9.1',
      description='A webservice and python backend for SQL-like queries to data and probabilistic models',
      url='https://github.com/lumen-org/lumen',
      author='Philip Lucas',
      author_email='philipp.lucas@dlr.de',
      license='lgpl-3.0',
      packages=find_packages(exclude=['bin']),
      install_requires=[
          'anytree', #84 requires scipy==1.2, https://github.com/lumen-org/modelbase/issues/84
          'astropy',
          'Cython',
          'dill',
          'flask>=1.1.1'
          'flask-cors',
          'flask-socketio',
          'graphviz',
          'multiprocessing_on_dill',
          'networkx',
          'numba',
          'numpy==1.18.4',
          'pandas>=0.24',
          'prettytable',
          'pymc3',
          'pymemcache',
          'pyopenssl',
          'rpy2>=2.9.4',
          'scikit-learn',
          'scipy==1.2',
          'spflow>=0.0.39',
          'statsmodels',
          'sympy',
          'tensorflow==2.3.1',
          'watchdog',
          'xarray',
      ],
      zip_safe=False)


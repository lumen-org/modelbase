from setuptools import find_packages
from setuptools import setup

setup(name='mb_modelbase',
      version='0.9.1',
      description='A webservice and python backend for SQL-like queries to data and probabilistic models',
      url='https://github.com/lumen-org/lumen',
      author='Philip Lucas',
      author_email='philipp.lucas@dlr.de',
      license='lgpl-3.0',
      packages=find_packages(exclude=['scripts']),
      install_requires=[
          'anytree',
          'xarray',
          'numpy==1.18.4'
          'pandas>=0.24',
          'scikit-learn',
          'scipy>=0.19.1', #84 requires scipy==1.2, https://github.com/lumen-org/modelbase/issues/84
          'flask>=1.1.1',
          'flask-cors',
          'graphviz',
          'multiprocessing_on_dill',
          'pyopenssl',
          'spflow',
          'rpy2>=2.9.4',
          'spflow>=0.0.39',
          'pymc3',
          'dill',
          'sympy',
          'dill',
          'pymemcache',
          'watchdog',
          'flask-socketio',
          'astropy',
          'networkx',
          'numba',
          'statsmodels',
          'tensorflow==1.14',
          'prettytable'
      ],
      zip_safe=False)


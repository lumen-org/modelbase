from setuptools import setup
from setuptools import find_packages

setup(name='mb_modelbase',
      version='0.9',
      description='A webservice/python backend for SQL-like queries to data and models based on the data',
      url='https://github.com/lumen-org/lumen',
      author='Philip Lucas',
      author_email='philipp.lucas@uni-jena.de',
      license='lgpl-3.0',
      packages=find_packages(exclude=['scripts']),
      install_requires=[
          'xarray',
          #'numpy<=1.17',
          'numpy<=1.18.4'
          'pandas>=0.24',
          'scikit-learn',
          'scipy>=0.19.1',
          'flask>=1.1.1',
          'flask-cors',
          'graphviz',
          'multiprocessing_on_dill',
          'pyopenssl',
          'spflow',
          'rpy2>=2.9.4',
          'pymc3',
          'dill',
          'sympy',
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


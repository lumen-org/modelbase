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
          'numpy',
          'pandas>=0.24',
          'scikit-learn',
          'scipy==1.2',  # see issue #84
          'flask>=1.1.1',
          'flask-cors',
          'graphviz',
          'multiprocessing_on_dill',
          'pyopenssl',
          'spflow>=0.0.39',
          'pymc3',
          'pyopenssl',
          'sympy',
          'dill',
          'pymemcache',
          'watchdog',
          'flask-socketio'
      ],
      zip_safe=False)


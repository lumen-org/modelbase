from setuptools import setup, find_namespace_packages

setup(name='mb-modelbase',
      version='0.9.1',
      description='A webservice and python backend for SQL-like queries to data and probabilistic models',
      url='https://github.com/lumen-org/modelbase',
      author='Philip Lucas',
      author_email='philipp.lucas@dlr.de',
      license='lgpl-3.0',
      keywords='visualization, data exploration, model exploration, model criticism,'
               ' probabilistic programming',
      packages=find_namespace_packages(),
      project_urls={
          'Source': 'https://github.com/lumen-org/modelbase',
          'Tracker': 'https://github.com/lumen-org/modelbase/issues',
      },
      install_requires=[
          'Cython',
          'dill',
          'flask>=1.1.1',
          'flask-cors',
          'flask-socketio',
          'multiprocessing_on_dill',
          'numpy',
          'pandas>=0.24',
          'prettytable',
          'pymemcache',
          'pyopenssl',
          'scikit-learn',
          'scipy',
          'sympy',
          'watchdog',
          'wheel',
          'xarray',
      ],
      zip_safe=False)

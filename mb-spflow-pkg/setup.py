from setuptools import setup, find_namespace_packages

setup(name='mb-spflow',
      version='0.0.1',
      description='spflow adapter to Lumens modelbase',
      author='Philip Lucas, Christian Lengert',
      author_email='philipp.lucas@dlr.de, christian.lengert@dlr.de',
      packages=find_namespace_packages(),
      install_requires=[
          # 'anytree', #84 requires scipy==1.2, https://github.com/lumen-org/modelbase/issues/84
          # 'astropy',
          # 'Cython',
          # 'dill',
          # 'flask>=1.1.1'
          # 'flask-cors',
          # 'flask-socketio',
          # 'graphviz',
          # 'multiprocessing_on_dill',
          # 'networkx',
          # 'numba',
          # 'numpy==1.18.4',
          # 'pandas>=0.24',
          # 'prettytable',
          # 'pymc3',
          # 'pymemcache',
          # 'pyopenssl',
          # 'rpy2>=2.9.4',
          # 'scikit-learn',
          # 'scipy==1.2',
          # 'spflow>=0.0.39',
          # 'statsmodels',
          # 'sympy',
          # 'tensorflow==1.14',
          # 'watchdog',
          # 'xarray',
      ],
      zip_safe=False)


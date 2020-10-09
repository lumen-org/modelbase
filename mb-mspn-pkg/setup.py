from setuptools import setup, find_namespace_packages

setup(name='mb-mspn',
      version='0.0.5',
      description='mspn model adapter for Lumens modelbase',
      author='Julien Klaus, Philip Lucas',
      author_email='julien.klaus@uni-jena.de, philipp.lucas@dlr.de',
      packages=find_namespace_packages(),
      install_requires=[
          'numpy',  # maybe requires 'numpy==1.18.4' ???
          'pandas',
          'scipy',
          'mb.modelbase',
          'scikit-learn',
          'numba',
          'networkx',
          'rpy2>=2.9.4',
          'astropy',
          'tensorflow',  # maybe requires 'tensorflow==1.14' ???
          'jsonpickle',
          'mpmath',
          'statsmodels',
          'joblib',
      ],
      zip_safe=False)

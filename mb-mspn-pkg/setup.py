from setuptools import setup, find_namespace_packages

setup(name='mb-mspn',
      version='0.0.5',
      description='mspn model adapter for Lumens modelbase',
      author='Julien Klaus, Philip Lucas',
      author_email='julien.klaus@uni-jena.de, philipp.lucas@dlr.de',
      packages=find_namespace_packages(),
      install_requires=[
            'numpy',
            'pandas',
            'scipy',
            'mb.modelbase',
            'scikit-learn',
            'numba',
            'networkx',
            'rpy2',
            'astropy',
            'tensorflow',
            'jsonpickle',
            'mpmath',
            'statsmodels'
            #'h2o'
      ],
      zip_safe=False)

from setuptools import setup, find_namespace_packages

setup(name='mb-mspn',
      version='0.0.5',
      description='mspn model adapter for Lumens modelbase',
      author='Philip Lucas',
      author_email='philipp.lucas@dlr.de',
      packages=find_namespace_packages(),
      install_requires=[
            'numpy',
            'pandas',
            'scipy',
            'mb.modelbase',
            'sklearn',
            'numba',
            'networkx',
            'rpy2',
            'astropy',
            'tensorflow'
      ],
      zip_safe=False)

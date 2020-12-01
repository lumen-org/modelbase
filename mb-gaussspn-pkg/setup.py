from setuptools import setup, find_namespace_packages

setup(name='mb-gaussspn',
      version='0.0.1',
      description='gauss-spn adapter to Lumens modelbase',
      author='Julien Klaus, Philipp Lucas',
      author_email='julien.klaus@uni-jena.de, philipp.lucas@dlr.de',
      packages=find_namespace_packages(),
      install_requires=[
          'mb-modelbase',
          'numpy',  # maybe requires 'numpy==1.18.4' ???
          'scipy',
          'graphviz',
      ],
      namespace_packages=['mb'],
      zip_safe=False)

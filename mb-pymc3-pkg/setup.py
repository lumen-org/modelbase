from setuptools import setup, find_namespace_packages

setup(name='mb-pymc3',
      version='0.2',
      description='pymc3 adapter to Lumens modelbase',
      author='Philip Lucas, Christian Lengert, Jonas Gütter, Julien Klaus',
      author_email='philipp.lucas@dlr.de, christian.lengert@dlr.de, Jonas.Aaron.Guetter@dlr.de, julien.klaus@uni-jena.de',
      packages=find_namespace_packages(),
      install_requires=[
            'mb-modelbase',
            'pymc3',
            'numpy',
            'pandas',
            'scipy',
      ],
      namespace_packages=['mb'],
      zip_safe=False)


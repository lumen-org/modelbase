from setuptools import setup, find_namespace_packages

setup(name='mb-data',
      version='0.2',
      description='some free data for building models in modelbase',
      author='Philip Lucas',
      author_email='philipp.lucas@dlr.de',
      packages=find_namespace_packages(),
      install_requires=[
            'mb.modelbase',
            'numpy',
            'pandas',
      ],
      zip_safe=False)


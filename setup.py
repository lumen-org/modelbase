from setuptools import setup
from setuptools import find_packages

setup(name='mb_modelbase',
      version='0.2',
      description='Tool for unified model and data visualization',
      url='https://bitbucket.org/phlpp/modelbase',
      author='Philip Lucas',
      author_email='philipp.lucas@uni-jena.de',
      license='Public',
      packages=find_packages(exclude=['scripts']),
      install_requires=[
          'xarray',
          'numpy',
          'pandas',
          'scikit-learn',
          'scipy',
      ],
      zip_safe=False)

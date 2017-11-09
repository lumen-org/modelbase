from setuptools import setup

setup(name='modelbase',
      version='0.1',
      description='Tool for unified model and data visualization',
      url='https://bitbucket.org/phlpp/modelbase',
      author='Philip Lucas',
      author_email='philipp.lucas@uni-jena.de',
      license='Public',
      packages=['modelbase'],
      install_requires=[
          'xarray',
          'numpy',
          'pandas',
          'scikit-learn',
          'scipy',
      ],
      zip_safe=False)

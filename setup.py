from setuptools import setup

setup(name='modelbase',
      version='0.1',
      description='Tool for data visualization',
      url='https://bitbucket.org/phlpp/modelbase',
      author='Flying Circus',
      author_email='philipp.lucas@uni-jena.de',
      license='Public',
      packages=['.'],
      install_requires=[
          'xarray',
          'numpy',
          'pandas',
          'sklearn',
          'scipy',
      ],
      zip_safe=False)

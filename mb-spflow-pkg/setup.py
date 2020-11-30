from setuptools import setup, find_namespace_packages

setup(name='mb-spflow',
      version='0.3',
      description='spflow adapter to Lumens modelbase',
      author='Philipp Lucas, Christian Lengert',
      author_email='philipp.lucas@dlr.de, christian.lengert@dlr.de',
      packages=find_namespace_packages(),
      install_requires=[
            'mb.modelbase',
            'numpy',
            'scipy==1.2', # see also: https://github.com/lumen-org/modelbase/issues/84
            'spn',
      ],
      zip_safe=False)


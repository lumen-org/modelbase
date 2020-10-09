from setuptools import setup, find_namespace_packages

setup(name='mb-spflow',
      version='0.0.1',
      description='spflow adapter to Lumens modelbase',
      author='Philip Lucas, Christian Lengert',
      author_email='philipp.lucas@dlr.de, christian.lengert@dlr.de',
      packages=find_namespace_packages(),
      install_requires=[
            'mb.modelbase',
            'numpy',
            'scipy',
            'spn',
      ],
      zip_safe=False)


from setuptools import setup, find_namespace_packages

setup(name='mb-libspn',
      version='0.0.1',
      description='libspn adapter to Lumens modelbase',
      author='Julien Klaus',
      author_email='Julien.Klaus@uni-jena.de',
      packages=find_namespace_packages(),
      install_requires=[
          'libspn',
          'tensorflow',  # maybe required 'tensorflow==1.14' ???
          'mb-modelbase',
      ],
      namespace_packages=['mb'],
      zip_safe=False)

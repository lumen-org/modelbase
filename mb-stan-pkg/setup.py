from setuptools import setup, find_namespace_packages

setup(name='mb-stan',
      version='0.0.1',
      description='stan adapter to Lumens modelbase',
      author='Christian Lengert',
      author_email='christian.lengert@dlr.de',
      packages=find_namespace_packages(),
      install_requires=[
            'mb.modelbase',
      ],
      zip_safe=False)


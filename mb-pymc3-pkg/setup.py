from setuptools import setup, find_namespace_packages
import os
import glob

# Find all csv-files recursively.
package_data_files = [_file for _path in os.walk('mb/pymc3/') \
      for _file in glob.glob(os.path.join(_path[0], '*.csv'))]
# Now every found csv-file is given by its relative path, e.g.:
# 'mb/pymc3/tests/data/*.csv'
# We need it in the following format:
# 'pymc3/tests/data/*.csv'
# Therefore remove the first three letters 'mb/'.
for i, _file in enumerate(package_data_files):
      package_data_files[i] = _file[3:]

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
      zip_safe=False,
      include_package_data=True,
      package_data={'': package_data_files})


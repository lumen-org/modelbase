from setuptools import setup, find_namespace_packages
import os
import glob

# Find all csv-files recursively.
package_data_files = [_file for _path in os.walk('mb/data/') \
      for _file in glob.glob(os.path.join(_path[0], '*.csv'))]
# Now every found csv-file is given by its relative path, e.g.:
# 'mb/data/iris/iris.csv'
# We need it in the following format:
# 'data/iris/iris.csv'
# Therefore remove the first three letters 'mb/'.
for i, _file in enumerate(package_data_files):
      package_data_files[i] = _file[3:]

setup(name='mb-data',
      version='0.2',
      description='some free data for building models in modelbase',
      author='Philip Lucas',
      author_email='philipp.lucas@dlr.de',
      packages=find_namespace_packages(),
      install_requires=[
            'mb-modelbase',
            'numpy',
            'pandas',
      ],
      namespace_packages=['mb'],
      zip_safe=False,
      include_package_data=True,
      package_data={'': package_data_files})


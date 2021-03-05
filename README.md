# README #

[![Travis](https://travis-ci.org/lumen-org/modelbase.svg?branch=ci_travis)](
    https://travis-ci.org/lumen-org/modelbase)
    
A SQL-like interface for python and the web to query data together with probability models on the data.

Version: 0.95

### Content ###

`modelbase` can be used to model tabular data with generic probabilistic modelling as well to analyse, use and explore both, the fitted model as well as the data.
To this end the fitted models offer different types of operations such as prediction, conditionalization or marginalization.
Semantically equivalent operations, namely aggregation, row select and column selection are also available for data.
 
An overview over the capabilities of `modelbase` and a short introductory example of its Python API usage can be found in the jupyter-notebook files `doc/Intro_example.ipynb` and `doc/simple_API_usage.ipynb`.

`modelbase` provides an API-level access to model and data. 
It provides a generic modelling and querying backend, similar to what data base management systems are for tabular data alone.
We have also developed [lumen](https://github.com/lumen-org/lumen) for visual-interactive access to probabiliistic models that requires no coding at all. 
`lumen` provides a web-application for exploration, comparison and validation of probabilistic models and its data. It uses the webservice interface of `modelbase`. 

### Repository Overview ###

The `modelbase` repository contains a number directories as follows:

 * `bin`: This folder contains scripts to run an instance of the webservice-backend locally. In particular you may execute `webservice.py` to run the webservice as a local Flask app (see below) 
 * `dev`: This folder contains stuff only relevant for development.
 * `docs`: This folder contains documentation and introductions as jupyter notebooks.
  These runnable examples that serve as starting points for different use cases and tasks.
 * `cgmodsel`: Contains a required external python package (resolved as a git submodule).
 * `mb-*-pkg`: each is a namespace package under the common namespace `mb`.
 * `tests`: contains tests.

### Setup modelbase ###

**Requirements:**

 * Python3
 * jupyterlab or jupyternotebook is required to run the intro examples and many other tutorials. See [here](https://jupyter.org/install) for instructions.
 * R may be required for some optional components.

**Setup:**

*Note: It is strongly recommended to use some virtual environment of Python to install this software.* 

1. Clone this repository into a folder of your choice. Let's call it `<root>`.
2. Install other dependencies that are only available as git repositories (so called submodules) as follows:
    * Install the `cgmodsel` package with `git submodule init && git submodule update`
    * Install submodule `pip3 install cgmodsel` from `<root>`.
3. Install the base package `mb.modelbase` of the backend locally, i.e, do `cd <root>/mb-modelbase-pkg && pip3 install .`
4. Install the data package `mb.data`, i.e, do `cd <root>/mb-data-pkg && pip3 install .`
4. Run `bin/initialize.py`: this will create some initial probabilistic models in `bin/fitted_models`. 
   This is also a sanity check that things are all right with your installation.    
    
**Setup of optional components:**
 
This project uses the namespace `mb`. 
In that namespace a number of packages exist.
Following the setup instructions above you just installed the core package `mb.modelbase` and the data package `mb.data`.
If you want to install additional optional components you simply install the corresponding namespace packages, analogous to above.

Note that these subpackages may have conflicting dependencies, due to particular dependencies on third-party packages. 
Hence, you may not be able to install all components at once.
 
The following additional optional components and corresponding namespace packages exist:
Each of them provide an additional type of model to work with. 
 * `mb.pymc3`: Use probabilistic programs / statistical models written in the PyMC3 probabilistic programming language.   
 * `mb.stan`: Use probabilistic programs / statistical models written in the STAN probabilistic programming language.
 * `mb.gaussspn`: A gauss-SPN adapter. SPN stands for Sum-Product-Network.
 * `mb.libspn`: A libspn adapter.
 * `mb.mspn`: A mixed-SPN model adapter, based on some R package. Note that you need to install R for this to work.
 * `mb.spflow` A SPN-adapter based on the Python3 package spflow.
 
### Updating modelbase

You have installed `modelbase` as a number of (namespace) packages in your local python installation.
To update `modelbase` you have to update each of these packages to the latest repository version by indivudually uninstalling them, fetching the latest version from git and installing them.

Here it is explained with the `mb.modelbase` core package

1. uninstall the current version: For instance for the core package do: `pip uninstall mb.modelbase`
2. change into the local repository <root>/mb-modelbase-pkg
2. pull the latest version from the repo: `git pull origin master`
3. install the latest version: `pip3 install .`

Alternatively, you can use the `--editable` when installing the packages above. Then, you simply need to pull the latest version from the repo. 

### Using modelbase

`modelbase` provides three layers of APIs:

1. a webservice that accepts JSON-formatted http/https requests with a SQL inspired syntax. 
 See 'Running the modelbase webservice' and 'configuring the modelbase webservice' below.
   
2. a python class `ModelBase`. 
 An instance of that class is like a instance of a data base management system - just (also) for probabilistic models. 
 Use its member methods to add, remove models and run queries against it. 
 See the class documentation for more information.
   
3. a python class `Model`, which is the base class of all concrete models implemented in `modelbase'.
 A instance of such a class hence represents one particular model. See the class documentation for more information. 

### Running the modelbase webservice

This repository contains a number of namespace pacakges which you should have installed by now.
It also contains the `bin` directory, which contains executable scripts that you can use to run the backend as a webservice.

1. Execute `webservice.py`. 
  This will start a Flask web server locally. 
  It is the default and recommened way of using `modelbase` if you just use it locally in combination with the sister project [`lumen`](https://github.com/lumen-org/lumen).
   
2. Run it as an WSGI application with (for example) apache2. 
   The `modelbase.wsgi` file is provided for your convenience. 

When you start the webservice it will load all models from the directory you provided (see configuration options below).

### Configuring the modelbase webservice

There is three ways to configure the webservice.
In order of precedence (highest to lowest):

  * use command line arguments to `webservice.py`.
  This cannot be used if you run modelbase as a WSGI application.
  See `webservice.py --help` for available options.
  * set options in `run_conf.cfg`. 
  This is respected by both ways of running `modelbase` as a webservice (see above). 
  `run_conf.cfg` may have any subset of the options in `default_run_conf.cfg` and has the same format.
  To find out what options are available, please see `run_conf.cfg`.
  * set options in `default_run_conf.cfg`. 
  Changing settings here is not recommended. Use `run_conf.cfg` instead.

#### Hosting probabilistic models 

An instance of the webservice watches a directory and loads all models in that directory so that you can execute queries on these models and their data.
Models are stored as `.mdl` files and are serialized (pickled) instances of `md.Model`, that is, instances of some probabilistic model class.
By default models are loaded from the `bin/fitted_models` directory, and this directory some models that are created/trained during the setup process of the `modelbase`.

-----
 
### Contact ###

For any questions, feedback, bug reports, feature requests, rants, etc please contact: [philipp.lucas@dlr.de](philipp.lucas@dlr.de).

### Copyright and Licence ###

© 2016-2021 Philipp Lucas (philipp.lucas@dlr.de)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

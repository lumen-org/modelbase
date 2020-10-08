# README #

[![Travis](https://travis-ci.org/lumen-org/modelbase.svg?branch=ci_travis)](
    https://travis-ci.org/lumen-org/modelbase)
    
A SQL-like interface for python and the web to query data together with probability models on the data.

Version: 0.95

### Content ###

`modelbase` can be used to model tabular data with generic probabilistic modelling as well to analyse, use and explore both, the fitted model as well as the data. To this end the fitted models offer different types of operations such as prediction, conditionalization or marginalization. Semantically equivalent operations, namely aggregation, row select and column selection are also available for data.
 
An overview over the capabilities of `mb_modelbase` and a short introductory example can be found in the jupyter-notebook files `doc/Functions_overview.ipynb` and `doc/Intro_example.ipynb`. There it is shown how the python-package `mb_modelbase` is applied.

We also developed [lumen](https://github.com/lumen-org/lumen), an interactive web application for exploration, comparision and validation of probability models and its data. For an online demo version see [here](http://lumen.inf-i2.uni-jena.de/). `lumen` uses the webservice interface of `modelbase`. 

### Repository Overview ###

The `modelbase` repository contains the python package `mb_modelbase` in the likewise named folder.

 * `bin`: This folder contains scripts to run an instance of the webservice-backend locally. In particular you may execute `webservice.py` to run the webservice as a local Flask app (see below) 
 * `docs`: This folder contains documentation.
 * `jupyter`: This directory contains runnable examples that serve as starting points for different use cases and tasks.
 * `cgmodsel`: Contains a required external python package (resolved as a git submodule).

### The Lumen Project ###

`modelbase` is part of a larger project, namely [Lumen](https://github.com/lumen-org). Within `Lumen` there exist two main projects: the back-end [modelbase](https://github.com/lumen-org/modelbase) and the front-end [lumen](https://github.com/lumen-org/lumen) (yeah, there is some name clash... :-p).

The `modelbase` package is the kernel of Lumen. It provides a generic modelling and querying backend, similar to what data base management systems are for tabular data alone. 

### Setup modelbase ###

**Requirements:**

 * `modelbase` requires python3.
 * jupyterlab or jupyternotebook is required to run the intro examples and many other tutorials. See [here](https://jupyter.org/install) for instructions.

**Optional requirements:** 
 * If you want to work with mspn (mixed sum-product networks) then R is required. See [here](https://www.r-project.org/) for instructions. 

**Setup:**

*Note: It is strongly recommended to use some virtual environment of python to install this software.* 

1. Clone this repository into a folder of your choice. Let's call it `<root>`.
2. Install the `mb_modelbase` package locally, i.e, do `cd <root> && pip3 install .`
3. Install other dependencies that are only available as git repositories (so called submodules):
    * Install the `cgmodsel` package with `git submodule init && git submodule update`
    * Install submodule `pip3 install cgmodsel` from `<root>`.
4. Run `bin/initialize.py`: this will create some simples models to start with in `bin/fitted_models`. This is also a
   a sanity check that things are all right with your installation.    
    
**Setup of optional components:**
 * If you want to work with mspn (mixed sum-product networks), then you need to configure R correctly. 
 That is, set your `R_HOME` path variable such that it contains your R install directory (for example in `home/.profile`).

For a development setup: see further below.

If you use Anaconda:

 * Create a new environment: `conda create -n venv_name`
 * install new pacakges like this: `<path-to-environment>/bin/pip install <path-to-package>`
 
See also here: https://stackoverflow.com/questions/41060382/using-pip-to-install-packages-to-anaconda-environment

### Updating modelbase

Since you have installed it as a package in your local python installation, you have to to the following in order to update it to the latest repository version:
1. uninstall the current version: `pip uninstall mb_modelbase`
2. change into the local repository
2. pull the latest version from the repo: `git pull origin master`
3. install the latest version: `pip3 install .`

### Using modelbase

`modelbase` provides three layers of APIs:

1. a webservice that accepts JSON-formatted http/https requests with a SQL inspired syntax. See 'Running the modelbase webservice' and 'configuring the modelbase webservice' below.
2. a python class `ModelBase`. An instance of that class is like a instance of a data base management system - just (also) for probabilistic models. Use its member methods to add, remove models and run queries against it. See the class documentation for more information.
3. a python class `Model`, which is the base class of all concrete models implemented in `modelbase'. A instance of such a class hence represents one particular model. See the class documentation for more information. Also 

### Running the modelbase webservice

This repository contains the code for the python package `mb_modelbase` (in the folder with identical name). If you 
 followed the setup steps above this package is installed to your python environment.
Apart from the package the repo also contains the `bin` directory which we will use to run the backend. 

There is two intended ways to run the modelbase backend.
1. execute `webservice.py`. This starts a simple Flask web server locally. _Note that this should not be used for
    production environments._
2. run it as an WSGI application with (for example) apache2. To this end, the `modelbase.wsgi` file is provided. 

When you start the webservice it will load the models from the directory you provided (see configuration options).

### Configuring the modelbase webservice

There is three ways to configure the backend. In order of precedence (highest to lowest):
  * use command line arguments to `webservice.py`. This cannot be used if you run modelbase as a WSGI
    application. See `webservice.py --help` for available options.
  * set options in `run_conf.cfg`. This is respected by both ways of running `modelbase` (see above). `run_conf.cfg`
   may have any subset of the options in `default_run_conf.cfg` and has the same format. Note that `run_conf.cfg` does initially *not* exist after cloning the project. To find out what options are available, please see `default_run_conf.cfg`.
  * set options in `default_run_conf.cfg`. Changing settings here is not recommended. 

#### Hosting models 

Notes:
 * once run the server does not immediately necessarily produce any output on the command line. *that is normal*.
 * don't forget to activate your custom environment, *if you configured one in the process above*.

### Development Setup

This section describes the _recommended_ development setup. 

I recommend using PyCharm as an IDE. You can set the virtual python environment to use (if any specific) in PyCharm like [this](https://docs.continuum.io/anaconda/ide_integration#pycharm).
Moreover, I recommend not installing it as a local package, but instead adding `<root>` (the path to your local clone of the repo) to the environment variable `$PYTHONPATH`. This make the workflow faster, because you do not need to update the local installation when you changed the code in the repository. See as follows:

1. setup public key authentication for the repository. This way you do not need to provide passwords when pulling/pushing.
2. clone repository to local machine into `<root>`
3. add `<root>` to `PYTHONPATH` environment variable in `.profile` (Linux) or in your system environment variables (windows). This way the package under development is found by Python without the need to install it.
   * Linux: `export PYTHONPATH="<path>:$PYTHONPATH"`
   * Windows: somewhere under advanced system settings ...
4. install [PyCharm IDE](https://www.jetbrains.com/pycharm/)
5. create python virtual environment for project 
   * using PyCharm IDE: open project -> settings -> project settings -> project interpreter -> add python interpreter
   * using pure virtual env: see [here](https://virtualenv.pypa.io/en/stable/userguide/#usage)
6. install package dependencies (but does not install the package):
   * using PyCharm:
    1. open up `setup.py` in PyCharm
    2. a dialog should be shown: "Package requirements ... are not satisfied"
    3. click "install requirements"
   * manually in a shell:
    1. change into `<path>`
    2. activate previously virtual environment 
    3. `pip3 install -e .` to install dependencies of this package only
7. install CGModelSelection from [here](https://ci.inf-i2.uni-jena.de/ra86ted/CGmodelselection)
   * either install it as a python package locally or add the local repository directory to `PYTHONPATH` as above
   * if you do not install it as a python package, anyway install its dependencies using `pip3 install -e .` from the repo directory
8. get tje data repository [from here](https://ci.inf-i2.uni-jena.de/gemod/mb_data)
   * either install it as a python package locally or add the local repository directory to `PYTHONPATH` as above.
   * This repository provides prepared data set and preset configurations to learn models from the data. As you probably want to change this every now an then as well, I recommend to not install the package but add it to `PYTHONPATH`, see above.
9. get the front-end [lumen](https://github.com/lumen-org/lumen)
   * you probably want the front-end as well.  See the README of the repository for more details.

-----
 
### Contact ###

For any questions, feedback, bug reports, feature requests, spam, etc please contact: [philipp.lucas@dlr.de](philipp.lucas@dlr.de).

### Copyright and Licence ###

© 2016-2018 Philipp Lucas (philipp.lucas@dlr.de)

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

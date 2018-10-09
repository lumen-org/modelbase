# README #

A Probability Model Backend for Python and the Web.

Version: 0.9

### Content ###

The `modelbase` repository contains a full-developed and installable python-package called `mb_modelbase` and the associated directory `scripts`. 

The python-package `mb_modelbase` can be used for exploring all kinds of data sets with generic probabilistic modelling. The fitted models itself offer different types of operations such as prediction, conditionalization or marginalization. For using Lumen, `mb_modelbase` has to be installed as a python-package (For instructions see below). An overview over the functions of `mb_modelbase` and a short introductory example how the package could be used can be found in the jupyter-notebook files `doc/Functions_overview.ipynb` and `doc/Intro_example.ipynb`. Here it is shown how the python-package `mb_modelbase` is applied.

The script folder contains the important file `ajax.py` which is used for starting the backend of Lumen (see below). 

### Classification in the Project ###

The `modelbase` repository is the kernel of Lumen. Here all the central functions are located in the subdirectories such as the different models or utils. Combined with the `ajax.py` file in the script-folder, the backend of Lumen is complete.

### Setup Lumen ###

For normal usage:

1. Clone this repository into a folder of your choice. Let's call it `<root>`.
2. Install the `mb_modelbase` package locally, i.e, do `cd <root> && pip3 install .`
3. Install the `CGModelSelection` package. This provides a model selection capabilities for various types of multivariate Gaussian and Conditional Gaussian (CG) models.  [See here](https://ci.inf-i2.uni-jena.de/ra86ted/CGmodelselection) for the package.

What happens here? The command `pip3 install` calls the `setup.py` script and copies the package to one of the python paths, therefore the python modules can be found by the `ajax.py` script.

### Updating modelbase ###

Consequently, if you want to update the package you have to:
1. uninstall the current version: `pip uninstall mb_modelbase`
2. change into the local repository
2. pull the latest version from the repo: `git pull origin master`
3. install the latest version: `pip3 install .`

### Running the modelbase backend ###

The repository contains the code for the python package `mb_modelbase` (in the folder with identical name), and this is typically installed to your local python installation.
It, however, also contains some scripts in the `scripts` directory. There, only `ajax.py` is of interest. 

To actually run the backend, just run the python script `ajax.py`, i.e. execute `ajax.py` on a systems shell. On windows/anaconda you might need to run it from an Anaconda prompt.
Note that it requires some command line parameters in order to select the models to load. Run  `ajax.py --help` for more information or check the helptext in the script.

`ajax.py` will start a local Flask Webserver that accepts and answers PQL-queries. See `ajax.py` for more information.

Note:
 * once run the server doesn't immediatly necessarly produce any output on the command line. *that is normal*
 * don't forget to activate your custom environment, *if you configured one in the process above*.
 * to actually use the backend, the frontend [PMV](https://ci.inf-i2.uni-jena.de/gemod/pmv) should be used. 

### (recommended) Development Setup ###

I recommend using PyCharm as an IDE. You can set the virtual python environment to use (if any specific) in PyCharm like [this](https://docs.continuum.io/anaconda/ide_integration#pycharm).
Moreover, I recommend not installing it as a local package, but instead adding `<root>` to the path of python, which is defined in the variable `$PYTHONPATH`. This make the workflow faster, because you do not need to update the local installation when you changed the code in the repository. See as follows:

1. setup public key authentification for repository. This way you do not need to provide passwords when pulling/pushing.
2. clone repository to local machine into <path>
3. add <path> to PYTHONPATH environment variable in .profile (Linux) or in your system environvent variables (windows). This way the package under development is found by Python with the need to install it.
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
    3. `pip install -e .` to install depedencies of this package only
7. install CGModelSelection from [here](https://ci.inf-i2.uni-jena.de/ra86ted/CGmodelselection)
  * either install it locally or add the local repository directory to `PYTHONPATH` as above
8. get data repository [from here](https://ci.inf-i2.uni-jena.de/gemod/mb_data)
  * It provides prepared data and methods to preset configurations to learn models from the data. As you probably want to change this every now an then as well, I recommend to not install the package but add it to `PYTHONPATH`.
9. get lumen front-end [from here](https://ci.inf-i2.uni-jena.de/gemod/pmv)

### Contact ###

For any questions, feedback, bug reports, feature requests, spam, etc please contact: [philipp.lucas@uni-jena.de](philipp.lucas@uni-jena.de) or come and see me in my office #3311.

### Copyright ###

© 2016-2018 Philipp Lucas (philipp.lucas@uni-jena.de) All Rights Reserved

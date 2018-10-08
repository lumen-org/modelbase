# README #

A Probability Model Backend for Python and the Web.

Version: 0.2

### Content ###

The `modelbase` repository contains a full-developed and installable python-package called `mb_modelbase` and the associated directory `scripts`. 

The python-package `mb_modelbase` can be used for exploring all kinds of data sets with generic probabilistic modelling. The fitted models itself offer different types of operations such as prediction, conditionalization or marginalization. For using Lumen, `mb_modelbase` has to be installed as a python-package (For instructions see below). An overview over the functions of `mb_modelbase` and a short introductory example how the package could be used can be found in the jupyter-notebook files `doc/Functions_overview.ipynb` and `doc/Intro_example.ipynb`. Here it is shown how the python-package `mb_modelbase` is applied.

The script folder contains the important file `ajax.py` which is used for starting the backend of Lumen (see below). 

### Classification In The Project ###

The `modelbase` repository is the kernel of Lumen. Here all the central functions are located in the subdirectories such as the different models or utils. Combined with the `ajax.py` file in the script-folder, the backend of Lumen is complete.

### Setup Lumen ###

For normal usage:

1. Clone this repository into a folder of your choice. Let's call it `<root>`.
2. Install the `mb_modelbase` package locally, i.e, do `cd <root> && pip install .`

What happens here? The command `pip install` calls the `setup.py` script and copies the package to one of the python paths, therefore the python modules can be found by the `ajax.py` script. Consequently, if you want to update the package you have to install it again after pulling the latest version of the project. Otherwise the old installation is used. See below how to use the package without the manually update process. 

For development:

 - I recommend using PyCharm as an IDE. You can set the environment to use (if any specific) in PyCharm like [this](https://docs.continuum.io/anaconda/ide_integration#pycharm).
 - I recommend not installing it as a local package, but instead adding `<root>` to the path of python, which is defined in the variable `$PYTHONPATH`. This make the workflow faster, because you do not need to update the local installation when you changed the code in the repository.

### Running The ModelBase Backend ###

The repository contains the code for the python package `mb_modelbase` (in the folder with identical name), and this is typically installed to your local python installation.
It, however, also contains some scripts in the `scripts` directory. There, only `ajax.py` is of interest. 

To actually run the backend, just run the python script `ajax.py`, i.e. execute `ajax.py` on a systems shell. On windows/anaconda you might need to run it from an Anaconda prompt.
Note that it requires some command line parameters in order to select the models to load. Run  `ajax.py --help` for more information or check the helptext in the script.

`ajax.py` will start a local Flask Webserver that accepts and answers PQL-queries. See `ajax.py` for more information.

Note:
 * once run the server doesn't immediatly necessarly produce any output on the command line. *that is normal*
 * don't forget to activate your custom environment, *if you configured one in the process above*.
 * to actually use the backend, the frontend [PMV](https://ci.inf-i2.uni-jena.de/gemod/pmv) should be used. 

### recommended Development Setup ###

1. setup public key authentification for repository 
2. clone repository to local machine into <path>
3. add <path> to pythonpath in .profile (Linux) or in your system environvent variables (windows). This way the package under development is found by Python with the need to install it.
  * Linux: PYTHONPATH="<path>:$PYTHONPATH"
  * Windows: similar
4. install PyCharm IDE
5. create python virtual environment for project 
  1. using PyCharm IDE: open project -> settings -> project settings -> project interpreter -> add python interpreter
  2. using pure virtual env: see [here](https://virtualenv.pypa.io/en/stable/userguide/#usage)
6. install package dependencies (but does not install the package):
  1. activate virtual environment 
  2. pip install -e .

### Contact ###

For any questions, feedback, bug reports, feature requests, spam, etc please contact: [philipp.lucas@uni-jena.de](philipp.lucas@uni-jena.de) or come and see me in my office #3311.

### Copyright ###

© 2016 Philipp Lucas (philipp.lucas@uni-jena.de) All Rights Reserved

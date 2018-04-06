# README #

A Probability Model Backend for Python and the Web.

Version: 0.2

### Content ###

The `modelbase` repository contains a full-developed and installable python-package called `mb_modelbase` and the associated directory `scripts`. 

The python-package `mb_modelbase` can be used for exploring all kinds of data sets with generic probabilistic modelling. The fitted models itself offer different types of operations such as prediction, conditionalization or marginalization. For using the Moo-Software, `mb_modelbase` has to be installed as a python-package (For instructions see below). An example how to use the package itself can be found in the jupyter-notebook file `Introduction.ipynb`. Here it is shown how the python-package `mb_modelbase` is applied.

The script folder contains the important file `ajax.py` what is used for starting the whole backend of the Moo-Software (see below). 

### Classification In The Project ###

The `modelbase` repository is the kernel of the Moo-Software. Here we find all the central functions in the subdirectories like the different models or utils. Combined with the `ajax.py` file in the script-folder, the backend of the Moo-Software is complete.

### Setup ###

For normal usage:

1. Clone this repository into a folder of your choice. Let's call it `<root>`.
2. Install the `mb_modelbase` package locally, i.e, do `cd <root> && pip install .`

For development:

 - I recommend using PyCharm as an IDE. You can set the environment to use (if any specific) in PyCharm like [this](https://docs.continuum.io/anaconda/ide_integration#pycharm).
 - I recommend not installing it as a local package, but instead adding root to the path of python. This make the workflow faster, because you do not need to update the local installation when you changed the code in the repository.

### Running The ModelBase Backend ###

The repository contains the code for the python package `mb_modelbase` (in the folder with identical name), and this is typically installed to your local python installation.
It, however, also contains some scripts in the `scripts` directory. There, only `ajax.py` is of interest. 

To actually run the backend, just run the python script `ajax.py`, i.e. execute `ajax.py` on a systems shell. On windows/anaconda you might need to run it from an Anaconda prompt.
Note that it requires some command line parameters in order to select the models to load. Run  `ajax.py --help` for more information.

`ajax.py` will start a local Flask Webserver that accepts and answers PQL-queries. See `ajax.py` for more information.

Note:
 * once run the server doesn't immediatly necessarly produce any output on the command line. *that is normal*
 * don't forget to activate your custom environment, *if you configured one in the process above*.

### Contact ###

For any questions, feedback, bug reports, feature requests, spam, etc please contact: [philipp.lucas@uni-jena.de](philipp.lucas@uni-jena.de) or come and see me in my office #3311.

### Copyright ###

© 2016 Philipp Lucas (philipp.lucas@uni-jena.de) All Rights Reserved

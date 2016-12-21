# README #

A Probability Model Backend for Python and the Web.

Version: 0.2

### Setup ###

1. Clone this repository into a folder of your choice. Let's call it `root`.
2. Install [anaconda](https://www.continuum.io/downloads) (Python 3.5+) on your system
    * Note: generally you can also use another Python installation, however using Anaconda simplifies the process of installing the required packages
2. Next we make sure that all required packages are available:
    a) we create (actually import) a [virtual Python environment](http://conda.pydata.org/docs/using/envs.html#) that contains all required packages
        * change into `root` on a command line
        * import the environment using the command: `conda env create -f environment.yml -n NAME-OF-NEW-ENV`
        * Note: in case you don't use Anaconda, you can use the [virtualenv packages](http://docs.python-guide.org/en/latest/dev/virtualenvs/) instead. `environment.yml` contains plain-text information about required packages.
    b) we just manually install the required packages in our environment, the packages are:
        * flask logging json traceback xarray numpy copy pandas math sklearn seaborn pickle random string
        * (probably quite a few of them are distributed with a default python installation anyway... I should clean this up)

If you simply want to run it, you are done. For development, I recommend using PyCharm as an IDE. You can set the environment to use in PyCharm like [this](https://docs.continuum.io/anaconda/ide_integration#pycharm).

### Running the Webinterface ###

Just run `ajax.py` (using your environment!). It will start a local Flask Webserver that accepts and answers PQL-queries. See `ajax.py` for more information.

### Contact ###

For any questions, feedback, bug reports, feature requests, spam, etc please contact: [philipp.lucas@uni-jena.de](philipp.lucas@uni-jena.de) or come and see me in my office #3311.

### Copyright ###

© 2016 Philipp Lucas (philipp.lucas@uni-jena.de) All Rights Reserved
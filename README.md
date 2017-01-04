# README #

A Probability Model Backend for Python and the Web.

Version: 0.2

### Setup ###

1. Clone this repository into a folder of your choice. Let's call it `root`.
2. make sure you have a working python 3.5+ on your system
    * if you don't, you can for example install [anaconda](https://www.continuum.io/downloads) which comes with python
3. install required packages:
    * run `pip install flask xarray numpy pandas sklearn seaborn` on a system shell (*not python shell*)
    * note: you can use a [virtual Python environment](http://conda.pydata.org/docs/using/envs.html#) in conda or the [virtualenv packages](http://docs.python-guide.org/en/latest/dev/virtualenvs/) to create a so-called virual environment, i.e. a specific python environment that contains only the packages your select. This is, however, *not* strictly necessary

If you simply want to run it, you are done. For development, I recommend using PyCharm as an IDE. You can set the environment to use (if any specific) in PyCharm like [this](https://docs.continuum.io/anaconda/ide_integration#pycharm).

### Running the ModelBase backend ###

Just run the python script `ajax.py`, i.e. execute `python ajax.py` on a systems shell from the repositories root directory. It will start a local Flask Webserver that accepts and answers PQL-queries. See `ajax.py` for more information.

Note: 

 * once run the server doesn't immediatly necessarly produce any output on the command line. *that is normal*
 * don't forget to activate your custom environment, *if you configured one in the process above*.

### Contact ###

For any questions, feedback, bug reports, feature requests, spam, etc please contact: [philipp.lucas@uni-jena.de](philipp.lucas@uni-jena.de) or come and see me in my office #3311.

### Copyright ###

© 2016 Philipp Lucas (philipp.lucas@uni-jena.de) All Rights Reserved
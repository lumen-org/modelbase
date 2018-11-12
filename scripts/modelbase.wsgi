from webservice import init 

args = {
  loglevel: "INFO",
  directory: "../lumen_data/fitted_models",
  name: "WSGI backend modelbase",
}

init(args)

from webservice import app as application


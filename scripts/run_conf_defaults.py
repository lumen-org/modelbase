# Copyright (C) 2014-2018 , Philipp Lucas, philipp.lucas@gmail.com
"""Default run configuration for the modelbase webservice

The dictionary `cfg` contains the configuration that is read in when modelbase is run as a webservice with the script
`webservice.py` or as a WSGI service with `webservice.wsgi`.

See below for available options and their meanings.
"""
cfg = {

    'port': 52104,  # the port where the service is provided

    'loglevel': 'INFO',  # The global log level. see https://docs.python.org/3/library/logging.html#levels.

    'modules': {
        # Modules are independent functional units of the modelbase webservice which are available under
        # <this-host>/<route-of-the-module>. They may individually be enabled or disabled using the `enable` key.

        'root': {
            'enable': True,
            'route': '/',
        },
        'modelbase': {
            'enable': True,
            'route': '/webservice',
            'directory': './experiments/fitted_models',
            'name': 'modelbase management system',
        },
        'activitylogger': {
            'enable': True,
            'route': '/activitylogger',
        },
        'webquery': {
            'enable': False,
            'route': '/webquery',
        }
    },

    'ssl': {
        # SSL settings only have an effect if the built-in flask web server is used. Otherwise use apache/nginx... to enable SSL
        'enable': False,  # [False, True]. En/disables SSL
        'cert_chain_path': '../ssl/fullchain.pem',  # path to certificate chain
        'cert_priv_key_path': '../ssl/privkey.pem',  # path to private key
        # 'cert_chain_path': '/opt/lumen/ssl/modelvalidation.mooo.com/fullchain.pem',
        # 'cert_priv_key_path': '/opt/lumen/ssl/modelvalidation.mooo.com/privkey.pem',
        # 'cert_chain_path': '/opt/lumen/ssl/lumen.inf-i2.uni-jena.de.ca',
        # 'cert_priv_key_path': '/opt/lumen/ssl/lumen.inf-i2.uni-jena.de.key',
    },
}

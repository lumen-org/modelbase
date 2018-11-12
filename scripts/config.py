# Copyright (C) 2014-2018 , Philipp Lucas, philipp.lucas@gmail.com

cfg = {
    'modules': {
        'root': {
            'enable': True,
            'route': '/',
        },
        'modelbase': {
            'enable': True,
            'route': '/webservice',
            'directory': '../../fitted_models',
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
        'enable': False,
        'cert_chain_path': '../ssl/fullchain.pem',
        'cert_priv_key_path': '../ssl/privkey.pem',
        # 'cert_chain_path': '/opt/lumen/ssl/modelvalidation.mooo.com/fullchain.pem',
        # 'cert_priv_key_path': '/opt/lumen/ssl/modelvalidation.mooo.com/privkey.pem',
        # 'cert_chain_path': '/opt/lumen/ssl/lumen.inf-i2.uni-jena.de.ca',
        # 'cert_priv_key_path': '/opt/lumen/ssl/lumen.inf-i2.uni-jena.de.key',
    },

    'loglevel': 'INFO',
}
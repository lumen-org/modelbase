

# default config
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

    'loglevel': 'INFO',
}
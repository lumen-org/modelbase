#!/usr/bin/env python
# Copyright (C) 2014-2018 , Philipp Lucas, philipp.lucas@gmail.com

from flask import Flask, request
from flask_cors import cross_origin
import logging
import json
import traceback

from mb_modelbase.utils import utils, ActivityLogger
from mb_modelbase.server import modelbase as mbase

#from mb_modelbase.utils.utils import is_running_in_debug_mode
# if is_running_in_debug_mode():
#     print("running in debug mode!")
#     import mb_modelbase.models_core.models_debug

app = Flask(__name__, static_url_path='/static/')

logger = None  # create module variable


def add_path_of_file_to_python_path():
    """Add the absolute path of __file__ to the python search path."""
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.insert(0, path)


# load config. user config overrides default config.
add_path_of_file_to_python_path()
from run_conf_defaults import cfg
try:
    from run_conf import cfg as user_cfg
except ModuleNotFoundError:
    # user config may not exist, but that is ok
    pass
else:
    cfg = utils.deep_update(cfg, user_cfg)


def add_root_module():
    # the (static) start page
    c = cfg['modules']['root']
    @app.route(c['route'])
    @cross_origin()  # allows cross origin requests
    def index():
        return "webservice up an running!"


def add_modelbase_module():
    # webservice interface to the model base

    c = cfg['modules']['modelbase']

    # start ModelBase
    logger.info("starting modelbase ... ")
    mb = mbase.ModelBase(name=c['name'], model_dir=c['directory'])
    logger.info("... done (starting modelbase).")

    @app.route(c['route'], methods=['GET', 'POST'])
    @cross_origin()  # allows cross origin requests
    def modebase_service():
        # return usage information
        if request.method == 'GET':
            return "send a POST request to this url containing your model query and you will get your answer :-)"
        # handle model request
        else:
            try:
                # extract json formatted query
                query = request.get_json()
                logger.info('received QUERY:' + str(query))
                # process query
                result = mb.execute(query)
                logger.info('result of query:' + utils.truncate_string(str(result)))
                # return answer
                return result
            except Exception as inst:
                msg = "failed to execute query: " + str(inst)
                logger.error(msg + "\n" + traceback.format_exc())
                return msg, 400


def add_activitylogger_module():
    # user activity logger
    activitylogger = ActivityLogger()

    c = cfg['modules']['activitylogger']
    @app.route(c['route'], methods=['POST'])
    @cross_origin()  # allows cross origin requests
    def activitylogger_service():
        # log as requested
        try:
            activity = request.get_json()
            logger.debug('received LOG:' + str(activity))
            activitylogger.log(activity)
            return json.dumps({'STATUS': 'success'})
        except Exception as inst:
            msg = "failed to log: " + str(inst)
            logger.error(msg + "\n" + traceback.format_exc())
            return msg, 400


def add_webquery_module():
    # the webclient
    cfg_webquery = cfg['modules']['webquery']
    @app.route(cfg_webquery['route'], methods=['GET'])
    @cross_origin()  # allows cross origin requests
    def webquery_service():
        return app.send_static_file('webqueryclient.html')


def init():
    # setup root logger and local logger
    logging.basicConfig(
        level=cfg['loglevel'],
        format='%(asctime)s.%(msecs)03d %(levelname)s %(filename)s :: %(message)s',
        datefmt='%H:%M:%S'
    )
    global logger
    logger = logging.getLogger(__name__)

    # setup modules
    if cfg['modules']['root']['enable']:
        add_root_module()

    if cfg['modules']['modelbase']['enable']:
        add_modelbase_module()

    if cfg['modules']['activitylogger']['enable']:
        add_activitylogger_module()

    if cfg['modules']['webquery']['enable']:
        add_webquery_module()


# trigger to start the web server if this script is run
if __name__ == "__main__":
    import argparse
    # import pdb

    description = """
    Starts a local web server that acts as an interface to a modelbase, i.e. the equivalent of a
    data base, but for graphical models. This interface provides various routes,
    as follows.
    
      * '/': the index page
      * '/webservice': a user can send PQL queries in a POST-request to this route
      * '/webqueryclient': provides a simple website to sent PQL queries to this
          model base (probably not functional at the moment)
      * '/playground': just for debugging / testing / playground purposes
    
    Usage:
        Run this script to start the server locally!
    """
    cfg_mb = cfg['modules']['modelbase']
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--name", help="A name for the modelbase to start. Defaults to '{}'".format(cfg_mb['name']),
                        type=str, default=cfg_mb['name'])
    parser.add_argument("-d", "--directory", help="directory that contains the models to be loaded initially. Defaults "
                                                  "to '{}'".format(cfg_mb['directory']),
                        type=str, default=cfg_mb['directory'])
    parser.add_argument("-l", "--loglevel", help="loglevel for command line output. You can set it to: CRITICAL, ERROR,"
                                                 " WARNING, INFO or DEBUG. Defaults to {}".format(cfg['loglevel']),
                        type=str, default=cfg['loglevel'])

    # overwrite config of run_conf.py
    args = parser.parse_args()
    cfg['modules']['modelbase']['directory'] = args.directory
    cfg['modules']['modelbase']['name'] = args.name
    cfg['loglevel'] = args.loglevel

    init()

    if cfg['ssl']['enable']:
        from OpenSSL import SSL
        context = (cfg['ssl']['cert_chain_path'], cfg['ssl']['cert_priv_key_path'])
        app.run(host='0.0.0.0', port=cfg['port'], ssl_context=context, threaded=True)
    else:
        app.run(host='0.0.0.0', port=cfg['port'], threaded=True)

    logger.info("web server running...")
    # pdb.run('app.run()')

#!/usr/bin/env python
# Copyright (C) 2014-2020 , Philipp Lucas, philipp.lucas@dlr.de

from flask import Flask, request
from flask_cors import cross_origin
from flask_socketio import SocketIO

import argparse
import logging
import json
import traceback
import os
import sys

from configparser import ConfigParser

from mb_modelbase.server import modelbase as mbase
from mb_modelbase.utils import utils, ActivityLogger
from mb_modelbase import DictCache


# from mb_modelbase.utils.utils import is_running_in_debug_mode
# if is_running_in_debug_mode():
#     print("running in debug mode!")
#     import mb_modelbase.core.models_debug


def add_path_of_file_to_python_path():
    """Add the absolute path of __file__ to the python search path."""
    path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, path)


add_path_of_file_to_python_path()

app = Flask(__name__, static_url_path='/static/')
socketio = SocketIO(app)  # adds socket listener to Flask app

flask_logger = logging.getLogger('__name__')
flask_logger.setLevel(logging.WARNING)
logger = None  # create module variable


def add_root_module():
    # the (static) start page
    @app.route(config['ROOT']['route'])
    @cross_origin()  # allows cross origin requests
    def index():
        return "webservice up an running!"


def add_modelbase_module():
    # webservice interface to the model base
    logger.info("starting modelbase ... ")
    c = config['MODELBASE']

    if c.getboolean('cache_enable'):
        model_cache = DictCache(
            cache_dir=os.path.abspath(c['cache_path']),
            save_interval=int(c['cache_interval'])
        )
    else:
        model_cache = None

    mb = mbase.ModelBase(
        name=c['name'],
        model_dir=os.path.abspath(c['model_directory']),
        auto_load_models={
            'reload_on_overwrite': c.getboolean('reload_on_overwrite'),
            'reload_on_creation': c.getboolean('reload_on_creation')
        },
        cache=model_cache
    )
    logger.info("... done (starting modelbase).")

    @app.route(c['route'], methods=['GET', 'POST'])
    @cross_origin()  # allows cross origin requests
    def modebase_service():
        dont_log = {"SHOW": "MODELS"}
        # return usage information
        if request.method == 'GET':
            return "send a POST request to this url containing your model query and you will get your answer :-)"
        # handle model request
        else:
            try:
                # extract json formatted query
                query = request.get_json()
                if query != dont_log:
                    logger.info('received QUERY:' + str(query))
                # process query
                result = mb.execute(query)
                if query != dont_log:
                    logger.info('result of query:' + utils.truncate_string(str(result)))
                # logger.info('result of query:' + str(result))
                # return answer
                return result
            except Exception as inst:
                msg = "failed to execute query: " + str(inst)
                logger.error(msg + "\n" + traceback.format_exc())
                return msg, 400
            # else:
            #     return mb.upload_files(request.get_data())

    @socketio.on('models')
    def handle_model_send(models):
        """
        When event models is invoked call function models which saves all models into the model-dir

        :param models: list of dill dumped models
        :return: success or error code
        """
        return mb.upload_files(models)


def add_activitylogger_module():
    # user activity logger
    activitylogger = ActivityLogger()

    c = config['ACTIVITYLOGGER']

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
    cfg_webquery = config['WEBQUERY']

    @app.route(cfg_webquery['route'], methods=['GET'])
    @cross_origin()  # allows cross origin requests
    def webquery_service():
        return app.send_static_file('webqueryclient.html')


def init():
    # setup root logger and local logger
    logging.basicConfig(
        level=config['GENERAL']['loglevel'],
        format='%(asctime)s.%(msecs)03d %(levelname)s %(filename)s :: %(message)s',
        datefmt='%H:%M:%S'
    )
    global logger
    logger = logging.getLogger(__name__)

    # setup modules
    if config.getboolean('ROOT', 'enable'):
        add_root_module()

    if config.getboolean('MODELBASE', 'enable'):
        add_modelbase_module()

    if config.getboolean('ACTIVITYLOGGER', 'enable'):
        add_activitylogger_module()

    if config.getboolean('WEBQUERY', 'enable'):
        add_webquery_module()


# trigger to start the web server if this script is run
if __name__ == "__main__":
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

    logger = logging.getLogger(__name__)
    bin_dir = os.path.dirname(os.path.abspath(__file__))

    conf_default_path = os.path.join(bin_dir, 'run_conf_defaults.cfg')
    conf_path = os.path.join(bin_dir, 'run_conf.cfg')

    # load config from file
    config = ConfigParser()
    config.read(conf_default_path)
    if not os.path.isfile(conf_path):
        logger.warning('run_conf.cfg is missing. All default configs apply.')
    else:
        config.read(conf_path)

    # get command line args
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--name",
                        help="A name for the modelbase to start. Defaults to '{}'".format(
                            config['MODELBASE']['name']),
                        type=str, default=config['MODELBASE']['name'])
    parser.add_argument("-d", "--directory",
                        help="directory that contains the models to be loaded initially. Defaults "
                             "to '{}'".format(config['MODELBASE']['model_directory']),
                        type=str, default=config['MODELBASE']['model_directory'])
    parser.add_argument("-l", "--loglevel",
                        help="loglevel for command line output. You can set it to: CRITICAL, ERROR,"
                             " WARNING, INFO or DEBUG. Defaults to {}".format(
                            config['GENERAL']['loglevel']),
                        type=str, default=config['GENERAL']['loglevel'])

    # overwrite config of run_conf.cfg
    args = parser.parse_args()
    config['MODELBASE']['model_directory'] = args.directory
    config['MODELBASE']['name'] = args.name
    config['GENERAL']['loglevel'] = args.loglevel

    init()

    if config.getboolean('SSL', 'enable'):
        from OpenSSL import SSL

        context = (config['SSL']['cert_chain_path'], config['SSL']['cert_priv_key_path'])
        app.run(host='0.0.0.0', port=int(config['GENERAL']['port']), ssl_context=context,
                threaded=True)
    else:
        app.run(host='0.0.0.0', port=int(config['GENERAL']['port']), threaded=True)

    logger.info("web server running...")
    # pdb.run('app.run()')

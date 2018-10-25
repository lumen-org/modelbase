#!/usr/bin/env python
# Copyright (C) 2014-2018 , Philipp Lucas, philipp.lucas@gmail.com

from flask import Flask, request
from flask_cors import cross_origin
import logging
import json
import traceback

from mb_modelbase.utils import utils, ActivityLogger
from mb_modelbase.server import modelbase as mbase

from mb_modelbase.utils.utils import is_running_in_debug_mode
# if is_running_in_debug_mode():
#     print("running in debug mode!")
#     import mb_modelbase.models_core.models_debug

app = Flask(__name__, static_url_path='/static/')

logger = None  # create module variable

# the (static) start page
@app.route('/')
@cross_origin()  # allows cross origin requests
def index():
    return app.send_static_file('index.html')


# webservice interface to the model base
@app.route('/webservice', methods=['GET', 'POST'])
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


# user activity logger
@app.route('/activitylogger', methods=['POST'])
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


# the webclient
@app.route('/webquery', methods=['GET'])
@cross_origin()  # allows cross origin requests
def webquery_service():
    return app.send_static_file('webqueryclient.html')


# route that returns a valid sample query
@app.route('/sample_query', methods=['GET', 'POST'])
@cross_origin()  # allows cross origin requests
def sample_query():
    if request.method == 'GET':
        return "send a POST request to this url to get a valid query that you can use at the '/webservice' interface"
    else:
        filepath = 'test-model-query_03.json'
        # open file, read as json
        query = json.load(open(filepath))
        # serialize to string and return
        return json.dumps(query)


# a "playground" webservice interface
@app.route('/playground', methods=['GET', 'POST'])
def playground():
    # return usage information
    if request.method == 'GET':
        return "this is just for playing around and testing how HTTP POST is working..."
    # handle model request
    else:
        result = '{"age":[0,5,2,3,2,561,0], "income":[1,2,3,4,5,6,7]}'
        return result


def init(args):
    # setup root logger and local logger
    logging.basicConfig(
        level=args.loglevel,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(filename)s :: %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # setup activity logger
    activitylogger = ActivityLogger()

    # start ModelBase
    logger.info("starting modelbase ... ")
    mb = mbase.ModelBase(name=args.name, model_dir=args.directory)
    logger.info("... done (starting modelbase).")


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
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--name", help="A name for the modelbase to start. Defaults to 'my_mb'",
                        type=str, default='my_mb')
    parser.add_argument("-d", "--directory", help="directory that contains the models to be loaded initially. Defaults"
                                                  " to 'data_models'", type=str, default='data_models')
    parser.add_argument("-l", "--loglevel", help="loglevel for command line output. You can set it to: CRITICAL, "
                                                 "ERROR, WARNING, INFO or DEBUG. Defaults to INFO",
                        type=str, default='INFO')

    args = parser.parse_args()
    init(args)
    app.run()
    logger.info("web server running...")
    # pdb.run('app.run()')
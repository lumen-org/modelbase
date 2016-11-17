"""
@author: Philipp Lucas

This module provides a webinterface to a modelbase, i.e. the equivalent of a
data base, but for graphical models. That interface provides various routes,
as follows.

  * '/': the index page
  * '/webservice': a user can send PQL queries in a POST-request to this route
  * '/webqueryclient': provides a simple website to sent PQL queries to this
      model base

There is other routes available:
  * '/playground': just for debugging / testing / playground purposes

Usage:
    Run this script to start the server locally!
"""

from flask import Flask, request
from flask_cors import CORS, cross_origin
import logging
import json
import modelbase as mbase
import traceback

app = Flask(__name__, static_url_path='/static/')
mb = mbase.ModelBase("mymb")

# setup root logger and local logger
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(filename)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# the (static) start page
@app.route('/')
def index():
    return app.send_static_file('index.html')


# webservice interface to the model base
@app.route('/webservice', methods=['GET', 'POST'])
@cross_origin()  # allows cross origin requests
def service():
    # return usage information
    if request.method == 'GET':
        return "send a POST request to this url containing your model query and you will get your answer :-)"
    # handle model request
    else:
        try:
            # extract json formatted query
            query = request.get_json()
            logger.info('received query:' + str(query))
            # process query
            result = mb.execute(query)
            logger.info('result of query:' + str(result))
            # return answer
            return result
        except Exception as inst:
            msg = "failed to execute query: " + str(inst)
            logger.error(msg + "\n" + traceback.format_exc())
            return msg, 400


# the webclient
@app.route('/webquery', methods=['GET'])
@cross_origin()  # allows cross origin requests
def webquery():
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


# trigger to start the web server if this script is run
if __name__ == "__main__":
    import pdb
    app.run()
    #pdb.run('app.run()')

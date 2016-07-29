"""
@author: Philipp Lucas

Provides a webservice to query graphical models at the route '/webservice'. 
Run this script to start the server locally!
"""

from flask import Flask, request
from flask_cors import CORS, cross_origin
import logging
import json
import modelbase as mbase
import traceback

app = Flask(__name__, static_url_path='/static/')
mb = mbase.ModelBase("Philipps ModelBase")
logging.basicConfig(
    level = logging.WARNING,
    format = '%(asctime)s %(levelname)s %(filename)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# the (static) start page
@app.route('/')
def index():
   return app.send_static_file('index.html')

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
      
# webservice interface to the model base
@app.route('/webservice', methods=['GET', 'POST'])
@cross_origin() # allows cross origin requests
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
          # process           
          status, result = mb.execute(query)
          logger.info('status of query:' + str(status))
          logger.info('result of query:' + str(result))
          # return answer as a serialized json
          return json.dumps( {"status":status, "result": result} )
      except Exception as inst:
          msg = "failed to execute query: " + str(inst)
          logger.error(msg)
          logger.error(traceback.format_exc())
          return json.dumps( {"status":"error", "result": msg} )

# webservice interface that returns a valid sample query 
@app.route('/sample_query', methods=['GET', 'POST'])
def sample_query():
    if request.method == 'GET':
        return "send a POST request to this url to get a valid query that you can use at the '/webservice' interface"
    else:
        #filePath = 'test-model-query_02.json'
        filePath = 'test-model-query_03.json'        
        #filePath = 'test-show-query.json'
        #filePath = 'test-predict-query_02.json'
        #filePath = 'test-predict-query_03.json'
        # open file, read as json
        query = json.load( open(filePath) )
        # serialize to string and return
        return json.dumps(query)

# the webclient
@app.route('/client', methods=['GET'])
def client():
    return app.send_static_file('client.html')       
    
# example for dynamic routes
@app.route('/user/<username>')
def show_profile(username):
   return 'User %s' % username  

if __name__ == "__main__":
    from functools import reduce    
    import pdb    
    #pdb.run('app.run()')
    app.run()    
from flask import Flask, request
from flask_cors import CORS, cross_origin
import logging
import json
import modelbase as mb


logger = logging.getLogger(__name__)
app = Flask(__name__, static_url_path='/static/')
mb = mb.ModelBase("Philipps ModelBase")

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
@cross_origin()
def service():
   # return usage information
   if request.method == 'GET':
      return "send a POST request to this url containing your model query and you will get your answer back :-)"
   # handle model request
   else:
      # extract json formatted query
      query = request.get_json() 
      logger.info('received query: ' + str(query))
      # process     
      status, result = mb.execute(query)
      logger.info('status of query: ' + str(status))
      logger.info('result of query: ' + str(result))
      # return answer as a serialized json
      return json.dumps( {"status":status, "result": result} )

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
    import pdb
    from functools import reduce
    logger.setLevel(logging.INFO)
    pdb.run('app.run()')    
    #app.run()
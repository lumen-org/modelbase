from flask import Flask, request
import json

app = Flask(__name__, static_url_path='/static/')

# the start page
@app.route('/')
def index():
   return app.send_static_file('index.html')

# the actual webservice interface
@app.route('/webservice', methods=['GET', 'POST'])
def service():
   # return usage information
   if request.method == 'GET':
      return "send a POST request to this url containing your model query and you will get your answer back :-)"
   # handle model request
   else:
      # extract json formatted query
      query = request.get_json() 
      print(query)
      # validate query
      # ...
      # process     
      result = '{"age":[0,5,2,3,2,561,0], "income":[1,2,3,4,5,6,7]}'
      # return answer      
      return result

@app.route('/user/<username>')
def show_profile(username):
   return 'User %s' % username  

#if __name__ == "__main__":
#    app.run()

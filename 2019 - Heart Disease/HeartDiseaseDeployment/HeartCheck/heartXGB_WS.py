from flask import Flask
from flask import request
from flask_cors import CORS
from heartXGB import predict
from heartXGB import addToModel
from heartXGB import resetModel
app = Flask(__name__)
CORS(app)
@app.route('/heart', methods=['POST'])
def post():
    data = request.get_json()
    oper = data['oper']
    del data['oper']
    if oper == 'predict':
      result = predict(data)
    elif oper == 'addToModel':
      result = addToModel(data)
    elif oper == 'resetModel':
      result = resetModel()
    else:
      result = None
    return result
#app.run(debug=False, host='127.0.0.1', port=5000)
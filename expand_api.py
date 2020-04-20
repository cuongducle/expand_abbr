from flask import request, Flask, jsonify
from flask_cors import CORS
from eval_teacher import expand
from flask import jsonify
import pickle


expand("tôi là người ~ vn # .")


app = Flask(__name__)
CORS(app)

@app.route('/expand', methods=["POST"])
def service():
    content = request.json['mytext']
    result = expand(content)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port="5050")

from flask import request, Flask, jsonify
from eval_teacher import expand
from flask import jsonify
import pickle


expand("tôi là người ~ vn # .")


app = Flask(__name__)

@app.route('/expand', methods=["POST"])
def service():
    content = request.json['mytext']
    result = expand(content)
    return jsonify(result)

#app.run('0.0.0.0', port=5050)

from flask import request, Flask, jsonify
from eval_teacher import expand
from flask import jsonify
import pickle
import time

expand("tôi là người ~ vn # .")


app = Flask(__name__)

@app.route('/expand', methods=["POST"])
def service():
    content = request.json['mytext']
    time_start = time.time()
    result,score = expand(content)
    time_pred = time.time() - time_start
    out = {"expand":result,"score":score,"time":time_pred}
    return jsonify(out)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="5050")

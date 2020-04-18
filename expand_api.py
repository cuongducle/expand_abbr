from flask import request, Flask, jsonify
from eval_teacher import evaluate
from flask import jsonify

evaluate("tôi là người ~ vn # .")[1].item()

app = Flask(__name__)

@app.route('/expand', methods=["POST"])
def service():
    content = request.json['mytext']
    result = evaluate(content)
    expand = result[0]
    score = result[1].item()
    time = result[2]
    out = {"expand": expand, "score": score,"time": time}
    return jsonify(out)

#app.run('0.0.0.0', port=5050)

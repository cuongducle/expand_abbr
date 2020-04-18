from flask import request, Flask, jsonify
from eval_teacher import evaluate

print(evaluate("tôi là người ~ vn # .")[0])

app = Flask(__name__)

@app.route('/expand', methods=["POST"])
def service():
    # index_input = request.json['text']
    # text = str(index_input)
    content = request.json['mytext']
    result = evaluate(content)[0]
    #result = print(text)
    return result

# app.run('0.0.0.0', port=5050)

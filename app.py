from flask import Flask, render_template, request, jsonify
import base64, io, json, os
from dotenv import load_dotenv
load_dotenv()
from agents.non_dl_agent import NonDLAgent
from agents.dl_agent import DLAgent

app = Flask(__name__)

non_dl_agent = NonDLAgent()
dl_agent = DLAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plan', methods=['POST'])
def plan():
    data = request.json
    lat = data.get('lat')
    lng = data.get('lng')
    map_image_b64 = data.get('map_image')  # base64 PNG of map crop
    agent_type = data.get('agent', 'dl')   # 'dl' or 'non_dl'
    preferences = data.get('preferences', {})

    if map_image_b64:
        img_bytes = base64.b64decode(map_image_b64.split(',')[-1])
    else:
        img_bytes = None

    if agent_type == 'non_dl':
        result = non_dl_agent.run(lat, lng, img_bytes, preferences)
    else:
        result = dl_agent.run(lat, lng, img_bytes, preferences)

    return jsonify(result)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    from evaluation.evaluator import run_evaluation
    results = run_evaluation()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

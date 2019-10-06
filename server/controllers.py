from server import app, db
from flask import request, jsonify
from datetime import datetime, timedelta
from server.models.conv_net import *

cnn = CNN({}, 'test')

@app.route('/api/classify', methods=['GET'])
def get_classify():
    data = request.get_json(force=True)
    image = data['image']
    h = cnn.predict(image)
    return jsonify({'result': h})

@app.route('/api/entry', methods=['POST'])
def post_entry():
    pass

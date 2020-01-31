from server import app, db
from flask import request, jsonify
from datetime import datetime, timedelta
from server.model.NN import NN
from scipy.misc import imread

cnn = NN.load_weights()

@app.route('/api/classify', methods=['POST'])
def classify():
  data = request.data
  im = imread
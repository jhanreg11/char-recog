from server import app
from flask import request, jsonify
import numpy as np
from server.model.nn import NN
from server.model.preprocessor import Preprocessor

cnn = NN.load_weights()
vocab = [c for c in '0123456789']
prep = Preprocessor()

@app.route('/classify', methods=['POST'])
def classify():
  data = request.get_json(force=True)['imgURI']
  nn_input = prep.preprocess(data)
  out = cnn.ff(nn_input)
  predicted_character = vocab[np.argmax(out)]

  if np.max(out) < 0.15:
    predicted_character = 'A scribble'
  return jsonify({'result': predicted_character})

@app.route('/health-check', methods=['POST', 'GET'])
def health_check():
  return 'I\'m alive'

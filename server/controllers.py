from server import app
from flask import request, jsonify
from server.model.nn import NN
import numpy as np

cnn = NN.load_weights()
vocab = [c for c in '0123456789']
print(len(vocab))
@app.route('/classify', methods=['POST'])
def classify():
  data = request.get_json(force=True)

  nn_input = np.array(data['pixels']).reshape((1, 1, 28, 28)).astype(np.float32)
  nn_input /= 255

  out = cnn.ff(nn_input)
  predicted_character = vocab[np.argmax(out)]

  return jsonify({'result': predicted_character})

@app.route('/health-check', methods=['POST', 'GET'])
def health_check():
  return 'I\'m alive'

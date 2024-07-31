
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin  # Import both CORS and cross_origin

import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Initialize a global model
global_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

weights = global_model.get_weights()

@app.route('/update_model', methods=['POST'])
@cross_origin()  # This decorator is now properly imported
def update_model():
    global weights
    try:
        client_weights = request.json['weights']
        client_weights = [np.array(w) for w in client_weights]

        # Federated averaging
        weights = [(w1 + w2) / 2 for w1, w2 in zip(weights, client_weights)]
        global_model.set_weights(weights)
        return jsonify(success=True)
    except Exception as e:
        print(f"Error in /update_model: {e}")
        return jsonify(error=str(e)), 400

@app.route('/get_model', methods=['GET'])
@cross_origin()
def get_model():
    global weights
    try:
        return jsonify(weights=[w.tolist() for w in weights])
    except Exception as e:
        print(f"Error in /get_model: {e}")
        return jsonify(error=str(e)), 400

@app.route('/evaluate_model', methods=['GET'])
@cross_origin()
def evaluate_model():
    try:
        # Load test data
        test_df = pd.read_csv('../client/mnist_data/test.csv')
        x_test = test_df.drop('label', axis=1).values
        y_test = test_df['label'].values

        # Preprocess test data
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

        # Evaluate the global model
        global_model.set_weights(weights)
        loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        return jsonify(loss=loss, accuracy=accuracy)
    except Exception as e:
        print(f"Error in /evaluate_model: {e}")
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
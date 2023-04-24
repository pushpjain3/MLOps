from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model = joblib.load('saved_model.joblib')
    
    # Get the data from the POST request
    data = request.get_json()
    
    # Convert the data into a numpy array
    data = np.array(list(data.values()))
    
    # Make the prediction using the loaded model
    prediction = model.predict(data.reshape(1, -1))
    
    # Return the prediction as a JSON response
    print(prediction)
    output = {'prediction': int(prediction[0])}
    return jsonify(output)


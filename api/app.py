import json
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    form_data = request.form

    # Store form data in a JSON file
    with open('data.json', 'w') as json_file:
        json.dump(form_data, json_file)

    # Placeholder prediction
    prediction_result = 'Some prediction result'

    # Perform prediction with ML model using form_data

    # Return the prediction result
    return jsonify({'prediction': prediction_result})

@app.route('/')
def serve_form():
    return open('form.html').read()

if __name__ == '__main__':
    app.run()

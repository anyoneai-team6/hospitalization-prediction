from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    form_data = request.form

    # Perform prediction with ML model using form_data

    # Return the prediction result
    return jsonify({'prediction': prediction_result})

@app.route('/')
def serve_form():
    return open('form.html').read()

if __name__ == '__main__':
    app.run()

import json
from flask import Flask, request, jsonify
from middleware import model_predict

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

app = Flask(__name__)

# @app.route('/predict', methods=["GET", "POST"])
# def predict():
#     if request.method == "GET":
#         return open('form.html').read()
    
#     if request.method == "POST":
#         form_data = request.form
#         # pred, score =model_predict(form_data)
#         with open('data.json', 'w') as json_file:
#             json.dump(form_data, json_file)
        
#         return redirect(request.url)

@app.route('/', methods=['GET', 'POST'])
def serve_form():
    if request.method == "GET":
        return open('form.html').read()
    elif request.method == "POST":
        form_data = request.form
        
        pred, score =model_predict(form_data)
        
        with open('data.json', 'w') as json_file:
            json.dump(form_data, json_file)
        return redirect(request.url)
    else:
        return "Method Not Allowed", 405

if __name__ == '__main__':
    app.run()

import json
from flask import Flask, request, render_template
from middleware import model_predict

app = Flask(__name__, template_folder='./templates')

@app.route('/', methods=['GET', 'POST'])
def serve_form2():
    if request.method == 'POST':
        data_dict = request.form.to_dict()
        pred,prob=model_predict(data_dict)
        return render_template('predict.html', prob=prob)
    else:
        return render_template('form.html')

if __name__ == '__main__':
    app.run()
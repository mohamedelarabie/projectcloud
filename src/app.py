import numpy as np
import pandas as pd
import model as MM
from model import BertClassifier
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)

import torch
model = BertClassifier(freeze_bert=True, version="mini")
model.load_state_dict(torch.load('model.pth' , map_location=torch.device('cpu')))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form.values()
    output = MM.predict_single_full_name(int_features,model )


    return render_template('index.html', prediction_text= output)


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    output = MM.predict_single_full_name(int_features,model)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
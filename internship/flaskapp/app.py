# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:05:00 2020

@author: Anusha
"""


import pandas as pd
import numpy as np
from flask import Flask, request, render_template,jsonify,redirect,url_for
app = Flask(__name__) 
import pickle
model = pickle.load(open('ckd.pkl','rb'))
@app.route('/')
def root():
    return render_template('root.html')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[float(x) for x in request.form.values()]]
    print(x_test)
   
    
   
    
    prediction= model.predict(x_test)
    print(prediction)
    output=prediction[0]
    
    if(output==0):
        pred= "have ckd.Medical checkup needed"
        print
    else:
        pred="do not have ckd. congratulations,you are healthy"
    
    return render_template('predict.html', prediction_text='You {}'.format(pred))
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
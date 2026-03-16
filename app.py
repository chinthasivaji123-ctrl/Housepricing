import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import pickle
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

app =Flask(__name__)
#load the model
regmodel=pickle.load(open('regression.pkl','rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))
@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
  data=request.json['data']
  print(data)
  print(np.array(list(data.values())).reshape(1,-1))
  new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
  output = regmodel.predict(new_data)
  print(output[0])
  return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
  data=[float(x) for x in request.form.values()]
  final_input=scalar.transform(np.array(data).reshape(1,-1))
  print(final_input)
  output=regmodel.predict(final_input)[0]
  actual_price=output*1000
  formatted_price="${:,.2f}".format(actual_price)
  return render_template("home.html",prediction_text="the house price prediction is {}".format(formatted_price))
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

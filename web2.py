from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

app = Flask(__name__)

with open('model_pkl','rb') as f:
    loaded_model=pickle.load(f)

with open('model_pkl2','rb') as f:
    loaded_model2=pickle.load(f)

def model_arima(date):
    target_date = pd.Timestamp(date) 
    future_index = pd.date_range(start=loaded_model.fittedvalues.index[-1] + pd.DateOffset(1), end=target_date, freq='B')
    forecast_steps = len(future_index)
    forecast = loaded_model.get_forecast(steps=forecast_steps, alpha=0.05)
    pred_mean = forecast.predicted_mean
    pred_mean.index = future_index
    nearest_index = (pred_mean.index - target_date).argmin()
    predicted_value = pred_mean.iloc[nearest_index]
    return np.exp(predicted_value)

def create_dataset(dataset, timestep=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-timestep-1):
        a = dataset[i:(i+timestep)]
        dataX.append(a)
        dataY.append(dataset[i+timestep])
    return np.array(dataX), np.array(dataY)

data = pd.read_csv(r'C:\Users\sayan\OneDrive\Desktop\New\AAPL.csv')
data['Date'] = pd.to_datetime(data['Date'], format="%d-%m-%Y")
data.index = data.pop('Date')
trainData = np.log(data['Close'])
X_train, Y_train = create_dataset(trainData)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

def model_lstm(date):
    future_date = pd.to_datetime(date) 
    last_timestep = trainData[-X_train.shape[1]:].values.reshape(1, -1, 1) 
    prediction = np.exp(loaded_model2.predict(last_timestep))
    return prediction

@app.route('/')
def welcome():
    return render_template('index2.html')

@app.route('/plot_data_arima',methods=['POST'])
def plot_data_arima():
    if request.method=='POST':
        date=request.form['col']
        prediction_val = model_arima(date)
        return render_template('result2.html', prediction_values=prediction_val, modelmethod="ARIMA", date=date)

@app.route('/plot_data_lstm',methods=['POST'])
def plot_data_lstm():
    if request.method=='POST':
        date=request.form['col']
        prediction_val=model_lstm(date)
        return render_template('result2.html',prediction_values=prediction_val,modelmethod="LSTM",date=date)

if __name__ == '__main__':
    app.run(debug=True,port=80)
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

app = Flask(__name__)


#ARIMA
def stock_prediction_arima(file, col, years):
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'],format="%d-%m-%Y")
    data.index = data.pop('Date')
    trainData = np.log(data[col])
    model =pm.auto_arima(trainData, trace=False, suppress_warnings=True)
    best_order = model.get_params()['order']
    resultModel = ARIMA(trainData, order=best_order)
    fitted = resultModel.fit()
    forecast_index = pd.date_range(start=trainData.index[-1], periods=years * 252, freq='B')
    pred = fitted.get_forecast(steps=len(forecast_index), alpha=0.05).predicted_mean
    pred.index = forecast_index
    return np.exp(pred),data[col]

#LSTM
def stock_prediction_lstm(file, col, years):
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'], format="%d-%m-%Y")
    data.index = data.pop('Date')
    df = data[[col]].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)
    training_size = int(len(df_scaled) * 0.8)
    training_data = df_scaled[:training_size]
    
    def create_sequences(dataset, time_steps=1):
        x_data, y_data = [], []
        for i in range(len(dataset) - time_steps):
            x_data.append(dataset[i:(i + time_steps), 0])
            y_data.append(dataset[i + time_steps, 0])
        return np.array(x_data), np.array(y_data)

    time_steps = 100 
    x_train, y_train = create_sequences(training_data, time_steps)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=20, batch_size=32)

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=years * 252, freq='B')
    future_data = pd.DataFrame(index=future_dates, columns=[col])
    input_sequence = df_scaled[-time_steps:].reshape(1, -1, 1)
    predictions = []
    for _ in range(len(future_dates)):
        prediction = model.predict(input_sequence)
        predictions.append(prediction[0, 0])
        input_sequence = np.roll(input_sequence, -1)
        input_sequence[-1] = prediction

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_data[col] = predictions
    combined_data = pd.concat([data, future_data], axis=0)
    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction using LSTM')
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.plot(combined_data[col], label='Actual Data')
    plt.plot(combined_data[col].iloc[-len(future_data):], label='Predicted Data', color='red')
    plt.legend()
    plt.show()

    return combined_data[col][-len(future_data):]



@app.route('/')
def welcome():
    return render_template('index.html')
#ARIMA
@app.route('/plot_data', methods=['POST'])
def plot_data():
    if request.method == 'POST':
        #file_path = request.form['File']
        column_name = request.form['col']
        prediction_years = int(request.form['prediction_years'])
        prediction_values,trainData = stock_prediction_arima(r'C:\Users\sayan\OneDrive\Desktop\New\AAPL.csv', column_name, prediction_years)
        #prediction_values,trainData = stock_prediction(file_path, column_name, prediction_years)
        prediction_tuples = list(zip(prediction_values.index, prediction_values))
        trainData.plot(label="Train Data",color="blue")
        prediction_values.plot(label="Predicted data",color="red")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title('Stock Prediction')
        plt.legend()
        plt.plot(figsize=(20,10))
        img='static/plot.png'
        plt.savefig(img)
        plt.close()
        return render_template('result.html',prediction_values=prediction_tuples,img=img,modelmethod="ARIMA")

#LSTM
@app.route('/plot_data_lstm', methods=['POST'])
def plot_data_lstm():
    if request.method == 'POST':
        #file_path = request.form['File']
        column_name = request.form['col']
        prediction_years = int(request.form['prediction_years'])
        prediction_values,trainData = stock_prediction_lstm(r'C:\Users\sayan\OneDrive\Desktop\New\AAPL.csv', column_name, prediction_years)
        #prediction_values,trainData = stock_prediction(file_path, column_name, prediction_years)
        prediction_tuples = list(zip(prediction_values.index, prediction_values))
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(trainData, label="Train Data", color="blue")
        ax.plot(prediction_values.index, prediction_values, label="Predicted data", color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel(column_name)  
        ax.set_title('Stock Prediction using LSTM')
        ax.legend(loc='lower right')
        img = 'static/plot.png'
        plt.savefig(img)
        plt.close()
        return render_template('result.html',prediction_values=prediction_tuples,img=img,modelmethod="LSTM")
        


if __name__ == '__main__':
    app.run(debug=True)
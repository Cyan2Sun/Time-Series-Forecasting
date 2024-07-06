import pandas as pd
import numpy as np
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
'''
def stock_prediction(file,col,years,date_format='%d-%m-%Y'):
    data=pd.read_csv(file)
    data['Date']=pd.to_datetime(data['Date'],format=date_format)
    data.index=data.pop('Date')
    trainData=np.log(data[col])
    model=pm.auto_arima(trainData,trace=False,suppress_warnings=True)
    best_order = model.get_params()['order']
    resultModel=ARIMA(trainData,order=best_order)
    fitted=resultModel.fit()
    forecast_index = pd.date_range(start=trainData.index[-1], periods=years*252, freq='B')  
    pred = fitted.get_forecast(steps=len(forecast_index), alpha=0.05).predicted_mean
    pred.index = forecast_index
    print("\nPredicted Values: ")
    print(np.exp(pred))
    print("\n")
    trainData.plot(label='Train data',color='blue')
    pred.plot(label='Predicted data',color='red')
    plt.legend(loc='best')
    plt.plot(figsize=(12,10))
    plt.show()
'''
#LSTM
'''
def create_dataset(dataset,timestep=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-timestep-1):
        a=dataset[i:(i+timestep)]
        dataX.append(a)
        dataY.append(dataset[i+timestep])
    return np.array(dataX),np.array(dataY)

def stock_prediction_lstm(file, col, years):
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'],format="%d-%m-%Y")
    data.index = data.pop('Date')
    trainData = np.log(data[col])
    X_train,Y_train=create_dataset(trainData)
    X_test,Y_test=create_dataset(testData)         
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict_series = pd.Series(train_predict.flatten(), index=trainData.index[:len(train_predict)])
    test_predict_series = pd.Series(test_predict.flatten(), index=testData.index[:len(test_predict)])
    #Plot the result
    trainData.plot(label='Train data',color='blue')
    testData.plot(label='Test data',color='green')
    #train_predict_series.plot(label='Predicted Train data',color='red')
    test_predict_series.plot(label='Predicted Test data',color='red')
    plt.legend(loc='best')
    plt.plot(figsize=(20,10))
    plt.show()
    #Accuracy check
    mse=mean_squared_error(Y_test,test_predict)
    mae=mean_absolute_error(Y_test,test_predict)
    rmse=math.sqrt(mse)
    mape=np.mean(np.abs(test_predict-Y_test)/np.abs(Y_test))
    print("Accuracy Metrics: ")
    print("MSE: ",mse)
    print("MAE: ",mae)
    print("RMSE: ",rmse)
    print("MAPE: ",mape)
    print("Model is ",round(100-(mape*100),2),"% accurate.")

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_dataset(dataset, timestep=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-timestep-1):
        a = dataset[i:(i+timestep), 0]  # Adjusted to select the first (and only) feature
        dataX.append(a)
        dataY.append(dataset[i+timestep, 0])  # Adjusted to select the first (and only) feature
    return np.array(dataX), np.array(dataY)

def stock_prediction_lstm(file, col, train_years, test_years, future_steps):
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'], format="%d-%m-%Y")
    data.index = data.pop('Date')
    
    train_data = data.iloc[:-(train_years * 252)][col].values.reshape(-1, 1)
    test_data = data.iloc[-(test_years * 252):][col].values.reshape(-1, 1)

    train_data_log = np.log(train_data)
    test_data_log = np.log(test_data)

    X_train, Y_train = create_dataset(train_data_log)
    X_test, Y_test = create_dataset(test_data_log)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64, verbose=1)
    
    future_input = X_test[-1:]

    # Generate future predictions
    future_predictions = []
    for _ in range(future_steps):
        future_prediction = model.predict(future_input)
        future_predictions.append(future_prediction.flatten()[0])
        future_input = np.append(future_input[:, 1:], future_prediction.reshape(1, 1, 1))

    if np.min(data[col]) > 0:
        future_predictions = np.exp(future_predictions)

    # Plotting
    train_data = data.iloc[:-(train_years * 252)][col]
    test_data = data.iloc[-(test_years * 252):][col]
    train_data.plot(label='Train data', color='blue')
    test_data.plot(label='Test data', color='green')
    plt.plot(np.arange(len(test_data), len(test_data) + future_steps), future_predictions, label='Future Predictions', color='orange')
    
    plt.legend(loc='best')
    plt.show()

stock_prediction_lstm(r'C:\Users\sayan\OneDrive\Desktop\New\AAPL.csv', 'Close',30,5, 20)




import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("Imported libraries succesfully")
# load the dataset

#For other stocks, change directory accordingly
df = pd.read_csv("STOCK A/A.csv", na_values=['null'], index_col='Date', parse_dates=True, infer_datetime_format=True)
print("Dataframe Shape: ", df.shape)

# check for null values
print("Null Value Present: ", df.isnull().values.any())
df = df.dropna()
print("Dataframe Shape: ", df.shape)
print("Null Value Present: ", df.isnull().values.any())

# extract features and output variable
output_var = pd.DataFrame(df['Adj Close'])
features = ['Open', 'High', 'Low', 'Volume']
print("Target variable defined")

# scale the features
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)
print("Features scaled using minmaxscaler")

# create lagged features using XGBoost
X, y = feature_transform, output_var.values.ravel()
data_dmatrix = xgb.DMatrix(data=X,label=y)
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print((cv_results["test-rmse-mean"]).tail(1))

# select training and testing data
timesplit = TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
    y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()
print("Split Testing and training data")

# reshape the data for LSTM model
trainX, testX = np.array(X_train), np.array(X_test)
trainX = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
testX = testX.reshape(X_test.shape[0], 1, X_test.shape[1])
trainY, testY = np.array(y_train), np.array(y_test)

# build the LSTM model
print("Trying to create a Recurrent Neural Network")
lstm = Sequential()
lstm.add(LSTM(16, input_shape=(1, trainX.shape[2]), activation='relu', return_sequences=True))
lstm.add(Dropout(0.5))
lstm.add(LSTM(8,activation='relu',return_sequences=False))
lstm.add(Dropout(0.5))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mse'])
print("LSTM compile success")
lstm.summary()

# fit the LSTM model
history = lstm.fit(trainX, trainY, epochs=50, batch_size=8, verbose=1, shuffle=False, validation_split=0.15)

# make predictions
trainPredict = lstm.predict(trainX)
testPredict = lstm.predict(testX)

# calculate metrics
trainScore = mean_squared_error(trainY, trainPredict)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))
testScore = mean_squared_error


plt.style.use('dark_background')
plt.plot(y_test, label='True Value',color='yellow')
plt.plot(testPredict, label='XGBoost-LSTM Value',color='cyan')
plt.legend()
plt.ylabel("Values in INR ₹")
plt.title("LSTM prediction with dropout for 50epochs")
plt.savefig("Stock A LSTM with dropout")
plt.show()

predicted_returns = trainPredict[1:]/trainPredict[:-1] - 1
actual_returns = y_test[1:]/y_test[:-1] - 1


sharpe_ratio_predicted = predicted_returns.mean()/predicted_returns.std()
sharpe_ratio_actual = actual_returns.mean()/actual_returns.std()
print("Sharpe Ratio for chosen stock for chosen stock is :",sharpe_ratio_predicted*(252**0.5))

sortino_ratio_predicted = predicted_returns.mean() / np.sqrt(np.mean(np.minimum(predicted_returns, 0) ** 2))
sortino_ratio_actual = actual_returns.mean() / np.sqrt(np.mean(np.minimum(actual_returns, 0) ** 2))
print("Sortino ratio for chosen stock for chosen stock is :",sortino_ratio_predicted)

beta = linear_model.LinearRegression().fit(trainPredict.reshape(-1, 1), y_train).coef_[0]
treynor_ratio_predicted = (predicted_returns.mean() - 0.01)/beta
treynor_ratio_actual = (actual_returns.mean() - 0.01)/beta
print("Treynor Ratio for chosen stock for chosen stock is :",treynor_ratio_predicted)

cumulative_returns_predicted = np.cumprod(1 + predicted_returns)[-1]
cumulative_returns_actual = np.cumprod(1 + actual_returns)[-1]
print("Cumulative returns are : ",cumulative_returns_predicted)

cumulative_predicted = np.cumprod(1 + predicted_returns) - 1
cumulative_actual = np.cumprod(1 + actual_returns) - 1
max_drawdown_predicted = np.max(np.abs(np.maximum.accumulate(cumulative_predicted) - cumulative_predicted))
max_drawdown_actual = np.max(np.abs(np.maximum.accumulate(cumulative_actual) - cumulative_actual))
print("Max Drawdown for chosen stock for chosen stock is :",max_drawdown_predicted)

highest_return_predicted = np.max(predicted_returns)
highest_return_actual = np.max(actual_returns)
print("Highest actual return for chosen stock for chosen stock is :",highest_return_predicted)
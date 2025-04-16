from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
# 设置 TensorFlow 随机数种子
#tf.random.set_seed(42)

#从文件加载数据
data = pd.read_excel(r'C:\Users\hongm\OneDrive\桌面\预测作业\Code_Examination\工作簿4.xlsx')

#获取特征和目标变量
X= data.drop("Target",axis=1)
y = data['Target']

#data = pd.read_csv(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\YularaData2.csv')
#.set_index('date',inplace=True)
#获取特征和目标变量
#X= data.drop(['Power', 'Time'],axis=1)
#y = data['Power']
# Data normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Function to create multi-step dataset for BPNN
def create_multistep_dataset(X, y, steps):
    X_multi, y_multi = [], []
    for i in range(len(y) - steps):
        X_multi.append(X[i:i + steps])  # Multi-step input features
        y_multi.append(y[i + steps])    # Single-step output target
    return np.array(X_multi), np.array(y_multi)

# Set steps to 6
steps = 6

# Create regression dataset
X_multi, y_multi = create_multistep_dataset(X, y, steps)
print(X_multi, y_multi)
# Splitting the dataset into train and test sets
n_samples_total = len(y_multi)
train_size = int(n_samples_total * 0.8)
X_train, X_test = X_multi[:train_size], X_multi[train_size:]
y_train, y_test = y_multi[:train_size], y_multi[train_size:]

# 设置特征数为20*************************************************************************一定要改自变量数量
n_features=10
# Build BPNN model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(steps * n_features,)))  # Input shape: (steps * num_features,)
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
start_time = time.time()
history=model.fit(X_train.reshape((X_train.shape[0], -1)), y_train, epochs=50, batch_size=32, verbose=0)
# Predict using the model
y_pred = model.predict(X_test.reshape((X_test.shape[0], -1)))
end_time = time.time()
# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('BPNN Regression Prediction')
plt.xlabel('Time')
plt.ylabel('CO')
plt.legend()
plt.show()

# Directional Accuracy
#方向准确性
def compute_Dstat(actual_prices, predicted_prices):
    N = len(actual_prices)
    correct_predictions = 0
    for t in range(N - 1):
        if (actual_prices[t + 1] - actual_prices[t]) * (predicted_prices[t + 1] - predicted_prices[t]) >= 0:
            correct_predictions += 1
    Dstat = (correct_predictions / (N - 1)) * 100
    return Dstat


dstat = compute_Dstat(y_test, y_pred)
print("Dstat: ", dstat)

execution_time = end_time - start_time
print(f"BPNN Execution Time: {execution_time:.2f} seconds")

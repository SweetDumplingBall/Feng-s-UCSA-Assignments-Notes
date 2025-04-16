import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

# Load data from file
data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\AQ.xlsx')
# # Get features and target variable
X = data.drop('CO(GT)',axis=1)  # Features
y = data['CO(GT)']  # Target variable 'CO(GT)'

#data = pd.read_csv(r'D:\研究生\01_班级学业工作\01_课程方面\02_预测理论与方法_2024秋_汤玲老师\YularaData.csv')

#获取特征和目标变量
#X = data.drop(['y','date'],axis=1)
#y = data['y']

#X = data.drop(['Power','Time'],axis=1)
#y = data['Power']

# Data normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 计时器开始
start_time = time.time()

# Set steps to 6
steps = 6

# Function to create multi-step dataset for LSTM
def create_multistep_dataset(X, y, steps):
    X_multi, y_multi = [], []
    for i in range(len(y) - steps):
        X_multi.append(X[i:i + steps])  # Multi-step input features
        y_multi.append(y[i + steps])    # Single-step output target
    return np.array(X_multi), np.array(y_multi)

# Create regression dataset
X_multi, y_multi = create_multistep_dataset(X, y, steps)

# Splitting the dataset into train and test sets
n_samples_total = len(y_multi)
train_size = int(n_samples_total * 0.8)
X_train_multi,X_test_multi = X_multi[:train_size],X_multi[train_size:]
y_train_multi,y_test_multi = y_multi[:train_size],y_multi[train_size:]

# 建立 LSTM model
model = Sequential()
model.add(LSTM(500, activation='relu', input_shape=(steps, X.shape[1])))  # Input shape: (steps, num_features)
model.add(Dense(500, activation='relu'))
model.add(Dense(1))  # Output layer
# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')
# 训练模型
model.fit(X_train_multi, y_train_multi, epochs=200, batch_size=32, verbose=0)
# 进行预测
y_pred = model.predict(X_test_multi)
# 计时器结束
end_time = time.time()
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


# Calculate metrics
mae = mean_absolute_error(y_test_multi, y_pred)
mse = mean_squared_error(y_test_multi, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_multi, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
dstat = compute_Dstat(y_test_multi, y_pred)
print("Dstat: ", dstat)
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} ")

# Visualize results
# 绘制实际值和预测值的对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test_multi, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.title('LSTM Regression Prediction')
plt.xlabel('Time')
plt.ylabel('T')
plt.legend()
plt.show()

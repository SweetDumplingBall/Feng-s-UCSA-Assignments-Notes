import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.svm import NuSVR
import time

#从文件加载数据
data = pd.read_excel(r'C:\Users\hongm\OneDrive\桌面\预测作业\Code_Examination\工作簿4.xlsx')

#获取特征和目标变量
X= data.drop("Target",axis=1)
y = data['Target']

#数据归一化处理
scaler=StandardScaler()
X=scaler.fit_transform(X)
#计时器开始
start_time=time.time()

#将数据集转换为监督学习问题
# 设置步长为6
steps = 6

##准备数据集，多步预测
def create_multistep_dataset(X,y, steps):
    X_multi,y_multi =[],[]
    for i in range(len(y)-steps):
        X_multi.append(X[i:i+steps]) # 多步输入特征
        y_multi.append(y[i+steps])#单步输出目标值
    return np.array(X_multi),np.array(y_multi)

#创建回归数据集
X_multi,y_multi =create_multistep_dataset(X,y,steps)

print(X_multi,y_multi)

n_samples,n_steps,n_features =X_multi.shape

X_supervised =X_multi.reshape((n_samples,n_steps * n_features))
y_supervised =y_multi

#print(X_supervised,y_supervised)

# 根据时间序列划分训练集和测试集
n_samples_total = len(y_supervised)
train_size = int(n_samples_total * 0.8)
X_train, X_test = X_supervised[:train_size], X_supervised[train_size:]
y_train, y_test = y_supervised[:train_size], y_supervised[train_size:]

#创建SVR回归模型
#model = SVR(kernel='linear',C=0.1)
# 设置 numpy 的随机种子
#np.random.seed(42)
model = SVR(kernel='rbf')
#训练模型
model.fit(X_train,y_train)
#进行预测
y_pred = model.predict(X_test)
# 计时器结束
end_time = time.time()
execution_time = end_time - start_time

print(f"SVM execution_time: {execution_time:.2f} seconds")

# 计算 MAE
mae = mean_absolute_error(y_test, y_pred)
# 计算 MSE
mse = mean_squared_error(y_test, y_pred)
# 计算 RMSE
rmse = np.sqrt(mse)
# 计算 R²
r2 =r2_score(y_test,y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R^2 Score: {r2}')

#方向准确性
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

print("Dstat: {:.2f}".format(dstat))

# 可视化结果
plt.figure(figsize=(10,6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('SVM Regression Prediction')
plt.xlabel('Time')
plt.ylabel('temp')
plt.legend()
plt.show()
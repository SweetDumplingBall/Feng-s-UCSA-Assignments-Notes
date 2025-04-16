import pandas as pd#载入pandas模块，并简称为pd
from sklearn.model_selection import train_test_split#载入train_test_split模块
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score#载入mean_squared_error, r2_score模块
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import numpy as np#载入numpy模块，并简称为np
import matplotlib.pyplot as plt#载入matplotlib.pyplot模块，并简称为plt
import time

#从文件加载数据
data = pd.read_excel(r'C:\Users\hongm\OneDrive\桌面\预测作业\Code_Examination\工作簿4.xlsx')

#X= data.drop("CO(GT)",axis=1)
#y = data['CO(GT)']


#获取特征和目标变量
X= data.drop("Target",axis=1)
y = data['Target']

#数据归一化处理
scaler=StandardScaler()
X=scaler.fit_transform(X)
#计时器开始
start_time=time.time()

# 设置步长为6
steps =6
#准备数据集，多步预测
def create_multistep_dataset(X,y,steps):
    X_multi,y_multi =[],[]
    for i in range(len(y)-steps):
        X_multi.append(X[i:i+steps]) # 多步输入特征
        y_multi.append(y[i+steps]) #单步输出目标值
    return np.array(X_multi),np.array(y_multi)
#创建回归数据集
X_multi,y_multi=create_multistep_dataset(X,y, steps)

#调整数据形状以适应决策树模型
X_multi =X_multi.reshape(X_multi.shape[0],-1)
#划分训练集和测试集
#X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi,test_size=0.2,random_state=42)
#根据时间序列划分训练集和测试集
n_samples_total=len(y_multi)
train_size =int(n_samples_total *0.8)
X_train_multi, X_test_multi =X_multi[:train_size], X_multi[train_size:]
y_train_multi, y_test_multi=y_multi[:train_size], y_multi[train_size:]

#创建决策树回归模型
#model = DecisionTreeRegressor(random_state=42)
model = DecisionTreeRegressor()
#训练模型
model.fit(X_train_multi, y_train_multi)
#进行预测
y_pred = model.predict(X_test_multi)
#计时器结束
end_time = time.time()
print(end_time)

mae = mean_absolute_error(y_test_multi, y_pred)  # 计算平均绝对误差(MAE)
mse = mean_squared_error(y_test_multi, y_pred)  # 计算均方误差(MSE)
rmse = np.sqrt(mse)  # 计算均方根误差(RMSE)
r2 = r2_score(y_test_multi, y_pred)  # 计算R^2分数
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R^2 Score: {r2}')

# 定义并计算方向准确性
#方向准确性
def compute_Dstat(actual_prices, predicted_prices):
    N = len(actual_prices)
    correct_predictions = 0
    for t in range(N - 1):
        if (actual_prices[t + 1] - actual_prices[t]) * (predicted_prices[t + 1] - predicted_prices[t]) >= 0:
            correct_predictions += 1
    Dstat = (correct_predictions / (N - 1)) * 100
    return Dstat

d_stat = compute_Dstat(y_test_multi, y_pred)  # 计算方向准确性(Dstat)
print(f'Direction Accuracy (Dstat): {d_stat}')

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(y_test_multi, label='Actual')  # 绘制实际值
plt.plot(y_pred, label='Predicted')  # 绘制预测值
plt.title('Decision Tree Regression Prediction')  # 设置图形标题
plt.xlabel('Time')  # 设置X轴标签
plt.ylabel('CO')  # 设置Y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形
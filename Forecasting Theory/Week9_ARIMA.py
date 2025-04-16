import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox #模型残差检验
from sklearn.metrics import mean_absolute_error, mean_squared_error ,r2_score


data = pd.read_excel(r'C:\Users\hongm\OneDrive\桌面\预测作业\Code_Examination\工作簿3.xlsx')
#确定是否平稳，用几阶差分，差分后是否平稳
'''
#平稳性检验：时序图检验
data['CarbonP'].plot(figsize=(30,4))
#平稳性检验：自相关图检验
plot_acf(data['CarbonP'],lags=40)
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.title('ACF')
plt.show()
#随机性：单位根检验
adf_test = adfuller(data['CarbonP'])
print("Level ADF statistic:",adf_test[0])
print("Level P-value:",adf_test[1])
'''

#非平稳需要做差分处理
#一阶差分
#data['Diff_CarbonP'] = data['CarbonP'].diff()

power=data['sale']
# 确保数据列是数值类型
#data['CarbonP']= pd.to_numeric(data['CarbonP'], errors='coerce')

train_size = int(len(power) * 0.8)
train_data = power[:train_size]
test_data = power[train_size+1:]

#ARIMA模型寻优

train_results = sm.tsa.arma_order_select_ic(train_data, ic=['aic', 'bic'], trend='n', max_ar=5, max_ma=5)
print('AIC 最优参数:', train_results.aic_min_order)
print('BIC 最优参数:', train_results.bic_min_order)


#模型构建
model=ARIMA(train_data,order=(1,0,1))
model_fit=model.fit()
print(model_fit.summary())

#模型残差检验
print(acorr_ljungbox(model_fit.resid,lags=[10],boxpierce=True))

#模型预测
start_idx = len(train_data)
end_idx = len(train_data) + len(test_data) - 1
predicted = model_fit.predict(start=start_idx, end=end_idx)

#绘制实际价格和预测价格的图表
plt.figure(figsize=(8,5))#可以指定图表的大小
plt.plot(test_data.index,test_data,label='actual')
plt.plot(predicted.index,predicted,label='predict')
plt.legend(['actual','predict'])
plt.xlabel('Time(Day)')
plt.ylabel('Price')
plt.title('Actual vs Predicted Carbon Price')
plt.show()

# 模型评价
mae = mean_absolute_error(test_data, predicted)
mse = mean_squared_error(test_data, predicted)
rmse = np.sqrt(mse)
r2_best = r2_score(test_data, predicted)
# 打印结果
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print(f'R2S:{r2_best}')


# 对模型方向性进行评价
def compute_Dstat(actual_prices, predicted_prices):
    N = len(actual_prices)
    correct_predictions = 0
    for t in range(N - 1):
        if (actual_prices[t + 1] - actual_prices[t]) * (predicted_prices[t + 1] - predicted_prices[t]) >= 0:
            correct_predictions += 1
    Dstat = (correct_predictions / (N - 1))
    return Dstat


# 检查并转换数据类型
if not isinstance(test_data, np.ndarray):
    test_data = np.array(test_data)
if not isinstance(predicted, np.ndarray):
    predicted = np.array(predicted)


# 检查并处理数据长度
if len(test_data)!= len(predicted):
    print("Warning: Data length mismatch.")
    # 这里可以根据实际情况处理长度不匹配的问题，例如截取较长的数组
    min_length = min(len(test_data), len(predicted))
    test_data = test_data[:min_length]
    predicted = predicted[:min_length]


# 检查并处理 NaN 值
test_data = np.nan_to_num(test_data)
predicted = np.nan_to_num(predicted)


dstat = compute_Dstat(test_data, predicted)
print("Dstat:", dstat)
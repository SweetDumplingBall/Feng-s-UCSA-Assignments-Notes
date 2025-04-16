#1、观察时序的平稳性/随机性/季节性
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

#导入数据
data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\Champagne_Sales_Prices.xlsx')
price = data['Price']
'''
#2、平稳性检验
diff_price=np.diff(price)
diff_steps=12
#进行几阶差分
diff_price =data['Price'].diff(diff_steps).dropna()
#进行ADF单位根检验
adf_test = adfuller(price)
adf_test_2= adfuller(diff_price)
#打印检验的结果
print("Level ADF Statistic:",adf_test[0])
print("Level p-value:",adf_test[1])
print("diff ADF Statistic:",adf_test_2[0])
print("diff p value:",adf_test_2[1])
'''
# 确保数据列是数值类型
#data['CarbonP']= pd.to_numeric(data['CarbonP'], errors='coerce')
train_size = int(len(power) * 0.8)
train_data = power[:train_size]
test_data = power[train_size+1:]

#3、根据ACF/PACF确定模型参数
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
# 确保日期列是日期类型
data['Date']= pd.to_datetime(data['Date'])
#设置日期列为索引
data.set_index('Date',inplace=True)
#绘制ACF图
plt.figure(figsize=(12,6))
plot_acf(diff_price,lags=40)# lags参数表示你希望计算的滞后阶数
plt.show()
#绘制PACF图
plt.figure(figsize=(12,6))
plot_pacf(diff_price,lags=40) # lags参数表示你希望计算的滞后阶数
plt.show()

#4、模型构建
import itertools
#定义季节性参数P,Q,取0到2之间的任意值
P=Q=range(0,3)
s=12 #季节性周期，这里设置为12,表示年度季节性
#生成P和Q的组合，并创建季节性参数列表
seasonal_pdq =[(x[0],0,x[1],s)for x in list(itertools.product(P,Q))]
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 对所有的季节性参数组合进行遍历
for seasonal_param in seasonal_pdq:
    try:
        # 建立季节性 ARIMA 模型
        model = SARIMAX(endog=diff_price,seasonal_order=seasonal_param)
        model_fit = model.fit() #拟合模型
        aic  = model_fit.aic
        # 初始化最优 AIC 为一个大的数值
        best_aic = float('inf')
        # 如果当前模型的 AIC 小于最优 AIC，则更新最优模型、最优季节性参数和最优 AIC
        if aic < best_aic:
            best_model = model_fit
            best_seasonal_param = seasonal_param
            best_aic = aic
    except Exception as e:
        print(f"Model fitting failed for parameters {seasonal_param}: {e}")

# 循环结束后，best_model将包含最佳模型
if best_model:
    print(f"Best model AIC: {best_aic}")
    print(f"Best seasonal parameters: {best_seasonal_param}")
    #你可以用 best_model 进行进一步的分析或预测
else:
    print("No model was fitted successfully.")

#5、模型预测
#设置SARIMA模型的参数
order =(2,0,4)#p:AR阶数，d:差分阶数，q:MA阶数
seasonal_order=(2,0,1,12) #P:季节性AR阶数，D:季节性差分阶数，Q:季节性MA阶数，s:季节周期
#拟合SARIMA模型
model = SARIMAX(train_data,order=order,seasonal_order=seasonal_order)
result = model.fit()
result.summary()

#6、模型检查
from statsmodels.stats.diagnostic import acorr_ljungbox
#residuals是你从SARIMAX模型中得到的残差序列
residuals = result.resid
#进行Ljung-Box检验
lb_test_result =acorr_ljungbox(residuals,lags=[10],boxpierce=True)
print('result'+str(lb_test_result))

#7.模型评估
import numpy as np
start_idx = len(train_data)
end_idx = len(train_data) + len(test_data) - 1
predict_power = result.predict(start=start_idx, end=end_idx)
# 模型评价
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
# 模型评价
mae = mean_absolute_error(test_data, predict_power)
mse = mean_squared_error(test_data, predict_power)
rmse = np.sqrt(mse)
r2_best = r2_score(test_data, predict_power)
# 打印结果
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print(f'R2S:{r2_best}')

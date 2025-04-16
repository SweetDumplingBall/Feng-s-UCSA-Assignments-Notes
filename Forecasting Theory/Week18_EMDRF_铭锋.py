from PyEMD import EMD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 从文件加载数据
data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\AQ.xlsx')
df = data['CO(GT)']
df.dropna(inplace=True)  # 删除缺失值
co_gt = df.values

# 进行EMD分解
emd = EMD()
imfs = emd(co_gt)  # 分解为多个本征模态函数（IMF）
print(f"Number of IMFs: {len(imfs)}")

predicted_results = np.zeros_like(co_gt)

# 遍历每个IMF并应用随机森林回归
for i, imf in enumerate(imfs):
    X_train = np.arange(len(imf)).reshape(-1, 1)  # 特征变量，时间序列
    y_train = imf  # 目标变量，直接使用一维数组

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # 训练模型

    predicted_inf = rf_model.predict(X_train)  # 预测
    predicted_results += predicted_inf  # 累加预测结果

    # 绘制每个IMF及其预测结果
    plt.subplot(len(imfs) + 1, 1, i + 1)
    plt.plot(imf, 'g', label=f'IMF {i + 1}')
    plt.plot(predicted_inf, 'r', label=f'预测 IMF {i + 1}')
    plt.legend()

plt.tight_layout()
plt.show()

# 绘制原始信号和预测信号
plt.figure(figsize=(12, 6))
plt.plot(co_gt, 'b', label='原始信号')
plt.plot(predicted_results, 'r', label='预测信号')
plt.legend()
plt.show()

# 计算并打印模型性能指标（可选）
for i, imf in enumerate(imfs):
    X_train = np.arange(len(imf)).reshape(-1, 1)
    y_train = imf
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    predicted_inf = rf_model.predict(X_train)

    mae = mean_absolute_error(y_train, predicted_inf)
    mse = mean_squared_error(y_train, predicted_inf)
    r2 = r2_score(y_train, predicted_inf)
    print(f"IMF {i + 1} - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

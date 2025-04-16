import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 计算预测准确度
def compute_Dstat(actual_prices, predicted_prices):
    N = len(actual_prices)
    correct_predictions = 0
    for t in range(N - 1):
        if (actual_prices[t + 1] - actual_prices[t]) * (predicted_prices[t + 1] - predicted_prices[t]) >= 0:
            correct_predictions += 1
    Dstat = (correct_predictions / (N - 1)) * 100
    return Dstat


# 从文件加载数据
data = pd.read_excel(r'E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\AQ.xlsx')
df = data['CO(GT)']
df.dropna(inplace=True)  # 删除缺失值
co_gt = df.values

# 进行小波分解
wavelet_name = 'db4'
coeffs = pywt.wavedec(co_gt, wavelet_name, level=4)


# 初始化 predicted_results 为与 co_gt 相同的形状
predicted_results = np.zeros_like(co_gt)


# 遍历每个小波系数并应用随机森林回归
for i, coeff in enumerate(coeffs):
    # 仅使用 coeff 的长度作为特征和目标变量
    X_train = np.arange(len(coeff)).reshape(-1, 1)
    y_train = coeff

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # 训练模型

    predicted_inf = rf_model.predict(X_train)  # 预测
    # 仅在 coeff 的有效长度内累加预测结果
    predicted_results[:len(coeff)] += predicted_inf

    # 计算 RMSE 和预测准确率
    mse = mean_squared_error(y_train, predicted_inf)
    rmse = np.sqrt(mse)
    accuracy = compute_Dstat(y_train, predicted_inf)  # 计算准确率，使用自定义的 Dstat 函数

    # 绘制每个小波系数及其预测结果
    plt.subplot(len(coeffs) + 1, 1, i + 1)
    plt.plot(coeff, 'g', label=f'Coefficient {i + 1}')
    plt.plot(predicted_inf, 'r', label=f'预测 Coefficient {i + 1}')
    plt.legend()
    print(f"Coefficient {i + 1} - RMSE: {rmse:.4f}, Accuracy (Dstat): {accuracy:.4f}")


plt.tight_layout()
plt.show()


# 绘制原始信号和预测信号
plt.figure(figsize=(12, 6))
plt.plot(co_gt, 'b', label='原始信号')
plt.plot(predicted_results, 'r', label='预测信号')
plt.legend()
plt.show()


# 计算并打印模型性能指标（可选）
# 计算总体的性能指标
# 确保使用相同长度的数据进行计算
overall_mae = mean_absolute_error(co_gt[:len(predicted_results)], predicted_results)
overall_mse = mean_squared_error(co_gt[:len(predicted_results)], predicted_results)
overall_rmse = np.sqrt(overall_mse)
overall_r2 = r2_score(co_gt[:len(predicted_results)], predicted_results)
overall_accuracy = compute_Dstat(co_gt[:len(predicted_results)], predicted_results)  # 对于总体也使用自定义的 Dstat 函数


print(f"Overall - MAE: {overall_mae:.4f}, MSE: {overall_mse:.4f}, RMSE: {overall_rmse:.4f}, R²: {overall_r2:.4f}, Accuracy (Dstat): {overall_accuracy:.4f}")


for i, coeff in enumerate(coeffs):
    X_train = np.arange(len(coeff)).reshape(-1, 1)
    y_train = coeff
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    predicted_inf = rf_model.predict(X_train)


    mae = mean_absolute_error(y_train, predicted_inf)
    mse = mean_squared_error(y_train, predicted_inf)
    r2 = r2_score(y_train, predicted_inf)
    rmse = np.sqrt(mse)
    accuracy = compute_Dstat(y_train, predicted_inf)  # 计算准确率，使用自定义的 Dstat 函数
    print(f"Coefficient {i + 1} - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}, Accuracy (Dstat): {accuracy:.4f}")
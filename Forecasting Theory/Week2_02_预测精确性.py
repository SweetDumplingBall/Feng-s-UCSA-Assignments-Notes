import numpy as np


def evaluation_metrics(true_values, predicted_values):
    """
    该函数用于计算多种评估指标，包括 MSE, RMSE, MAE, MAPE 和 R2。
    :param true_values: 真实值列表
    :param predicted_values: 预测值列表
    :return: 包含 MSE, RMSE, MAE, MAPE 和 R2 的字典
    """
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    n = len(true_values)

    # 均方误差 (MSE)
    mse = np.mean((true_values - predicted_values) ** 2)

    # 均方根误差 (RMSE)
    rmse = np.sqrt(mse)

    # 平均绝对误差 (MAE)
    mae = np.mean(np.abs(true_values - predicted_values))

    # 平均绝对百分比误差 (MAPE)
    # 为避免除以 0 的情况，添加一个极小值
    mape = np.mean(np.abs((true_values - predicted_values) / (true_values + 1e-10))) * 100

    # 决定系数 (R2)
    y_mean = np.mean(true_values)
    ss_total = np.sum((true_values - y_mean) ** 2)
    ss_residual = np.sum((true_values - predicted_values) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }


# 示例使用
true_list = [10, 12, 8, 15, 13]
predicted_list = [11, 13, 7, 16, 12]
metrics = evaluation_metrics(true_list, predicted_list)
print(metrics)
def directionality_calculation(true_values, predicted_values):
    """
    该函数用于计算方向性并添加方向准确度指标。
    :param true_values: 真实值列表
    :param predicted_values: 预测值列表
    :return: 方向性结果列表，以及方向准确度指标
    """
    direction_results = []
    correct_direction_count = 0  # 用于统计方向正确的次数
    total_comparisons = len(true_values) - 1  # 总比较次数，最后一个元素不参与比较
    for i in range(len(true_values)):
        if i == len(true_values) - 1:
            direction_results.append(None)  # 最后一个元素没有下一个元素进行比较，所以为 None
        else:
            true_diff = true_values[i + 1] - true_values[i]  # 修改为与下一个元素比较
            pred_diff = predicted_values[i + 1] - predicted_values[i]  # 修改为与下一个元素比较
            if true_diff * pred_diff > 0:
                direction_results.append(1)  # 同向变化
                correct_direction_count += 1
            elif true_diff * pred_diff < 0:
                direction_results.append(-1)  # 反向变化
            else:
                direction_results.append(0)  # 无变化
    if total_comparisons == 0:  # 处理总比较次数为 0 的情况，避免除以 0 的错误
        direction_accuracy = 0
    else:
        direction_accuracy = correct_direction_count / total_comparisons
    return direction_results, direction_accuracy


# 示例使用
true_list = [10, 12, 20, 15, 13]
predicted_list = [11, 13, 7, 16, 12]
direction_results, direction_accuracy = directionality_calculation(true_list, predicted_list)
print("方向性结果:", direction_results)
print("方向准确度:", direction_accuracy)
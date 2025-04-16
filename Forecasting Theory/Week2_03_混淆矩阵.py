def precision_recall_f1(y_true, y_pred):
    """
    该函数用于计算精确率、召回率和 F1 值
    :param y_true: 真实结果列表
    :param y_pred: 预测结果列表
    :return: 精确率、召回率和 F1 值
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            true_positives += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            false_positives += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            false_negatives += 1
    # 计算精确率
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)
    # 计算召回率
    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)
    # 计算 F1 值
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


# 示例使用
y_true = [1, 1, 0, 1, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1]
precision, recall, f1 = precision_recall_f1(y_true, y_pred)
print("精确率:", precision)
print("召回率:", recall)
print("F1 值:", f1)
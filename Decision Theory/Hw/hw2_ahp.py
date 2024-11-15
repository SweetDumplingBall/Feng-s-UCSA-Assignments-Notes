import numpy as np

# 定义随机一致性指标 RI 值
RI_dict = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49
}

def compute_ahp(A):
    """
    计算给定成对比较矩阵的 λ_max、CI、CR 和权重向量。

    参数：
        A (numpy.ndarray): 成对比较矩阵。

    返回：
        dict: 包含 λ_max、CI、CR 和权重向量的字典。
    """
    n = A.shape[0]  # 判断矩阵的阶数

    # 步骤1：计算每列的和
    col_sum = np.sum(A, axis=0)

    # 步骤2：标准化矩阵
    A_norm = A / col_sum

    # 步骤3：计算权重向量（每行的平均值）
    weights = np.mean(A_norm, axis=1)

    # 步骤4：计算 Aw
    Aw = np.dot(A, weights)

    # 步骤5：计算 λ_max
    lambda_max = np.sum(Aw / weights) / n

    # 步骤6：计算 CI
    CI = (lambda_max - n) / (n - 1)

    # 步骤7：计算 CR
    RI = RI_dict.get(n, 0)
    CR = CI / RI if RI != 0 else 0

    # 判断一致性
    consistency = "一致性可以接受" if CR < 0.1 else "一致性不可接受，请重新调整判断矩阵"

    return {
        'lambda_max': lambda_max,
        'CI': CI,
        'CR': CR,
        'weights': weights,
        'consistency': consistency
    }

# 定义所有的成对比较矩阵
matrices = {
    'Attributes': np.array([
        [1,   3,   5,   3,   5],
        [1/3, 1,   3,   1,   3],
        [1/5, 1/3, 1,   1/3, 3],
        [1/3, 1,   3,   1,   3],
        [1/5, 1/3, 1/3, 1/3, 1]
    ]),
    'y1': np.array([
        [1,   1,   5],
        [1,   1,   5],
        [1/5, 1/5, 1]
    ]),
    'y2': np.array([
        [1,   3,   5],
        [1/3, 1,   2],
        [1/5, 1/2, 1]
    ]),
    'y3': np.array([
        [1,     4,     7],
        [1/4,   1,     4],
        [1/7,   1/4,   1]
    ]),
    'y4': np.array([
        [1,     1/2,   1/3],
        [2,     1,     1],
        [3,     1,     1]
    ]),
    'y5': np.array([
        [1,     1/2,   1/3],
        [2,     1,     1],
        [3,     1,     1]
    ])  # y5 与 y4 相同
}

# 存储计算结果
results = {}

# 对每个矩阵进行计算
for key, A in matrices.items():
    result = compute_ahp(A)
    results[key] = result
    print(f"--- {key} 的计算结果 ---")
    print(f"最大特征值 λ_max：{result['lambda_max']:.4f}")
    print(f"一致性指标 CI：{result['CI']:.4f}")
    print(f"一致性比率 CR：{result['CR']:.4f}")
    print(f"权重向量：{np.round(result['weights'], 4)}")
    print(f"判断：{result['consistency']}\n")

# 提取属性权重
attribute_weights = results['Attributes']['weights']

# 提取各属性下备选方案的权重
alternative_weights = {
    'y1': results['y1']['weights'],
    'y2': results['y2']['weights'],
    'y3': results['y3']['weights'],
    'y4': results['y4']['weights'],
    'y5': results['y5']['weights']
}

# 构建综合评价矩阵
# 假设备选方案为 x1, x2, x3
# 行表示备选方案，列表示属性
num_alternatives = len(alternative_weights['y1'])
num_attributes = len(attribute_weights)
synthesis_matrix = np.zeros((num_alternatives, num_attributes))

attribute_keys = ['y1', 'y2', 'y3', 'y4', 'y5']

for j, attr in enumerate(attribute_keys):
    synthesis_matrix[:, j] = alternative_weights[attr]

# 计算综合得分
overall_scores = np.dot(synthesis_matrix, attribute_weights)

# 归一化综合得分
overall_scores_normalized = overall_scores / np.sum(overall_scores)

# 排序
ranking = np.argsort(-overall_scores_normalized)  # 降序排列
print("\n--- 方案排序 ---")
for idx in ranking:
    print(f"第 {idx+1} 名：备选方案 x{idx+1}，得分：{overall_scores_normalized[idx]:.4f}")

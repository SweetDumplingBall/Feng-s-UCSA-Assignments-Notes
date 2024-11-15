import numpy as np
import pandas as pd

# 原始数据
data = {
    'Price': [1018, 850, 892, 1128, 1094, 1190],
    'Time': [74, 80, 72, 63, 53, 50],
    'Electricity': [0.8, 0.75, 0.8, 0.8, 0.9, 0.9],
    'Water': [342, 330, 405, 354, 420, 405]
}

# 创建数据框
df = pd.DataFrame(data, index=['Machine1', 'Machine2', 'Machine3', 'Machine4', 'Machine5', 'Machine6'])
print("原始数据：")
print(df)

# 所有指标均为代价型，值越小越好
# 需要在标准化前将代价型指标转换为效益型指标，或在标准化公式中调整

# 第一步：标准化决策矩阵

# 1. 线性变换标准化
def linear_normalization(df):
    df_norm = df.copy()
    for column in df.columns:
        max_value = df[column].max()
        min_value = df[column].min()
        # 代价型指标，使用 (max - x) / (max - min)
        df_norm[column] = (max_value - df[column]) / (max_value - min_value)
    return df_norm

df_linear = linear_normalization(df)
print("\n线性变换标准化后的矩阵：")
print(df_linear)

# 2. 极小-极大归一化（与线性变换标准化相同）
df_minmax = df_linear.copy()
print("\n极小-极大归一化后的矩阵：")
print(df_minmax)

# 3. 向量归一化
def vector_normalization(df):
    df_norm = df.copy()
    for column in df.columns:
        norm = np.sqrt((df[column] ** 2).sum())
        df_norm[column] = df[column] / norm
    return df_norm

df_vector = vector_normalization(df)
print("\n向量归一化后的矩阵：")
print(df_vector)

# 第二步：使用熵权法计算权重，基于线性变换标准化后的矩阵
def entropy_weight_method(df):
    df_norm = df.copy()
    # 计算比重 p_ij
    for column in df_norm.columns:
        total = df_norm[column].sum()
        df_norm[column] = df_norm[column] / total

    # 计算熵值 e_j
    k = 1 / np.log(len(df_norm))
    e = {}
    for column in df_norm.columns:
        p = df_norm[column]
        e_j = -k * (p * np.log(p + 1e-12)).sum()  # 加上1e-12防止log(0)
        e[column] = e_j

    # 计算权重 w_j
    d = {key: 1 - value for key, value in e.items()}
    d_total = sum(d.values())
    w = {key: value / d_total for key, value in d.items()}
    return w

weights = entropy_weight_method(df_linear)
print("\n熵权法计算得到的权重：")
for key, value in weights.items():
    print(f"{key}: {value:.4f}")

# 第三步：应用加权和法（WSM）和加权乘积法（WPM）进行排序

# 1. 加权和法（WSM）
def weighted_sum_model(df, weights):
    scores = {}
    for index, row in df.iterrows():
        score = sum([row[col] * weights[col] for col in df.columns])
        scores[index] = score
    return scores

wsm_scores = weighted_sum_model(df_linear, weights)
wsm_rank = sorted(wsm_scores.items(), key=lambda x: x[1], reverse=True)
print("\n加权和法（WSM）得分及排序：")
for i, (machine, score) in enumerate(wsm_rank, 1):
    print(f"Rank {i}: {machine}, Score: {score:.4f}")

# 2. 加权乘积法（WPM）
def weighted_product_model(df, weights):
    scores = {}
    for index, row in df.iterrows():
        score = np.prod([row[col] ** weights[col] for col in df.columns])
        scores[index] = score
    return scores

wpm_scores = weighted_product_model(df_linear, weights)
wpm_rank = sorted(wpm_scores.items(), key=lambda x: x[1], reverse=True)
print("\n加权乘积法（WPM）得分及排序：")
for i, (machine, score) in enumerate(wpm_rank, 1):
    print(f"Rank {i}: {machine}, Score: {score:.6f}")

# 第四步：使用 TOPSIS 方法进行排序（基于向量归一化后的矩阵）
def topsis(df, weights):
    df_norm = df.copy()
    # 1. 乘以权重
    for column in df_norm.columns:
        df_norm[column] = df_norm[column] * weights[column]
    # 2. 确定正理想解和负理想解
    ideal_best = df_norm.min()  # 因为是代价型指标
    ideal_worst = df_norm.max()
    # 3. 计算与正理想解和负理想解的距离
    distances_best = np.sqrt(((df_norm - ideal_best) ** 2).sum(axis=1))
    distances_worst = np.sqrt(((df_norm - ideal_worst) ** 2).sum(axis=1))
    # 4. 计算相对贴近度
    scores = distances_worst / (distances_best + distances_worst)
    return scores

topsis_scores = topsis(df_vector, weights)
topsis_rank = topsis_scores.sort_values(ascending=False)
print("\nTOPSIS 方法得分及排序：")
for i, (machine, score) in enumerate(topsis_rank.items(), 1):
    print(f"Rank {i}: {machine}, Score: {score:.4f}")

# 第五步：使用 VIKOR 方法进行排序
def vikor(df, weights):
    f_star = df.min()  # 最优值（代价型指标）
    f_minus = df.max()  # 最劣值
    S = {}
    R = {}
    for index, row in df.iterrows():
        S_i = sum([weights[col] * (row[col] - f_star[col]) / (f_minus[col] - f_star[col]) for col in df.columns])
        R_i = max([weights[col] * (row[col] - f_star[col]) / (f_minus[col] - f_star[col]) for col in df.columns])
        S[index] = S_i
        R[index] = R_i
    S_star = min(S.values())
    S_minus = max(S.values())
    R_star = min(R.values())
    R_minus = max(R.values())
    Q = {}
    v = 0.5  # 权重系数，一般取0.5
    for index in df.index:
        Q_i = v * (S[index] - S_star) / (S_minus - S_star + 1e-12) + (1 - v) * (R[index] - R_star) / (R_minus - R_star + 1e-12)
        Q[index] = Q_i
    Q_sorted = sorted(Q.items(), key=lambda x: x[1])
    return Q_sorted

vikor_rank = vikor(df_linear, weights)
print("\nVIKOR 方法得分及排序：")
for i, (machine, score) in enumerate(vikor_rank, 1):
    print(f"Rank {i}: {machine}, Q: {score:.4f}")

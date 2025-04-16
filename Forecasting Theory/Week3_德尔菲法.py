import numpy as np
from scipy.stats import kendalltau


# 变异系数计算及专家意见集中程度
def coefficient_of_variation(ratings):
    """
    此函数用于计算变异系数及专家意见集中程度（中位数和上下四分位数）
    :param ratings: 评估者的评分结果，二维 numpy 数组，每一列是一个对象的评分
    :return: 每个对象的变异系数、中位数、下四分位数、上四分位数
    """
    means = np.mean(ratings, axis=0)  # 计算每个对象的平均值
    stds = np.std(ratings, axis=0)  # 计算每个对象的标准差
    cvs = stds / means  # 计算变异系数
    medians = np.median(ratings, axis=0)  # 计算每个对象的中位数
    q1s = np.percentile(ratings, 25, axis=0)  # 计算每个对象的下四分位数
    q3s = np.percentile(ratings, 75, axis=0)  # 计算每个对象的上四分位数
    return cvs, medians, q1s, q3s


# 假设我们有以下数据，这里是 5 个评估者对 3 个对象的评分结果
# 每一列代表一个对象的评分结果
# 例如，第一个对象的评分为 [85, 88, 82, 84, 86]
ratings = np.array([
    [85, 78, 92],
    [88, 80, 90],
    [82, 75, 88],
    [84, 82, 91],
    [86, 79, 89]
])


# 调用函数计算变异系数及专家意见集中程度
cvs, medians, q1s, q3s = coefficient_of_variation(ratings)
print("Coefficient of Variation for each object:", cvs)
print("Medians for each object:", medians)
print("First Quartiles for each object:", q1s)
print("Third Quartiles for each object:", q3s)


# 肯德尔和谐系数计算及专家意见集中程度
def kendall_concordance(rankings):
    """
    此函数用于计算肯德尔和谐系数及专家意见集中程度（中位数和上下四分位数）
    :param rankings: 评估者的排序结果，二维 numpy 数组，每一行是一个评估者的排序
    :return: 肯德尔和谐系数、每个对象的中位数、下四分位数、上四分位数
    """
    num_judges = rankings.shape[0]  # 评估者的数量
    num_items = rankings.shape[1]  # 评估对象的数量
    total_ranks = np.sum(rankings, axis=0)  # 对每个对象的排序总和
    mean_ranks = total_ranks / num_judges  # 平均排序
    S = np.sum((total_ranks - mean_ranks.mean() * num_judges) ** 2)  # 计算 S
    W = 12 * S / (num_judges ** 2 * (num_items ** 3 - num_items))  # 计算肯德尔和谐系数
    medians = np.median(rankings, axis=0)  # 计算每个对象的中位数
    q1s = np.percentile(rankings, 25, axis=0)  # 计算每个对象的下四分位数
    q3s = np.percentile(rankings, 75, axis=0)  # 计算每个对象的上四分位数
    return W, medians, q1s, q3s


# 假设我们有以下数据，这里是 3 个评估者对 5 个对象的排序结果
# 每一行代表一个评估者的排序结果
# 例如，第一个评估者将对象排序为 [2, 1, 3, 4, 5]，表示他认为第二个对象是最好的，第一个对象次之，以此类推
rankings = np.array([
    [3, 5, 2, 4, 1, 6],
    [4, 5, 3, 2, 1, 6],
    [2, 4, 3, 5, 1, 6],
    [2, 4, 3, 5, 1, 6],
    [2, 4, 3, 5, 1, 6],
    [3, 5, 2, 4, 1, 6]
])


# 调用函数计算肯德尔和谐系数及专家意见集中程度
W, medians, q1s, q3s = kendall_concordance(rankings)
print("Kendall's coefficient of concordance W:", W)
print("Medians for each object:", medians)
print("First Quartiles for each object:", q1s)
print("Third Quartiles for each object:", q3s)


# 使用 scipy 的 kendalltau 函数进行验证
# 这里将 rankings 转换为 rank 矩阵，将排序转换为排名
rank_matrix = []
for r in rankings:
    rank_matrix.append([(rankings.shape[1] - np.sum(r > r[i])) for i in range(rankings.shape[1])])
rank_matrix = np.array(rank_matrix)


# 计算肯德尔和谐系数
tau, p_value = kendalltau(rank_matrix[0], rank_matrix[1])
print("Kendall's tau:", tau)
print("p-value:", p_value)
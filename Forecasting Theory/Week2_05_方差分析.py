import pandas as pd
from scipy import stats


def main():
    """
    主函数，用于读取数据文件，计算差值，进行 t 检验并打印结果。
    """
    # 读取数据文件，假设文件是一个 CSV 文件，文件名为 data.csv
    data = pd.read_csv('data.csv')
    # 计算差值
    C_A = data['模型1预测值'] - data['真实值']
    C_B = data['模型2预测值'] - data['真实值']
    # 计算 u
    u = C_A - C_B
    # 进行单样本 t 检验
    t_value, p_value = stats.ttest_1samp(u, 0)
    print("u statistic:", t_value)
    print("μ pvalue:", p_value)


if __name__ == "__main__":
    main()
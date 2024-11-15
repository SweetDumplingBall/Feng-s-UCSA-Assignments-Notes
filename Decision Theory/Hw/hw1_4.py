# -*- coding: utf-8 -*-

# 概率和收益数据
P_theta = [0.25, 0.30, 0.45]  # [P(θ1), P(θ2), P(θ3)]
Profit = [15, 1, -6]          # 对应的收益

# 条件概率表 P(Hi|θj)
P_H_given_theta = [
    [0.65, 0.25, 0.10],  # P(H1|θ1), P(H1|θ2), P(H1|θ3)
    [0.25, 0.45, 0.15],  # P(H2|θ1), P(H2|θ2), P(H2|θ3)
    [0.10, 0.30, 0.75]   # P(H3|θ1), P(H3|θ2), P(H3|θ3)
]

# 1. 计算未进行市场调查时的期望收益 E0
E0 = sum([P_theta[i] * Profit[i] for i in range(3)])
print("未进行市场调查时的期望收益 E0：{:.4f} 万元".format(E0))

# 2. 计算各市场调查结果的边际概率 P(Hi)
P_H = []
for i in range(3):
    P_H_i = sum([P_H_given_theta[i][j] * P_theta[j] for j in range(3)])
    P_H.append(P_H_i)
    print("P(H{}): {:.4f}".format(i+1, P_H_i))

# 3. 计算后验概率 P(θj|Hi) 并计算每种 Hi 下的期望收益
P_theta_given_H = []
E_profit_H = []
for i in range(3):
    P_theta_given_H_i = []
    E_profit_H_i = 0
    print("\n市场调查结果 H{} 的后验概率和期望收益计算：".format(i+1))
    for j in range(3):
        P = P_H_given_theta[i][j] * P_theta[j] / P_H[i]
        P_theta_given_H_i.append(P)
        print("P(θ{}|H{}): {:.4f}".format(j+1, i+1, P))
        E_profit_H_i += P * Profit[j]
    P_theta_given_H.append(P_theta_given_H_i)
    E_profit_H.append(E_profit_H_i)
    print("E[收益|H{}]: {:.4f} 万元".format(i+1, E_profit_H_i))

# 4. 确定每种调查结果下的最优决策和收益
Optimal_Profit = []
Decision = []
for i in range(3):
    if E_profit_H[i] > 0:
        # 进行销售，期望收益为 E_profit_H_i
        Optimal_Profit.append(E_profit_H[i])
        Decision.append("进行销售")
    else:
        # 不进行销售，收益为0
        Optimal_Profit.append(0)
        Decision.append("不进行销售")
    print("\n市场调查结果 H{} 下的决策：{}，收益为 {:.4f} 万元".format(i+1, Decision[i], Optimal_Profit[i]))

# 5. 计算进行市场调查后的期望收益 E1
E1 = sum([P_H[i] * Optimal_Profit[i] for i in range(3)])
print("\n进行市场调查后的期望收益 E1：{:.4f} 万元".format(E1))

# 6. 计算样本信息的期望价值 EVSI
EVSI = E1 - E0
print("样本信息的期望价值 EVSI：{:.4f} 万元".format(EVSI))

# 7. 比较 EVSI 与 市场调查费用，决定是否进行市场调查
survey_cost = 0.6  # 市场调查费用（万元）
if EVSI > survey_cost:
    print("应进行市场调查，因为 EVSI ({:.4f}) > 市场调查费用 ({:.4f}) 万元".format(EVSI, survey_cost))
    should_survey = True
else:
    print("不应进行市场调查，因为 EVSI ({:.4f}) <= 市场调查费用 ({:.4f}) 万元".format(EVSI, survey_cost))
    should_survey = False

# 8. 计算抽样的期望净收益 ENGS
ENGS = EVSI - survey_cost
print("抽样的期望净收益 ENGS：{:.4f} 万元".format(ENGS))

# 9. 计算完全信息下的期望收益 E_PI
# 完全信息下的最优决策收益
Optimal_Profit_PI = []
for i in range(3):
    if Profit[i] > 0:
        Optimal_Profit_PI.append(Profit[i])
    else:
        Optimal_Profit_PI.append(0)

E_PI = sum([P_theta[i] * Optimal_Profit_PI[i] for i in range(3)])
print("完全信息下的期望收益 E_PI：{:.4f} 万元".format(E_PI))

# 10. 计算完全信息的期望价值 EVPI
EVPI = E_PI - E0
print("完全信息的期望价值 EVPI：{:.4f} 万元".format(EVPI))

# 11. 输出最终决策
print("\n最终决策：")
if should_survey:
    print("应进行市场调查。")
    print("决策方案：")
    for i in range(3):
        print("  如果市场调查结果是 H{}，则{}，收益为 {:.4f} 万元".format(i+1, Decision[i], Optimal_Profit[i]))
else:
    print("不应进行市场调查，直接按照未调查时的最优决策进行。")
    if E0 > 0:
        print("直接进行销售，期望收益为 {:.4f} 万元".format(E0))
    else:
        print("不进行销售，收益为 0 万元")

# -*- coding: utf-8 -*-

# 定义销售情况的概率
probabilities = {
    '高销量': 0.45,
    '销量一般': 0.30,
    '低销量': 0.25
}

# 定义每个车型在不同销售情况的收益（单位：万元）
models = {
    '车型1': {
        '高销量': 4200,
        '销量一般': 2300,
        '低销量': 950
    },
    '车型2': {
        '高销量': 3800,
        '销量一般': 1900,
        '低销量': 1100
    },
    '车型3': {
        '高销量': 4500,
        '销量一般': 2000,
        '低销量': 800
    }
}

# 1. 计算在无额外信息情况下的每个车型的期望收益
expected_profits = {}
for model_name, profit_dict in models.items():
    expected_profit = 0
    for sales_scenario, profit in profit_dict.items():
        expected_profit += probabilities[sales_scenario] * profit
    expected_profits[model_name] = expected_profit
    print(f"{model_name} 的期望收益：{expected_profit:.2f} 万元")

# 找出期望收益最高的车型
best_model = max(expected_profits, key=expected_profits.get)
best_expected_profit = expected_profits[best_model]
print(f"\n在无额外信息的情况下，应该选择 {best_model}，其期望收益为 {best_expected_profit:.2f} 万元")

# 2. 计算在完全信息条件下的期望收益
# 对于每种销售情况，选择收益最大的车型
best_profits_per_scenario = {}
for sales_scenario in probabilities.keys():
    max_profit = 0
    best_model_for_scenario = ''
    for model_name, profit_dict in models.items():
        profit = profit_dict[sales_scenario]
        if profit > max_profit:
            max_profit = profit
            best_model_for_scenario = model_name
    best_profits_per_scenario[sales_scenario] = {
        'model': best_model_for_scenario,
        'profit': max_profit
    }
    print(f"在 {sales_scenario} 下，选择 {best_model_for_scenario}，收益为 {max_profit} 万元")

# 计算完全信息下的期望收益
expected_profit_with_perfect_info = 0
for sales_scenario, info in best_profits_per_scenario.items():
    expected_profit_with_perfect_info += probabilities[sales_scenario] * info['profit']
print(f"\n完全信息下的期望收益：{expected_profit_with_perfect_info:.2f} 万元")

# 3. 计算完全信息的期望价值（EVPI）
EVPI = expected_profit_with_perfect_info - best_expected_profit
print(f"\n完全信息的期望价值（EVPI）：{EVPI:.2f} 万元")

# 4. 比较EVPI与获取信息的成本
cost_of_information = 150  # 获取完整市场情报的费用（万元）
print(f"获取信息的成本：{cost_of_information} 万元")

if EVPI > cost_of_information:
    print("\n由于 EVPI 大于 获取信息的成本，因此获取完整市场情报的工作是值得做的。")
    decision = "进行市场情报获取工作"
else:
    print("\n由于 EVPI 不大于 获取信息的成本，因此获取完整市场情报的工作不值得做。")
    decision = "不进行市场情报获取工作"

# 5. 计算扣除信息成本后的净收益
net_profit_with_info = expected_profit_with_perfect_info - cost_of_information
print(f"\n扣除信息成本后的净收益为：{net_profit_with_info:.2f} 万元")

# 6. 最终决策
print(f"\n最终决策：{decision}")
if decision == "进行市场情报获取工作":
    print("根据市场情报选择最佳车型：")
    for sales_scenario, info in best_profits_per_scenario.items():
        print(f"  - 如果市场情报显示 {sales_scenario}，选择 {info['model']}，收益为 {info['profit']} 万元")
else:
    print(f"直接选择 {best_model}，期望收益为 {best_expected_profit:.2f} 万元")

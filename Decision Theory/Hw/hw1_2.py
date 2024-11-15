# 定义各方案在不同市场需求下的收益（单位：万元）
actions = {
    '新建': [600, 50, -200],
    '扩建': [250, 200, -100],
    '租赁': [100, 100, 100]
}

# 悲观准则（Maximin）
pessimistic_returns = {}
for action, returns in actions.items():
    min_return = min(returns)
    pessimistic_returns[action] = min_return

# 选择最优方案
pessimistic_optimal_action = max(pessimistic_returns, key=pessimistic_returns.get)
pessimistic_optimal_value = pessimistic_returns[pessimistic_optimal_action]

# 乐观准则（Maximax）
optimistic_returns = {}
for action, returns in actions.items():
    max_return = max(returns)
    optimistic_returns[action] = max_return

# 选择最优方案
optimistic_optimal_action = max(optimistic_returns, key=optimistic_returns.get)
optimistic_optimal_value = optimistic_returns[optimistic_optimal_action]

# Hurwicz准则（α=0.4）
alpha = 0.4
hurwicz_returns = {}
for action, returns in actions.items():
    max_return = max(returns)
    min_return = min(returns)
    hurwicz_value = alpha * max_return + (1 - alpha) * min_return
    hurwicz_returns[action] = hurwicz_value

# 选择最优方案
hurwicz_optimal_action = max(hurwicz_returns, key=hurwicz_returns.get)
hurwicz_optimal_value = hurwicz_returns[hurwicz_optimal_action]

# 输出结果
print("悲观准则（Maximin）：")
for action, value in pessimistic_returns.items():
    print(f"方案 {action} 的最小收益为 {value} 万元")
print(f"最优方案是 {pessimistic_optimal_action}，期望收益为 {pessimistic_optimal_value} 万元\n")

print("乐观准则（Maximax）：")
for action, value in optimistic_returns.items():
    print(f"方案 {action} 的最大收益为 {value} 万元")
print(f"最优方案是 {optimistic_optimal_action}，期望收益为 {optimistic_optimal_value} 万元\n")

print(f"Hurwicz 准则（α={alpha}）：")
for action, value in hurwicz_returns.items():
    print(f"方案 {action} 的 Hurwicz 值为 {value} 万元")
print(f"最优方案是 {hurwicz_optimal_action}，期望收益为 {hurwicz_optimal_value} 万元")

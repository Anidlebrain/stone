import pandas as pd
import numpy as np
import random

# 加载策略数据并转换为数值型
df = pd.read_csv('top10k_round5.csv')

# 提取策略列和胜场数列
strategies = df.iloc[:, 2:].values.astype(int)  # p1 到 p10 列的策略分配
win_counts = df['wins'].values  # 胜场数

# 计算一个策略在每一场对局中的表现


def calculate_score(strategy_a, strategy_b):
    # 计算每个沙堆上的有效分配
    effective_a = strategy_a.copy()
    effective_b = strategy_b.copy()

    # 记录失败次数
    failures_a = np.zeros(len(strategy_a), dtype=int)
    failures_b = np.zeros(len(strategy_b), dtype=int)

    # 比较两个策略在每个沙堆上的分配
    for i in range(len(strategy_a)):
        if strategy_a[i] < strategy_b[i]:
            failures_a[i] += 1
        elif strategy_b[i] < strategy_a[i]:
            failures_b[i] += 1

        # 更新有效分配
        effective_a[i] = max(0, strategy_a[i] - failures_a[i])
        effective_b[i] = max(0, strategy_b[i] - failures_b[i])

    # 计算得分
    score_a = np.sum(effective_a)
    score_b = np.sum(effective_b)

    return score_a, score_b

# 计算一个策略和邻域策略的胜场数


def calculate_win_for_pair(strategy_a, strategy_b, strategies):
    score_a, score_b = calculate_score(strategy_a, strategy_b)
    if score_a > score_b:
        return 1, 0  # A 赢
    elif score_b > score_a:
        return 0, 1  # B 赢
    else:
        return 0, 0  # 平局

# 爬山算法：从一个随机策略开始搜索最优策略


def hill_climb(start_strategy, strategies, max_iterations=1000):
    current_strategy = start_strategy
    best_strategy = current_strategy
    best_score = -1

    # 只计算当前策略和邻域策略的胜场数
    for iteration in range(max_iterations):
        # 随机选择邻域策略
        neighbor_strategy = strategies[random.randint(0, len(strategies) - 1)]

        # 计算当前策略和邻域策略的胜场数
        wins_current, wins_neighbor = 0, 0
        for i in range(len(strategies)):
            if np.array_equal(strategies[i], current_strategy):
                continue
            wins_a, wins_b = calculate_win_for_pair(
                current_strategy, strategies[i], strategies)
            wins_current += wins_a
            wins_neighbor += wins_b

        # 更新最优策略
        if wins_current > best_score:
            best_strategy = current_strategy
            best_score = wins_current

        # 打印日志
        if iteration % 10 == 0:
            print(
                f"Iteration {iteration}: Current Best Score = {best_score}, Best Strategy = {best_strategy}")

    return best_strategy, best_score


# 随机选择一个策略作为起点
start_strategy_index = random.randint(0, len(strategies) - 1)
start_strategy = strategies[start_strategy_index]

# 调用爬山算法进行搜索
best_strategy, best_score = hill_climb(start_strategy, strategies)

# 输出最优策略的排名和胜场数
best_strategy_index = np.where(
    np.all(strategies == best_strategy, axis=1))[0][0]
best_rank = df.iloc[best_strategy_index]['rank']
print(f"最优策略：{best_strategy}")
print(f"排名：{best_rank}")
print(f"胜场数：{best_score}")

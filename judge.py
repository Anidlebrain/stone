

import re
import numpy as np

try:
    from numba import njit, int16
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

def calculate_match_result_round8(a, b) -> int:
    """
    计算比赛结果的核心逻辑，不包含装饰器。
    返回 1 如果 A 胜利，-1 如果 B 胜利，0 如果平局。
    a, b: int16[10] - 代表每个玩家的 10 个沙堆值
    """
    win_count_a = 0
    win_count_b = 0
    value_a = 0
    value_b = 0

    for i in range(10):
        pile_value = i + 1  # 第1个沙堆价值1，第10个沙堆价值10

        if a[i] > b[i]:
            win_count_a += 1
            value_a += pile_value
        elif a[i] < b[i]:
            win_count_b += 1
            value_b += pile_value
        # 相等则双方都不得分，也不计入赢堆数量

    # 第一优先级：赢下的沙堆数量
    if win_count_a > win_count_b:
        return 1
    elif win_count_b > win_count_a:
        return -1

    # 第二优先级：赢下沙堆的总价值
    if value_a > value_b:
        return 1
    elif value_b > value_a:
        return -1

    # 仍相同则平局
    return 0

def calculate_match_result_round7(a, b) -> int:
    """
    计算比赛结果的核心逻辑，不包含装饰器。
    返回 1 如果 A 胜利，-1 如果 B 胜利，0 如果平局。
    a, b: int16[10] - 代表每个玩家的 10 个沙堆值
    """

    # 5 组，使用 0-based 下标：
    # (1,10) -> (0,9)
    # (2,9)  -> (1,8)
    # (3,8)  -> (2,7)
    # (4,7)  -> (3,6)
    # (5,6)  -> (4,5)
    pairs = [(0, 9), (1, 8), (2, 7), (3, 6), (4, 5)]

    score_a = 0
    score_b = 0

    for i, j in pairs:
        # 先判断这两个沙堆各自是谁赢
        a_win_i = a[i] > b[i]
        b_win_i = b[i] > a[i]

        a_win_j = a[j] > b[j]
        b_win_j = b[j] > a[j]

        a_wins_in_group = int(a_win_i) + int(a_win_j)
        b_wins_in_group = int(b_win_i) + int(b_win_j)

        # 组内最大的沙堆价值（j 对应的编号更大）
        max_value = j + 1

        # A 在该组的得分
        if a_wins_in_group == 1:
            score_a += max_value
        elif a_wins_in_group == 2:
            score_a += 11

        # B 在该组的得分
        if b_wins_in_group == 1:
            score_b += max_value
        elif b_wins_in_group == 2:
            score_b += 11

    if score_a > score_b:
        return 1
    elif score_b > score_a:
        return -1
    else:
        return 0

strategies5 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
    np.array([9, 11, 13, 12, 15, 0, 0, 0, 20, 20]),
]


def calculate_match_result_round5(a, b) -> int:
    """
    Return:
      1 if a beats b
      0 if tie
     -1 if a loses to b
    """
    loss_a = 0
    loss_b = 0
    score_a = 0
    score_b = 0

    for i in range(10):
        ea = a[i] - loss_a
        eb = b[i] - loss_b
        if ea < 0:
            ea = 0
        if eb < 0:
            eb = 0

        if ea > eb:
            score_a += i + 1
            loss_b += 1
        elif ea < eb:
            score_b += i + 1
            loss_a += 1
        # equal -> nobody scores, no new loss

    if score_a > score_b:
        return 1
    if score_a < score_b:
        return -1
    return 0


def calculate_match_result_round6(a, b) -> int:
    """
    计算比赛结果的核心逻辑，不包含装饰器。
    返回 1 如果 A 胜利，-1 如果 B 胜利，0 如果平局。
    a, b: int16[10] - 代表每个玩家的 10 个沙堆值
    """
    streak_a = 0  # 跟踪 A 连胜的次数
    streak_b = 0  # 跟踪 B 连胜的次数
    score_a = 0  # A 玩家累积的总分
    score_b = 0  # B 玩家累积的总分

    # 检查沙堆对比
    for i in range(10):
        if a[i] > b[i]:
            score_a += (i + 1)  # 沙堆的值是索引 + 1
            streak_a += 1
            streak_b = 0  # 重置 B 的连胜
        elif a[i] < b[i]:
            score_b += (i + 1)
            streak_b += 1
            streak_a = 0  # 重置 A 的连胜
        else:
            # 如果沙堆值相等，则两者都不算胜利
            streak_a = 0
            streak_b = 0

        # 如果任一玩家连续胜利 3 次，后续的所有沙堆归该玩家
        if streak_a == 3:
            score_a += sum(range(i+2, 11))
            break  # 无需继续检查后续沙堆
        if streak_b == 3:
            score_b += sum(range(i+2, 11))
            break  # 无需继续检查后续沙堆

    # 根据沙堆的总分决定胜者
    if score_a > score_b:
        return 1  # A 胜利
    elif score_b > score_a:
        return -1  # B 胜利
    else:
        return 0  # 平局

strategies8 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
]

strategies7 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
]

strategies6 = [
    np.array([33, 33, 33, 1, 0, 0, 0, 0, 0, 0]),
    np.array([1, 0, 33, 0, 0, 33, 0, 0, 33, 0]),
    np.array([0, 0, 34, 33, 33, 0, 0, 0, 0, 0]),
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
    np.array([1, 1, 91, 1, 1, 1, 1, 1, 1, 1]),
    np.array([25, 25, 50, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 50, 25, 25, 0, 0, 0, 0, 0]),
    np.array([20, 20, 60, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 60, 20, 20, 0, 0, 0, 0, 0]),
    np.array([30, 30, 40, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 40, 30, 30, 0, 0, 0, 0, 0]),
    np.array([1, 1, 51, 1, 1, 20, 1, 1, 20, 1]),
    np.array([1, 1, 42, 1, 1, 24, 1, 1, 17, 17]),
    np.array([20, 20, 40, 0, 0, 0, 0, 0, 0, 0]),
    np.array([1, 1, 40, 19, 19, 0, 0, 0, 0, 0]),
    np.array([12, 12, 76, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 76, 12, 12, 0, 0, 0, 0, 0]),
    np.array([0, 0, 76, 12, 12, 0, 0, 0, 0, 0]),
]

def calculate_match_result_round1(a, b) -> int:
    """
    计算比赛结果的核心逻辑，不包含装饰器。
    返回 1 如果 A 胜利，-1 如果 B 胜利，0 如果平局。
    a, b: int16[10] - 代表每个玩家的 10 个沙堆值
    """
    return calculate_match_result_round9(a, b)

def calculate_match_result_round9(a, b) -> int:
    """
    计算比赛结果的核心逻辑，不包含装饰器。
    返回 1 如果 A 胜利，-1 如果 B 胜利，0 如果平局。
    a, b: int16[10] - 代表每个玩家的 10 个沙堆值
    """
    value_a = 0
    value_b = 0

    for i in range(10):
        pile_value = i + 1  # 第1个沙堆价值1，第10个沙堆价值10

        if a[i] > b[i]:
            value_a += pile_value
        elif a[i] < b[i]:
            value_b += pile_value

    # 第二优先级：赢下沙堆的总价值
    if value_a > value_b:
        return 1
    elif value_b > value_a:
        return -1

    # 仍相同则平局
    return 0

strategies9 = [
    # np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
    # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 210]),
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 91]),
    np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 82]),
]

def calculate_match_result_round10(a, b) -> int:
    """
    计算比赛结果的核心逻辑，不包含装饰器。
    返回 1 如果 A 胜利，-1 如果 B 胜利，0 如果平局。
    a, b: int16[10] - 代表每个玩家的 10 个沙堆值
    """
    value_a = 0
    value_b = 0

    for i in range(10):
        pile_value = i + 1  # 第1个沙堆价值1，第10个沙堆价值10

        if a[i] > b[i]:
            value_a += pile_value
        elif a[i] < b[i]:
            value_b += pile_value

    # 第二优先级：赢下沙堆的总价值
    if value_a > value_b:
        return 1
    elif value_b > value_a:
        return -1

    # 仍相同则平局
    return 0


strategies9 = [
    # np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
    # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 210]),
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 91]),
    np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 82]),
]


strategies10 = [
    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
    # # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 210]),
    # np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 91]),
    # np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 82]),
]

round_no = 10

match_func = globals()[f"calculate_match_result_round{round_no}"]
name = f"results_round{round_no}"
strategies = globals()[f"strategies{round_no}"]

if NUMBA_OK:
    calculate_match_result = njit(cache=True)(match_func)
else:
    calculate_match_result = match_func
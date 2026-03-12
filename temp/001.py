import random
import math

N_PILES = 10
VALUES = list(range(1, N_PILES + 1))  # 第k堆价值k
BUDGET = 100

def clamp_int(x, lo, hi):
    return max(lo, min(hi, x))

def normalize_to_budget(arr, budget=BUDGET):
    """把非负整数数组调整到sum=budget（尽量少改动）"""
    arr = [max(0, int(round(a))) for a in arr]
    s = sum(arr)
    if s == 0:
        arr[-1] = budget
        return arr
    # 先按比例缩放
    scaled = [a * budget / s for a in arr]
    arr = [int(math.floor(x)) for x in scaled]
    # 补差额
    diff = budget - sum(arr)
    # 按小数部分从大到小补
    frac = [(scaled[i] - arr[i], i) for i in range(len(arr))]
    frac.sort(reverse=True)
    for t in range(abs(diff)):
        i = frac[t % len(arr)][1]
        arr[i] += 1 if diff > 0 else -1
        arr[i] = max(0, arr[i])
    # 若减多了导致负数（极少），再修
    while sum(arr) != budget:
        d = budget - sum(arr)
        if d > 0:
            arr[random.randrange(len(arr))] += 1
        else:
            i = random.randrange(len(arr))
            if arr[i] > 0:
                arr[i] -= 1
    return arr

def score_value(a, b):
    """a vs b 的总价值（赢某堆得k分，平局不得分，输不得分）"""
    s = 0
    for k in range(N_PILES):
        if a[k] > b[k]:
            s += (k + 1)
        elif a[k] == b[k]:
            s += 0
    return s

def win_loss(a, b):
    sa = score_value(a, b)
    sb = score_value(b, a)
    if sa > sb:
        return 1
    elif sa < sb:
        return -1
    else:
        return 0

# --------- 对手生成（按你的四类假设）---------

def gen_concentrated(top_indices, peak_strength=0.85, noise=0.15):
    """
    集中策略：把大部分预算打到top_indices里，其它位置少量随机
    peak_strength: 集中部分占预算比例
    noise: 其它堆随机占比
    """
    arr = [0.0] * N_PILES
    peak_budget = BUDGET * peak_strength
    rest_budget = BUDGET - peak_budget

    # 集中部分：在top_indices里随机分
    w = [random.random() for _ in top_indices]
    sw = sum(w)
    for wi, idx in zip(w, top_indices):
        arr[idx] += peak_budget * (wi / sw)

    # 其它部分：全局随机撒一点
    w2 = [random.random() for _ in range(N_PILES)]
    sw2 = sum(w2)
    for i in range(N_PILES):
        arr[i] += rest_budget * (w2[i] / sw2)

    # 额外噪声：轻微扰动避免全都一个样
    for i in range(N_PILES):
        arr[i] *= (1.0 + random.uniform(-noise, noise))

    return normalize_to_budget(arr)

def gen_uniform(jitter=0.25):
    """均匀分配：接近10上下波动，但不全等"""
    base = BUDGET / N_PILES  # 10
    arr = [base * (1 + random.uniform(-jitter, jitter)) for _ in range(N_PILES)]
    return normalize_to_budget(arr)

def gen_random():
    """完全随机：Dirichlet-like"""
    arr = [random.random() for _ in range(N_PILES)]
    return normalize_to_budget(arr)

def build_opponents(n_high_8_10=7, n_uniform=10, n_random=5, n_mid_6_8=8,
                    peak_strength_high=0.88, peak_strength_mid=0.85):
    opps = []
    # 8/9/10 -> index 7,8,9
    for _ in range(n_high_8_10):
        opps.append(gen_concentrated([7,8,9], peak_strength=peak_strength_high))
    for _ in range(n_uniform):
        opps.append(gen_uniform())
    for _ in range(n_random):
        opps.append(gen_random())
    for _ in range(n_mid_6_8):
        opps.append(gen_concentrated([5,6,7,8,9], peak_strength=peak_strength_mid))
    return opps

# --------- 评估与搜索 ---------

def beats_count(me, opponents):
    """战胜人数（平局不算）"""
    wins = 0
    ties = 0
    for op in opponents:
        r = win_loss(me, op)
        if r == 1:
            wins += 1
        elif r == 0:
            ties += 1
    return wins, ties

def random_candidate(bias_high=True):
    """
    随机生成候选：可以偏向高价值堆（更适合这个游戏）
    """
    if bias_high:
        # 给高堆更大权重的随机
        weights = [(i+1)**1.3 * random.random() for i in range(N_PILES)]
    else:
        weights = [random.random() for _ in range(N_PILES)]
    return normalize_to_budget(weights)

def mutate(me, step=6):
    """
    局部变异：随机选两个堆，挪动一些币
    """
    a = me[:]
    i, j = random.sample(range(N_PILES), 2)
    delta = random.randint(1, step)
    if a[i] >= delta:
        a[i] -= delta
        a[j] += delta
    return a

def search_best(opponents, iters=20000, pool=2000, step=6, restarts=10, seed=None):
    if seed is not None:
        random.seed(seed)

    best = None
    best_wins = -1
    best_ties = -1

    # 1) 先随机池筛选
    candidates = []
    for _ in range(pool):
        c = random_candidate(bias_high=True)
        w, t = beats_count(c, opponents)
        candidates.append((w, t, c))
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    # 取前restarts个做爬山起点
    starts = [candidates[i][2] for i in range(min(restarts, len(candidates)))]

    # 2) 多起点局部搜索
    for start in starts:
        cur = start
        cur_w, cur_t = beats_count(cur, opponents)

        for _ in range(iters):
            nxt = mutate(cur, step=step)
            w, t = beats_count(nxt, opponents)

            # 以“战胜人数”为第一目标，“平局数”为第二目标（少输多平也有用）
            if (w, t) > (cur_w, cur_t):
                cur, cur_w, cur_t = nxt, w, t

            if (cur_w, cur_t) > (best_wins, best_ties):
                best, best_wins, best_ties = cur, cur_w, cur_t

    return best, best_wins, best_ties

if __name__ == "__main__":
    opponents = build_opponents(
        n_high_8_10=100,
        n_uniform=20,
        n_random=20,
        n_mid_6_8=100,
        peak_strength_high=0.85,
        peak_strength_mid=0.80
    )

    # 你给的起步解也可以测一下
    base = [1, 4, 4, 5, 7, 11, 1, 18, 35, 14]
    print("Base:", base, "beats,ties =", beats_count(base, opponents))

    best, wins, ties = search_best(opponents, iters=20000, pool=3000, step=8, restarts=12, seed=42)
    print("Best:", best, "sum=", sum(best), "beats=", wins, "ties=", ties)
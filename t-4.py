


import numpy as np
import random
import time
import judge

TOTAL = 100
K = 10

BOOST = 1.5
CLOSE_LOSS = 5.0

# 你的策略
# MY = np.array([0, 1, 17, 18, 19, 0, 21, 0, 24, 0], dtype=float)
MY1 = np.array([0, 16, 0,0,19,21,22,22,0,0], dtype=float)
you1 = np.array([16,16,14,2,2,2,2,14,16,16])

print(judge.calculate_match_result_round8(MY1, you1))

# 0 16 0 0 19 21 22 22 0 0
# 0 1 17 18 19 0 21 0 24 0

# ==============================
# 第四回合规则
# ==============================

def play(a, b):
    return judge.calculate_match_result(a, b)

# ==============================
# 随机策略生成
# ==============================


def random_strategy():

    p = np.random.dirichlet(np.ones(10))
    x = np.random.multinomial(100, p)

    return x


# ==============================
# 生成不重复
# ==============================

def encode(v):

    base = 101
    s = 0

    for i in range(10):
        s += int(v[i]) * (base**i)

    return s


# ==============================
# 主程序
# ==============================

def main():

    N = 1_000_000

    seen = set()

    wins = 0
    ties = 0
    losses = 0

    start = time.time()

    i = 0
    trials = 0

    while i < N:

        opp = random_strategy()

        code = encode(opp)

        trials += 1

        if code in seen:
            continue

        seen.add(code)

        r = play(MY, opp)

        if r == 1:
            wins += 1
        elif r == 0:
            ties += 1
        else:
            losses += 1

        i += 1

        if i % 100000 == 0:

            dt = time.time() - start

            print(
                f"{i}/{N} "
                f"wins={wins} "
                f"ties={ties} "
                f"loss={losses} "
                f"win_rate={wins/i:.4f} "
                f"time={dt:.1f}s"
            )

    print("\n===== FINAL =====")

    print("wins   =", wins)
    print("ties   =", ties)
    print("losses =", losses)
    print("win_rate =", wins/N)


# if __name__ == "__main__":
#     main()

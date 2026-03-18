import time
import numpy as np
import judge15 as judge

# 14维格式: [buyA, buyB, buyC, insurance, p1..p10]
MY = np.array([1, 0, 1, 6, 1, 1, 2, 3, 5, 7, 9, 12, 20, 20], dtype=np.int16)


def play(a, b):
    return judge.calculate_match_result(a, b)


def random_strategy(rng):
    return judge.sample_random_strategy(rng, judge.min_values)


def encode(v):
    base = 101
    s = 0
    for i in range(judge.STRATEGY_LEN):
        s += int(v[i]) * (base ** i)
    return s


def main():
    rng = np.random.default_rng(20260316)
    n = 200_000
    seen = set()

    wins = 0
    ties = 0
    losses = 0
    start = time.time()
    i = 0

    print('Round:', judge.round_no)
    print('Skill costs:', judge.SKILL_COST_A, judge.SKILL_COST_B, judge.SKILL_COST_C)
    print('MY =', MY.tolist())
    print('MY format =', judge.format_strategy(MY))
    print('MY valid =', judge.is_valid_strategy(MY))

    while i < n:
        opp = random_strategy(rng)
        code = encode(opp)
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

        if i % 20000 == 0:
            dt = time.time() - start
            print(f'{i}/{n} wins={wins} ties={ties} loss={losses} win_rate={wins / i:.4f} time={dt:.1f}s')

    print('\n===== FINAL =====')
    print('wins   =', wins)
    print('ties   =', ties)
    print('losses =', losses)
    print('win_rate =', wins / n)


if __name__ == '__main__':
    a = np.array([1, 1, 0, 17, 4, 5, 6, 8, 12, 15, 21, 0, 0, 0], dtype=np.int16)
    b = np.array([1,1,0,4,0,0,0,6,8,10,12,13,16,19], dtype=np.int16)
    print(judge.strategy_skill_cost(a))
    print(judge.strategy_skill_cost(b))
    print(judge.strategy_total_used(a))
    print(judge.strategy_total_used(b))
    print(judge.calculate_match_result(a, b))

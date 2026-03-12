# -*- coding: utf-8 -*-
"""
Search best strategy against top10k.npz

Input:
    top10k.npz
contains:
    strategies: shape (N, 10), int16
    wins: shape (N,), int32   # not used here

Goal:
    Search one strategy x (10 ints, sum=100, each >=0)
    maximizing wins against all strategies in top10k.

Method:
    random init + hill climbing + multiple restarts
"""

import time
import math
import argparse
import numpy as np
import judge

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False


TOTAL = 100
K = 10
BOOST = 1.5
CLOSE_LOSS = 5.0


# =========================================================
# Core match logic
# =========================================================

if NUMBA_OK:
    @njit(cache=True)
    def match_result(a, b):
        """
        Return 1 if a wins, -1 if b wins, 0 if tie.
        a, b: int16[10] - representing the 10 sandpile values for each player.
        """

        return judge.calculate_match_result(a, b)

    @njit(cache=True)
    def evaluate_against_pool(x, pool):
        """
        x: shape (10,)
        pool: shape (N,10)
        return:
            wins, ties, losses
        """
        wins = 0
        ties = 0
        losses = 0
        for i in range(pool.shape[0]):
            r = match_result(x, pool[i])
            if r == 1:
                wins += 1
            elif r == 0:
                ties += 1
            else:
                losses += 1
        return wins, ties, losses

else:
    def match_result(a, b):
        """
        Return 1 if a wins, -1 if b wins, 0 if tie.
        a, b: int16[10] - representing the 10 sandpile values for each player.
        """

        return judge.calculate_match_result(a, b)

    def evaluate_against_pool(x, pool):
        wins = 0
        ties = 0
        losses = 0
        for i in range(pool.shape[0]):
            r = match_result(x, pool[i])
            if r == 1:
                wins += 1
            elif r == 0:
                ties += 1
            else:
                losses += 1
        return wins, ties, losses


# =========================================================
# Strategy generation / mutation
# =========================================================

def check_min_value(min_value):
    if min_value < 0:
        raise ValueError("min_value must be >= 0")
    if min_value * K > TOTAL:
        raise ValueError(
            f"invalid min_value={min_value}, because {min_value} * {K} > {TOTAL}"
        )

def random_strategy(rng, min_value=0):
    check_min_value(min_value)

    remain = TOTAL - K * min_value
    p = rng.dirichlet(np.ones(K))
    extra = rng.multinomial(remain, p).astype(np.int16)
    x = extra + min_value
    return x.astype(np.int16)


def repair_sum_100(x, min_value=0):
    """
    Ensure x is int array, each >= min_value, sum=TOTAL
    """
    check_min_value(min_value)

    x = np.asarray(x, dtype=np.int16).copy()

    # 先拉到下界
    x[x < min_value] = min_value

    s = int(x.sum())

    if s == TOTAL:
        return x

    if s < TOTAL:
        x[K - 1] += (TOTAL - s)
        return x

    # s > TOTAL，需要从“高于 min_value 的部分”扣掉
    extra = s - TOTAL
    for i in range(K - 1, -1, -1):
        can_take = int(x[i] - min_value)
        if can_take <= 0:
            continue
        take = min(can_take, extra)
        x[i] -= take
        extra -= take
        if extra == 0:
            break

    if extra != 0:
        raise ValueError("repair failed: cannot satisfy sum=TOTAL with given min_value")

    return x


def mutate_strategy(x, rng, max_step=8, moves=3, min_value=0):
    """
    Randomly move coins across piles, preserving sum=100
    and each pile >= min_value
    """
    check_min_value(min_value)

    y = x.copy()

    m = rng.integers(1, moves + 1)
    for _ in range(m):
        i = int(rng.integers(0, K))
        j = int(rng.integers(0, K))
        while j == i:
            j = int(rng.integers(0, K))

        available = int(y[i] - min_value)
        if available <= 0:
            continue

        step = int(rng.integers(1, max_step + 1))
        step = min(step, available)

        y[i] -= step
        y[j] += step

    return y


def jitter_strategy(base, max_delta=3, rng=None, min_value=0):
    """
    基于单条策略生成一条“类似策略”
    - 每个位置随机波动范围: [-max_delta, max_delta]
    - 再修复为非负且总和=TOTAL
    """
    if rng is None:
        rng = np.random.default_rng()

    base = np.asarray(base, dtype=np.int16)

    # 每一维做有限随机波动
    noise = rng.integers(-max_delta, max_delta + 1, size=base.shape)
    new_x = base + noise

    # 修复约束
    new_x = repair_sum_100(new_x, min_value=min_value)
    return new_x


def structured_seeds(max_delta=3, keep_original=True, rng=None, min_value=0):
    """
    返回种子策略：
    1. 保留原始策略（可选）
    2. 对每条原始策略随机生成 n 条类似策略

    参数
    ----
    n : int
        每条原始策略额外生成多少条
    max_delta : int
        每个位置允许的最大波动幅度
    keep_original : bool
        是否保留 judge.strategies 中原始策略
    rng : np.random.Generator or None
        随机数生成器，便于复现
    """
    n = 100
    if rng is None:
        rng = np.random.default_rng()

    seeds = []

    for s in judge.strategies:
        base = repair_sum_100(s, min_value=min_value)

        if keep_original:
            seeds.append(base.copy())

        for _ in range(n):
            new_s = jitter_strategy(base, max_delta=max_delta, rng=rng, min_value=min_value)
            seeds.append(new_s)

    return seeds

# =========================================================
# Search
# =========================================================


def score_tuple(w, t, l):
    """
    Primary: wins
    Secondary: ties
    """
    return (w, t, -l)


def local_search(pool, x0, rng, iters=3000, neighborhood=40, max_step=8, verbose=False, min_value=0):
    """
    Hill climbing:
    - Around current best x
    - Sample 'neighborhood' mutations
    - Take the best improving move
    - If none improves, do a random shake
    """
    x = x0.copy()
    w, t, l = evaluate_against_pool(x, pool)
    best_local = x.copy()
    best_w, best_t, best_l = w, t, l

    stall = 0

    for it in range(1, iters + 1):
        improved = False
        cand_best = None
        cand_score = None

        for _ in range(neighborhood):
            y = mutate_strategy(x, rng, max_step=max_step, moves=3, min_value=min_value)
            wy, ty, ly = evaluate_against_pool(y, pool)
            sc = score_tuple(wy, ty, ly)

            if cand_score is None or sc > cand_score:
                cand_score = sc
                cand_best = (y, wy, ty, ly)

        if cand_score > score_tuple(w, t, l):
            x, w, t, l = cand_best
            improved = True
            stall = 0

            if score_tuple(w, t, l) > score_tuple(best_w, best_t, best_l):
                best_local = x.copy()
                best_w, best_t, best_l = w, t, l
        else:
            stall += 1

        # no improvement for a while -> shake current point
        if not improved:
            x = mutate_strategy(
                best_local, rng, max_step=max_step * 2, moves=6, min_value=min_value)
            w, t, l = evaluate_against_pool(x, pool)

        if verbose and (it % 200 == 0):
            print(
                f"    iter={it:5d} | local_best wins={best_w} ties={best_t} losses={best_l} | stall={stall}")

    return best_local, best_w, best_t, best_l


def global_search(
    pool,
    restarts=30,
    iters=3000,
    neighborhood=40,
    max_step=8,
    seed=1,
    verbose=True,
    min_value=0,
):
    rng = np.random.default_rng(seed)

    global_best = None
    global_w = -1
    global_t = -1
    global_l = 10**9

    seeds = structured_seeds(min_value=min_value)

    t0 = time.time()

    for r in range(1, restarts + 1):
        # 初始化策略来源
        p = rng.random()
        print(p)

        if p < 0.4:
            # 来自 top10k
            idx = rng.integers(0, pool.shape[0])
            x0 = pool[idx].copy()
            x0 = repair_sum_100(x0, min_value=min_value)
            x0 = mutate_strategy(x0, rng, max_step=5, moves=2, min_value=min_value)

        elif p < 0.8:
            # 随机
            x0 = random_strategy(rng, min_value=min_value)

        else:
            # 手工结构
            seeds = structured_seeds(min_value=min_value)
            x0 = seeds[rng.integers(0, len(seeds))].copy()

        x, w, t, l = local_search(
            pool=pool,
            x0=x0,
            rng=rng,
            iters=iters,
            neighborhood=neighborhood,
            max_step=max_step,
            verbose=False,
            min_value=min_value,
        )

        if score_tuple(w, t, l) > score_tuple(global_w, global_t, global_l):
            global_best = x.copy()
            global_w, global_t, global_l = w, t, l

        if verbose:
            dt = time.time() - t0
            print(
                f"[restart {r:3d}/{restarts}] "
                f"best_this={w}/{t}/{l} | "
                f"global_best={global_w}/{global_t}/{global_l} | "
                f"x={global_best.tolist()} | {dt:.1f}s"
            )

    return global_best, global_w, global_t, global_l

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=judge.name + "/top10k.npz")
    parser.add_argument("--restarts", type=int, default=30,
                        help="number of global restarts")
    parser.add_argument("--iters", type=int, default=5000,
                        help="local search iterations per restart")
    parser.add_argument("--neighborhood", type=int, default=40,
                        help="mutations tried per iteration")
    parser.add_argument("--max_step", type=int, default=8,
                        help="max coins moved per mutation")
    parser.add_argument("--seed", type=int, default=202603082229)
    parser.add_argument("--save", type=str,
                        default=judge.name + "/best_vs_top10k.txt")
    parser.add_argument("--min_value", type=int, default=2,
                    help="minimum coins for each pile")
    args = parser.parse_args()
    check_min_value(args.min_value)
    print(f"NUMBA_OK={NUMBA_OK}")
    print(f"Loading {args.input} ...")
    data = np.load(args.input)
    pool = data["strategies"].astype(np.int16)
    # print(pool)
    # seeds = structured_seeds()
    # seeds = np.array(seeds, dtype=np.int16)
    # print(seeds)
    # pool = np.vstack([pool, seeds])
    # print(pool)
    print(f"Pool size = {pool.shape[0]:,}")

    best_x, best_w, best_t, best_l = global_search(
        pool=pool,
        restarts=args.restarts,
        iters=args.iters,
        neighborhood=args.neighborhood,
        max_step=args.max_step,
        seed=args.seed,
        verbose=True,
        min_value=args.min_value
    )

    print("\n=== FINAL BEST ===")
    print("strategy =", best_x.tolist())
    print(f"wins={best_w}, ties={best_t}, losses={best_l}")
    print(f"win_rate={best_w / pool.shape[0]:.4f}")

    with open(args.save, "a", encoding="utf8") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} min={args.min_value} wins={best_w} ties={best_t} losses={best_l} win_rate={best_w / pool.shape[0]:.6f}\n")
        f.write("strategy: " + " ".join(map(str, best_x.tolist())) + "\n")

    print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()

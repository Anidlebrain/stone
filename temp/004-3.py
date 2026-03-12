# -*- coding: utf-8 -*-
"""
Search best strategy against top10k.npz
Rule: Round 4
- If you lose pile n by <= 5 coins, then on pile n+1 your coins are multiplied by 1.5
- Trigger can happen multiple times across piles

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
        return:
            1  if a wins
            0  if tie
           -1  if b wins
        """
        score_a = 0
        score_b = 0
        boost_a = 0
        boost_b = 0

        for i in range(10):
            ea = a[i] * (1.5 if boost_a == 1 else 1.0)
            eb = b[i] * (1.5 if boost_b == 1 else 1.0)

            if ea > eb:
                score_a += i + 1
                diff = ea - eb
                boost_b = 1 if (diff > 0.0 and diff <= 5.0) else 0
                boost_a = 0
            elif eb > ea:
                score_b += i + 1
                diff = eb - ea
                boost_a = 1 if (diff > 0.0 and diff <= 5.0) else 0
                boost_b = 0
            else:
                boost_a = 0
                boost_b = 0

        if score_a > score_b:
            return 1
        elif score_b > score_a:
            return -1
        else:
            return 0

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
        score_a = 0
        score_b = 0
        boost_a = False
        boost_b = False

        for i in range(10):
            ea = float(a[i]) * (BOOST if boost_a else 1.0)
            eb = float(b[i]) * (BOOST if boost_b else 1.0)

            if ea > eb:
                score_a += i + 1
                diff = ea - eb
                boost_b = (0.0 < diff <= CLOSE_LOSS)
                boost_a = False
            elif eb > ea:
                score_b += i + 1
                diff = eb - ea
                boost_a = (0.0 < diff <= CLOSE_LOSS)
                boost_b = False
            else:
                boost_a = False
                boost_b = False

        if score_a > score_b:
            return 1
        elif score_b > score_a:
            return -1
        else:
            return 0

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

def random_strategy(rng):
    p = rng.dirichlet(np.ones(K))
    x = rng.multinomial(TOTAL, p).astype(np.int16)
    return x


def repair_sum_100(x):
    """
    Ensure x is int array, nonnegative, sum=100
    """
    x = np.asarray(x, dtype=np.int16).copy()
    x[x < 0] = 0
    s = int(x.sum())

    if s == TOTAL:
        return x

    if s < TOTAL:
        x[9] += (TOTAL - s)
        return x

    # s > TOTAL
    extra = s - TOTAL
    for i in range(K - 1, -1, -1):
        take = min(int(x[i]), extra)
        x[i] -= take
        extra -= take
        if extra == 0:
            break
    return x


def mutate_strategy(x, rng, max_step=8, moves=3):
    """
    Randomly move coins across piles, preserving sum=100
    """
    y = x.copy()

    m = rng.integers(1, moves + 1)
    for _ in range(m):
        i = int(rng.integers(0, K))
        j = int(rng.integers(0, K))
        while j == i:
            j = int(rng.integers(0, K))

        if y[i] == 0:
            continue

        step = int(rng.integers(1, max_step + 1))
        step = min(step, int(y[i]))
        y[i] -= step
        y[j] += step

    return y


def structured_seeds():
    """
    Some handcrafted starting points
    """
    seeds = []

    seeds.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 50, 50], dtype=np.int16))
    seeds.append(np.array([0, 0, 0, 0, 0, 0, 0, 34, 33, 33], dtype=np.int16))
    seeds.append(np.array([0, 0, 0, 0, 0, 0, 20, 25, 27, 28], dtype=np.int16))
    seeds.append(np.array([0, 0, 0, 0, 0, 16, 16, 17, 25, 26], dtype=np.int16))
    seeds.append(np.array([0, 0, 0, 0, 10, 10, 10, 20, 25, 25], dtype=np.int16))
    seeds.append(np.array([0, 0, 0, 8, 8, 8, 16, 18, 20, 22], dtype=np.int16))
    seeds.append(np.array([4, 4, 4, 4, 4, 10, 14, 16, 18, 22], dtype=np.int16))
    seeds.append(np.array([10] * 10, dtype=np.int16))

    return [repair_sum_100(s) for s in seeds]


# =========================================================
# Search
# =========================================================

def score_tuple(w, t, l):
    """
    Primary: wins
    Secondary: ties
    """
    return (w, t, -l)


def local_search(pool, x0, rng, iters=3000, neighborhood=40, max_step=8, verbose=False):
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
            y = mutate_strategy(x, rng, max_step=max_step, moves=3)
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
            x = mutate_strategy(best_local, rng, max_step=max_step * 2, moves=6)
            w, t, l = evaluate_against_pool(x, pool)

        if verbose and (it % 200 == 0):
            print(f"    iter={it:5d} | local_best wins={best_w} ties={best_t} losses={best_l} | stall={stall}")

    return best_local, best_w, best_t, best_l


def global_search(
    pool,
    restarts=30,
    iters=3000,
    neighborhood=40,
    max_step=8,
    seed=1,
    verbose=True,
):
    rng = np.random.default_rng(seed)

    global_best = None
    global_w = -1
    global_t = -1
    global_l = 10**9

    seeds = structured_seeds()

    t0 = time.time()

    for r in range(1, restarts + 1):
        # 初始化策略来源
        p = rng.random()

        if p < 0.4:
            # 来自 top10k
            idx = rng.integers(0, pool.shape[0])
            x0 = pool[idx].copy()
            x0 = mutate_strategy(x0, rng, max_step=5, moves=2)

        elif p < 0.8:
            # 随机
            x0 = random_strategy(rng)

        else:
            # 手工结构
            seeds = structured_seeds()
            x0 = seeds[rng.integers(0, len(seeds))].copy()

        x, w, t, l = local_search(
            pool=pool,
            x0=x0,
            rng=rng,
            iters=iters,
            neighborhood=neighborhood,
            max_step=max_step,
            verbose=False,
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


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results_round4/top10k.npz")
    parser.add_argument("--restarts", type=int, default=30, help="number of global restarts")
    parser.add_argument("--iters", type=int, default=3000, help="local search iterations per restart")
    parser.add_argument("--neighborhood", type=int, default=40, help="mutations tried per iteration")
    parser.add_argument("--max_step", type=int, default=8, help="max coins moved per mutation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save", type=str, default="best_vs_top10k.txt")
    args = parser.parse_args()

    print(f"NUMBA_OK={NUMBA_OK}")
    print(f"Loading {args.input} ...")
    data = np.load(args.input)
    pool = data["strategies"].astype(np.int16)
    print(f"Pool size = {pool.shape[0]:,}")

    best_x, best_w, best_t, best_l = global_search(
        pool=pool,
        restarts=args.restarts,
        iters=args.iters,
        neighborhood=args.neighborhood,
        max_step=args.max_step,
        seed=args.seed,
        verbose=True,
    )

    print("\n=== FINAL BEST ===")
    print("strategy =", best_x.tolist())
    print(f"wins={best_w}, ties={best_t}, losses={best_l}")
    print(f"win_rate={best_w / pool.shape[0]:.4f}")

    with open(args.save, "w", encoding="utf8") as f:
        f.write(f"wins={best_w} ties={best_t} losses={best_l} win_rate={best_w / pool.shape[0]:.6f}\n")
        f.write("strategy: " + " ".join(map(str, best_x.tolist())) + "\n")

    print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
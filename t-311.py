# -*- coding: utf-8 -*-
"""
Search best strategy against top10k.npz for round 11.

规则变化后：
- 策略不要求总和为 100
- 每个位置范围 [0, 1000]
- 搜索更适合使用“按维度独立扰动”而不是搬运总和
"""

import time
import argparse
import numpy as np
import judge

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

K = 10
MAX_BID = 1000


if NUMBA_OK:
    @njit(cache=True)
    def match_result(a, b):
        return judge.calculate_match_result(a, b)

    @njit(cache=True)
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
else:
    def match_result(a, b):
        return judge.calculate_match_result(a, b)

    def evaluate_against_pool(x, pool):
        wins = ties = losses = 0
        for i in range(pool.shape[0]):
            r = match_result(x, pool[i])
            if r == 1:
                wins += 1
            elif r == 0:
                ties += 1
            else:
                losses += 1
        return wins, ties, losses


def clip_strategy(x):
    x = np.asarray(x, dtype=np.int16).copy()
    np.clip(x, 0, MAX_BID, out=x)
    return x


def random_strategy(rng):
    p = rng.random()
    if p < 0.25:
        return rng.integers(0, 101, size=K, dtype=np.int16)
    if p < 0.50:
        x = np.zeros(K, dtype=np.int16)
        idx = rng.choice(np.arange(6, 10), size=int(rng.integers(2, 5)), replace=False)
        x[idx] = rng.integers(15, 110, size=idx.shape[0], dtype=np.int16)
        x[x == 0] = rng.integers(0, 6, size=int((x == 0).sum()), dtype=np.int16)
        return x
    if p < 0.75:
        base = rng.integers(0, 8, size=K)
        slope = int(rng.integers(2, 10))
        noise = rng.integers(-3, 4, size=K)
        return clip_strategy((base + slope * np.arange(K) + noise).astype(np.int16))
    x = rng.integers(0, 5, size=K, dtype=np.int16)
    hot = rng.choice(K, size=int(rng.integers(1, 4)), replace=False)
    x[hot] = rng.integers(30, 220, size=hot.shape[0], dtype=np.int16)
    return x


def structured_seeds(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    seeds = []
    for s in judge.strategies:
        base = clip_strategy(s)
        seeds.append(base.copy())
        for _ in range(60):
            y = base.astype(np.int32).copy()
            idx = rng.choice(K, size=int(rng.integers(1, 5)), replace=False)
            y[idx] += rng.integers(-12, 13, size=idx.shape[0])
            seeds.append(clip_strategy(y))
    return seeds


def mutate_strategy(x, rng, max_delta=20, hot_delta=60, edits=3):
    y = x.astype(np.int32).copy()
    m = int(rng.integers(1, edits + 1))
    for _ in range(m):
        i = int(rng.integers(0, K))
        if rng.random() < 0.75:
            delta = int(rng.integers(-max_delta, max_delta + 1))
            y[i] += delta
        else:
            y[i] = int(rng.integers(0, min(MAX_BID, hot_delta * 5) + 1))
    if rng.random() < 0.15:
        i = int(rng.integers(0, K))
        y[i] = 0
    if rng.random() < 0.15:
        i = int(rng.integers(0, K))
        y[i] = int(rng.integers(80, 201))
    return clip_strategy(y)


def score_tuple(w, t, l):
    return (w, t, -l)


def local_search(pool, x0, rng, iters=2500, neighborhood=32, max_delta=20, verbose=False):
    x = clip_strategy(x0)
    w, t, l = evaluate_against_pool(x, pool)
    best_local = x.copy()
    best_w, best_t, best_l = w, t, l
    stall = 0

    for it in range(1, iters + 1):
        cand_best = None
        cand_score = None
        for _ in range(neighborhood):
            y = mutate_strategy(x, rng, max_delta=max_delta, hot_delta=max_delta * 4, edits=4)
            wy, ty, ly = evaluate_against_pool(y, pool)
            sc = score_tuple(wy, ty, ly)
            if cand_score is None or sc > cand_score:
                cand_score = sc
                cand_best = (y, wy, ty, ly)

        if cand_score > score_tuple(w, t, l):
            x, w, t, l = cand_best
            stall = 0
            if score_tuple(w, t, l) > score_tuple(best_w, best_t, best_l):
                best_local = x.copy()
                best_w, best_t, best_l = w, t, l
        else:
            stall += 1
            x = mutate_strategy(best_local, rng, max_delta=max_delta * 2, hot_delta=max_delta * 8, edits=6)
            w, t, l = evaluate_against_pool(x, pool)

        if verbose and (it == 1 or it % 200 == 0):
            print(f"    iter={it:5d} | local_best wins={best_w} ties={best_t} losses={best_l} | stall={stall}")

    return best_local, best_w, best_t, best_l


def global_search(pool, restarts=30, iters=2500, neighborhood=32, max_delta=20, seed=1, verbose=True):
    rng = np.random.default_rng(seed)
    global_best = None
    global_w = -1
    global_t = -1
    global_l = 10**9
    seeds = structured_seeds(rng=rng)
    t0 = time.time()

    for r in range(1, restarts + 1):
        p = rng.random()
        if p < 0.35:
            idx = int(rng.integers(0, pool.shape[0]))
            x0 = mutate_strategy(pool[idx], rng, max_delta=8, hot_delta=40, edits=3)
            source = "pool"
        elif p < 0.70:
            x0 = random_strategy(rng)
            source = "random"
        else:
            x0 = seeds[int(rng.integers(0, len(seeds)))].copy()
            source = "seed"

        x, w, t, l = local_search(pool, x0, rng, iters=iters, neighborhood=neighborhood, max_delta=max_delta, verbose=False)

        if score_tuple(w, t, l) > score_tuple(global_w, global_t, global_l):
            global_best = x.copy()
            global_w, global_t, global_l = w, t, l

        if verbose:
            dt = time.time() - t0
            print(f"[restart {r:3d}/{restarts}] src={source:<6s} best_this={w}/{t}/{l} | global_best={global_w}/{global_t}/{global_l} | x={global_best.tolist()} | {dt:.1f}s")

    return global_best, global_w, global_t, global_l


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=judge.name + "/top10k.npz")
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--neighborhood", type=int, default=36)
    parser.add_argument("--max_delta", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260313)
    parser.add_argument("--save", type=str, default=judge.name + "/best_vs_top10k.txt")
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
        max_delta=args.max_delta,
        seed=args.seed,
        verbose=True,
    )

    print("\n=== FINAL BEST ===")
    print("strategy =", best_x.tolist())
    print(f"wins={best_w}, ties={best_t}, losses={best_l}")
    print(f"win_rate={best_w / pool.shape[0]:.4f}")

    with open(args.save, "a", encoding="utf8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} wins={best_w} ties={best_t} losses={best_l} win_rate={best_w / pool.shape[0]:.6f}\n")
        f.write("strategy: " + " ".join(map(str, best_x.tolist())) + "\n")

    print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()

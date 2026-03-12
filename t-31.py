# -*- coding: utf-8 -*-
"""
Round 8 anti-overfit search

规则：
- 赢更多沙堆个数的人获胜
- 若赢得沙堆个数相同，则比较赢得沙堆总价值
- 若仍相同，则平局

目标：
- 从 top10k.npz 中搜索一条更稳健、不容易过拟合的策略

改进点：
1. train/valid split
2. sample-based evaluation during search
3. opponent augmentation by jitter
4. final selection prefers valid performance, not just train
"""

import argparse
import time
import numpy as np
import judge

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False


TOTAL = 100
K = 10


# =========================================================
# Round 8 judge
# =========================================================

if NUMBA_OK:
    @njit(cache=True)
    def match_result(a, b):
        return judge.calculate_match_result(a, b)

    @njit(cache=True)
    def evaluate_against_pool_full(x, pool):
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

    def evaluate_against_pool_full(x, pool):
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
# Utils
# =========================================================

def repair_sum_100(x):
    x = np.asarray(x, dtype=np.int16).copy()
    x[x < 0] = 0
    s = int(x.sum())

    if s == TOTAL:
        return x

    if s < TOTAL:
        x[-1] += (TOTAL - s)
        return x

    extra = s - TOTAL
    for i in range(K - 1, -1, -1):
        take = min(int(x[i]), extra)
        x[i] -= take
        extra -= take
        if extra == 0:
            break
    return x


def random_strategy(rng):
    p = rng.dirichlet(np.ones(K))
    return rng.multinomial(TOTAL, p).astype(np.int16)


def mutate_strategy(x, rng, max_step=8, moves=3):
    y = x.copy()
    m = int(rng.integers(1, moves + 1))

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


def jitter_strategy(base, rng, max_delta=2):
    noise = rng.integers(-max_delta, max_delta + 1, size=K)
    y = np.asarray(base, dtype=np.int16) + noise
    return repair_sum_100(y)


def score_tuple(w, t, l):
    return (w, t, -l)


def robust_score(train_w, train_t, train_l, valid_w, valid_t, valid_l):
    """
    选择全局最优时，更看重 valid，避免 train 过拟合。
    """
    return (
        valid_w,
        valid_t,
        -valid_l,
        train_w,
        train_t,
        -train_l,
    )


# =========================================================
# Sampling / augmentation
# =========================================================

def sample_pool(pool, rng, sample_size):
    n = pool.shape[0]
    if sample_size >= n:
        return pool
    idx = rng.choice(n, size=sample_size, replace=False)
    return pool[idx]


def build_augmented_pool(pool, rng, copies=1, max_delta=2):
    """
    给训练集做轻微扰动增强
    """
    out = [pool]
    for _ in range(copies):
        aug = np.empty_like(pool)
        for i in range(pool.shape[0]):
            aug[i] = jitter_strategy(pool[i], rng=rng, max_delta=max_delta)
        out.append(aug)
    return np.vstack(out).astype(np.int16)


def train_valid_split(pool, rng, valid_ratio=0.2):
    n = pool.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_valid = max(1, int(n * valid_ratio))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    return pool[train_idx], pool[valid_idx]


# =========================================================
# Evaluation helper
# =========================================================

def evaluate_sample(x, pool, rng, sample_size):
    sub = sample_pool(pool, rng, sample_size)
    return evaluate_against_pool_full(x, sub)


# =========================================================
# Search
# =========================================================

def local_search(
    train_pool,
    valid_pool,
    x0,
    rng,
    iters=1500,
    neighborhood=24,
    max_step=8,
    train_sample=2000,
    shake_every=30,
    verbose=False,
):
    """
    局部搜索只在 train sample 上做提升判断
    但周期性查看 valid，避免一路朝着 train 过拟合
    """
    x = x0.copy()

    w, t, l = evaluate_sample(x, train_pool, rng, train_sample)
    best_local = x.copy()
    best_train = (w, t, l)
    best_valid = evaluate_against_pool_full(best_local, valid_pool)

    stall = 0

    for it in range(1, iters + 1):
        cand_best = None
        cand_score = None

        for _ in range(neighborhood):
            y = mutate_strategy(x, rng, max_step=max_step, moves=3)
            wy, ty, ly = evaluate_sample(y, train_pool, rng, train_sample)
            sc = score_tuple(wy, ty, ly)

            if cand_score is None or sc > cand_score:
                cand_score = sc
                cand_best = (y, wy, ty, ly)

        if cand_score > score_tuple(w, t, l):
            x, w, t, l = cand_best
            stall = 0
        else:
            stall += 1

        # 每隔几轮，用 full valid 检查一次
        if (it % 20 == 0) or (stall == 0):
            vw, vt, vl = evaluate_against_pool_full(x, valid_pool)
            if robust_score(w, t, l, vw, vt, vl) > robust_score(
                best_train[0], best_train[1], best_train[2],
                best_valid[0], best_valid[1], best_valid[2]
            ):
                best_local = x.copy()
                best_train = (w, t, l)
                best_valid = (vw, vt, vl)

        # 卡住时抖动一下
        if stall >= shake_every:
            x = mutate_strategy(best_local, rng, max_step=max_step * 2, moves=6)
            w, t, l = evaluate_sample(x, train_pool, rng, train_sample)
            stall = 0

        if verbose and (it % 100 == 0):
            print(
                f"    iter={it:4d} | "
                f"train={best_train[0]}/{best_train[1]}/{best_train[2]} | "
                f"valid={best_valid[0]}/{best_valid[1]}/{best_valid[2]}"
            )

    return best_local, best_train, best_valid


def make_seed_pool(base_pool, rng, extra_jitter_per_seed=2, jitter_delta=2):
    seeds = []

    # 原始 pool 抽一部分作种子
    take = min(500, base_pool.shape[0])
    idx = rng.choice(base_pool.shape[0], size=take, replace=False)

    for i in idx:
        s = base_pool[i].copy()
        seeds.append(s)
        for _ in range(extra_jitter_per_seed):
            seeds.append(jitter_strategy(s, rng=rng, max_delta=jitter_delta))

    return np.array(seeds, dtype=np.int16)


def global_search(
    pool,
    restarts=30,
    iters=1500,
    neighborhood=24,
    max_step=8,
    seed=1,
    valid_ratio=0.2,
    train_sample=2000,
    aug_copies=1,
    aug_delta=2,
    verbose=True,
):
    rng = np.random.default_rng(seed)

    # train / valid split
    train_base, valid_pool = train_valid_split(pool, rng, valid_ratio=valid_ratio)

    # augmentation 只对 train 做
    train_pool = build_augmented_pool(
        train_base, rng=rng, copies=aug_copies, max_delta=aug_delta
    )

    # 准备 seed 池
    seed_pool = make_seed_pool(train_base, rng=rng, extra_jitter_per_seed=2, jitter_delta=2)

    global_best = None
    global_train = (-1, -1, 10**9)
    global_valid = (-1, -1, 10**9)

    t0 = time.time()

    for r in range(1, restarts + 1):
        p = rng.random()

        if p < 0.35:
            # 来自训练池真实策略附近
            idx = int(rng.integers(0, train_base.shape[0]))
            x0 = train_base[idx].copy()
            x0 = mutate_strategy(x0, rng, max_step=5, moves=2)

        elif p < 0.70:
            # 完全随机
            x0 = random_strategy(rng)

        else:
            # 来自扰动种子池
            idx = int(rng.integers(0, seed_pool.shape[0]))
            x0 = seed_pool[idx].copy()

        x, train_res, valid_res = local_search(
            train_pool=train_pool,
            valid_pool=valid_pool,
            x0=x0,
            rng=rng,
            iters=iters,
            neighborhood=neighborhood,
            max_step=max_step,
            train_sample=train_sample,
            shake_every=30,
            verbose=False,
        )

        if robust_score(*train_res, *valid_res) > robust_score(*global_train, *global_valid):
            global_best = x.copy()
            global_train = train_res
            global_valid = valid_res

        if verbose:
            dt = time.time() - t0
            print(
                f"[restart {r:3d}/{restarts}] "
                f"train={train_res[0]}/{train_res[1]}/{train_res[2]} | "
                f"valid={valid_res[0]}/{valid_res[1]}/{valid_res[2]} | "
                f"global_valid={global_valid[0]}/{global_valid[1]}/{global_valid[2]} | "
                f"x={global_best.tolist()} | {dt:.1f}s"
            )

    # 最终对原始全池做 full eval
    full_res = evaluate_against_pool_full(global_best, pool)
    return global_best, global_train, global_valid, full_res


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=judge.name + "/top10k.npz")
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--neighborhood", type=int, default=24)
    parser.add_argument("--max_step", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260310)

    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--train_sample", type=int, default=4000)
    parser.add_argument("--aug_copies", type=int, default=1)
    parser.add_argument("--aug_delta", type=int, default=2)

    parser.add_argument("--save", type=str, default=judge.name + "/best_vs_top10k_robust.txt")
    args = parser.parse_args()

    print(f"NUMBA_OK={NUMBA_OK}")
    print(f"Loading {args.input} ...")

    data = np.load(args.input)
    pool = data["strategies"].astype(np.int16)

    print(f"Pool size = {pool.shape[0]:,}")

    print("=== sanity check 1 ===")
    a = np.array([10,10,10,10,10,10,10,10,10,10], dtype=np.int16)
    b = np.array([0,0,0,0,0,0,0,0,0,100], dtype=np.int16)
    print("expect 1, got:", match_result(a, b))

    print("=== sanity check 2 ===")
    a = np.array([1,1,1,1,1,1,1,1,1,91], dtype=np.int16)
    b = np.array([2,2,2,2,2,2,2,2,2,82], dtype=np.int16)
    print("expect -1, got:", match_result(a, b))

    print("=== sanity check 3 ===")
    for i in range(3):
        x = pool[i]
        w, t, l = evaluate_against_pool_full(x, pool[:200])
        print(i, x.tolist(), w, t, l)

    best_x, train_res, valid_res, full_res = global_search(
        pool=pool,
        restarts=args.restarts,
        iters=args.iters,
        neighborhood=args.neighborhood,
        max_step=args.max_step,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        train_sample=args.train_sample,
        aug_copies=args.aug_copies,
        aug_delta=args.aug_delta,
        verbose=True,
    )

    print("\n=== FINAL BEST ===")
    print("strategy =", best_x.tolist())
    print(f"train = {train_res[0]}/{train_res[1]}/{train_res[2]}")
    print(f"valid = {valid_res[0]}/{valid_res[1]}/{valid_res[2]}")
    print(f"full  = {full_res[0]}/{full_res[1]}/{full_res[2]}")
    print(f"full_win_rate = {full_res[0] / pool.shape[0]:.6f}")

    with open(args.save, "w", encoding="utf8") as f:
        f.write("strategy: " + " ".join(map(str, best_x.tolist())) + "\n")
        f.write(f"train={train_res[0]} {train_res[1]} {train_res[2]}\n")
        f.write(f"valid={valid_res[0]} {valid_res[1]} {valid_res[2]}\n")
        f.write(f"full={full_res[0]} {full_res[1]} {full_res[2]}\n")
        f.write(f"full_win_rate={full_res[0] / pool.shape[0]:.6f}\n")

    print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
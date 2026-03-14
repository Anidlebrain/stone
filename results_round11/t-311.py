# -*- coding: utf-8 -*-
"""
Robust constrained search for round 11.

新增硬约束：
1. 输出策略必须满足 sum(x) <= 100
2. 输出策略必须战胜全十序列 [10,10,...,10]
3. 原始 round11 判定规则不变

搜索思路：
- 继续使用 train / valid / fresh 三池稳健搜索，避免单池过拟合
- 但把搜索空间限制在“个人偏好可接受”的子空间内
- 通过手工种子 + 定向随机生成 + 约束修复，避免再出现 >100 的解
"""

import time
import argparse
import numpy as np
import judge

try:
    import t11 as genmod
except Exception:
    genmod = None

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

K = 10
MAX_BID = 1000
TOTAL_CAP = 100
DTYPE = np.int16
ALL_TENS = np.full(K, 10, dtype=DTYPE)


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
    x = np.asarray(x, dtype=np.int32).copy()
    np.clip(x, 0, MAX_BID, out=x)
    return x.astype(DTYPE)


def repair_sum_leq_100(x):
    """修到 sum<=100。优先从低价值堆扣，尽量保留高价值堆结构。"""
    y = np.asarray(x, dtype=np.int32).copy()
    y[y < 0] = 0
    np.clip(y, 0, MAX_BID, out=y)
    extra = int(y.sum()) - TOTAL_CAP
    if extra <= 0:
        return y.astype(DTYPE)

    for i in range(K):
        if extra <= 0:
            break
        take = min(int(y[i]), extra)
        y[i] -= take
        extra -= take

    if extra > 0:
        for i in range(K - 1, -1, -1):
            if extra <= 0:
                break
            take = min(int(y[i]), extra)
            y[i] -= take
            extra -= take

    return y.astype(DTYPE)


def l1_distance(a, b):
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).sum())


def beats_all_tens(x):
    return match_result(np.asarray(x, dtype=DTYPE), ALL_TENS) == 1


def safe_repair(x):
    return repair_sum_leq_100(clip_strategy(x))


def try_force_beats_all_tens(x):
    y = safe_repair(x).astype(np.int32)
    if beats_all_tens(y.astype(DTYPE)):
        return y.astype(DTYPE)

    templates = [
        np.array([0, 0, 0, 0, 0, 10, 11, 21, 22, 23], dtype=np.int32),
        np.array([0, 0, 0, 0, 0, 0, 14, 24, 25, 26], dtype=np.int32),
        np.array([0, 0, 0, 0, 0, 9, 12, 22, 23, 24], dtype=np.int32),
        np.array([0, 0, 0, 0, 0, 8, 13, 23, 24, 25], dtype=np.int32),
    ]
    for tpl in templates:
        z = tpl.copy()
        z[6:] = np.maximum(z[6:], y[6:])
        z = repair_sum_leq_100(z)
        if beats_all_tens(z):
            return z

    donors = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    targets = np.array([9, 8, 7, 6], dtype=np.int32)
    for _ in range(20):
        if beats_all_tens(y.astype(DTYPE)):
            return y.astype(DTYPE)
        donor = -1
        for i in donors:
            if y[i] > 0:
                donor = int(i)
                break
        if donor < 0:
            break
        target = None
        for j in targets:
            if y[j] <= 10:
                target = int(j)
                break
        if target is None:
            target = 9
        move = min(int(y[donor]), max(1, 11 - int(y[target])))
        y[donor] -= move
        y[target] += move
        y = repair_sum_leq_100(y).astype(np.int32)

    return y.astype(DTYPE)


def make_valid_strategy(x):
    return safe_repair(try_force_beats_all_tens(x))


def random_composition_leq100(rng, total_low=60, total_high=100):
    total = int(rng.integers(total_low, total_high + 1))
    p = rng.dirichlet(np.ones(K))
    x = rng.multinomial(total, p).astype(np.int32)
    return x.astype(DTYPE)


def random_tail_focus(rng):
    x = np.zeros(K, dtype=np.int32)
    x[:6] = rng.integers(0, 5, size=6)
    x[6:] = rng.integers(8, 26, size=4)
    x = repair_sum_leq_100(x)
    return x.astype(DTYPE)


def random_strategy(rng):
    for _ in range(60):
        p = rng.random()
        if p < 0.28:
            x = random_composition_leq100(rng, 65, 100)
        elif p < 0.56:
            x = random_tail_focus(rng)
        elif p < 0.78:
            if genmod is not None:
                batch, _ = genmod.generate_batch_mixed(rng, 1)
                x = repair_sum_leq_100(batch[0])
            else:
                x = random_composition_leq100(rng, 55, 95)
        else:
            bases = np.array([
                [0, 0, 0, 0, 0, 10, 11, 21, 22, 23],
                [0, 0, 0, 0, 0, 0, 14, 24, 25, 26],
                [0, 0, 0, 0, 0, 9, 12, 22, 23, 24],
                [0, 0, 0, 0, 1, 8, 13, 23, 24, 25],
                [1, 1, 1, 1, 1, 8, 12, 22, 24, 26],
            ], dtype=np.int32)
            b = bases[int(rng.integers(0, bases.shape[0]))].copy()
            noise = rng.integers(-3, 4, size=K)
            x = repair_sum_leq_100(b + noise)

        x = make_valid_strategy(x)
        if beats_all_tens(x):
            return x

    return np.array([0, 0, 0, 0, 0, 10, 11, 21, 22, 23], dtype=DTYPE)


def structured_seeds(rng):
    seeds = []
    hardcoded = [
        [0, 0, 0, 0, 0, 10, 11, 21, 22, 23],
        [0, 0, 0, 0, 0, 0, 14, 24, 25, 26],
        [0, 0, 0, 0, 0, 9, 12, 22, 23, 24],
        [0, 0, 0, 0, 1, 8, 13, 23, 24, 25],
        [1, 1, 1, 1, 1, 8, 12, 22, 24, 26],
        [2, 2, 2, 2, 2, 8, 12, 21, 23, 24],
    ]
    for s in hardcoded:
        base = make_valid_strategy(np.array(s, dtype=np.int32))
        if beats_all_tens(base):
            seeds.append(base.copy())
        for _ in range(25):
            y = base.astype(np.int32).copy()
            idx = rng.choice(K, size=int(rng.integers(1, 5)), replace=False)
            y[idx] += rng.integers(-4, 5, size=idx.shape[0])
            y = make_valid_strategy(y)
            if beats_all_tens(y):
                seeds.append(y)

    for s in getattr(judge, 'strategies', []):
        base = make_valid_strategy(np.asarray(s, dtype=np.int32))
        if beats_all_tens(base):
            seeds.append(base.copy())
        for _ in range(15):
            y = base.astype(np.int32).copy()
            idx = rng.choice(K, size=int(rng.integers(1, 4)), replace=False)
            y[idx] += rng.integers(-4, 5, size=idx.shape[0])
            y = make_valid_strategy(y)
            if beats_all_tens(y):
                seeds.append(y)

    if not seeds:
        seeds.append(
            np.array([0, 0, 0, 0, 0, 10, 11, 21, 22, 23], dtype=DTYPE))
    return seeds


def mutate_strategy(x, rng, max_delta=12, edits=3):
    y = np.asarray(x, dtype=np.int32).copy()
    m = int(rng.integers(1, edits + 1))
    for _ in range(m):
        r = rng.random()
        if r < 0.45:
            i = int(rng.integers(0, K))
            j = int(rng.integers(0, K))
            while j == i:
                j = int(rng.integers(0, K))
            can = int(y[i])
            if can > 0:
                step = min(can, int(rng.integers(1, max_delta + 1)))
                y[i] -= step
                y[j] += step
        elif r < 0.75:
            i = int(rng.integers(0, K))
            y[i] += int(rng.integers(-max_delta, max_delta + 1))
        elif r < 0.90:
            i = int(rng.integers(0, K))
            j = int(rng.integers(0, K))
            while j == i:
                j = int(rng.integers(0, K))
            y[i], y[j] = y[j], y[i]
        else:
            j = int(rng.choice(np.array([6, 7, 8, 9], dtype=np.int32)))
            y[j] += int(rng.integers(1, max_delta + 1))

    y = make_valid_strategy(y)
    if not beats_all_tens(y):
        y = np.array([0, 0, 0, 0, 0, 10, 11, 21, 22, 23], dtype=DTYPE)
    return y


def split_pool(pool, rng, train_ratio=0.7):
    n = pool.shape[0]
    idx = rng.permutation(n)
    cut = max(1, int(n * train_ratio))
    train = pool[idx[:cut]].copy()
    valid = pool[idx[cut:]].copy()
    if valid.shape[0] == 0:
        valid = train.copy()
    return train, valid


def make_fresh_pool(rng, n=2000):
    arr = np.empty((n, K), dtype=DTYPE)
    for i in range(n):
        arr[i] = random_strategy(rng)
    return arr


def robust_metrics(x, train_pool, valid_pool, fresh_pool):
    wt, tt, lt = evaluate_against_pool(x, train_pool)
    wv, tv, lv = evaluate_against_pool(x, valid_pool)
    wf, tf, lf = evaluate_against_pool(x, fresh_pool)
    r10 = match_result(x, ALL_TENS)
    return (wt, tt, lt), (wv, tv, lv), (wf, tf, lf), r10


def robust_score(metrics, x):
    (wt, tt, lt), (wv, tv, lv), (wf, tf, lf), r10 = metrics
    sx = int(np.asarray(x, dtype=np.int32).sum())
    valid_flag = 1 if (sx <= TOTAL_CAP and r10 == 1) else 0
    return (
        valid_flag,
        min(wt, wv, wf),
        wv,
        wf,
        wt,
        min(tt, tv, tf),
        -(lt + lv + lf),
        -sx,
    )


def local_search(train_pool, valid_pool, fresh_pool, x0, rng, iters=1500, neighborhood=20, max_delta=12, verbose=False):
    x = make_valid_strategy(x0)
    metrics = robust_metrics(x, train_pool, valid_pool, fresh_pool)
    best_local = x.copy()
    best_metrics = metrics
    current_metrics = metrics
    stall = 0

    for it in range(1, iters + 1):
        cand_best = None
        cand_score = None
        base = best_local if stall >= 6 else x
        for _ in range(neighborhood):
            y = mutate_strategy(base, rng, max_delta=max_delta, edits=4)
            my = robust_metrics(y, train_pool, valid_pool, fresh_pool)
            sc = robust_score(my, y)
            if cand_score is None or sc > cand_score:
                cand_score = sc
                cand_best = (y, my)

        if cand_score > robust_score(current_metrics, x):
            x = cand_best[0]
            current_metrics = cand_best[1]
            stall = 0
            if cand_score > robust_score(best_metrics, best_local):
                best_local = x.copy()
                best_metrics = current_metrics
        else:
            stall += 1
            x = mutate_strategy(
                best_local, rng, max_delta=max_delta * 2, edits=6)
            current_metrics = robust_metrics(
                x, train_pool, valid_pool, fresh_pool)

        if it % 150 == 0:
            fresh_pool = make_fresh_pool(rng, n=fresh_pool.shape[0])
            current_metrics = robust_metrics(
                x, train_pool, valid_pool, fresh_pool)
            best_metrics = robust_metrics(
                best_local, train_pool, valid_pool, fresh_pool)

        if verbose and (it == 1 or it % 150 == 0):
            (wt, tt, lt), (wv, tv, lv), (wf, tf, lf), r10 = best_metrics
            print(f"    iter={it:4d} | train={wt}/{tt}/{lt} | valid={wv}/{tv}/{lv} | fresh={wf}/{tf}/{lf} | sum={int(best_local.sum())} | vs10={r10} | stall={stall}")

    return best_local, best_metrics


def is_distinct(x, arrs, min_l1=16):
    for y in arrs:
        if l1_distance(x, y) < min_l1:
            return False
    return True


def global_search(pool, restarts=100, iters=1800, neighborhood=20, max_delta=12, seed=1, keep_top=12, verbose=True):
    rng = np.random.default_rng(seed)
    train_pool, valid_pool = split_pool(pool, rng, train_ratio=0.7)
    seeds = structured_seeds(rng)
    fresh_pool = make_fresh_pool(rng, n=min(
        3000, max(1200, pool.shape[0] // 3)))

    survivors = []
    t0 = time.time()
    for r in range(1, restarts + 1):
        p = rng.random()
        if p < 0.30:
            idx = int(rng.integers(0, train_pool.shape[0]))
            x0 = mutate_strategy(make_valid_strategy(
                train_pool[idx]), rng, max_delta=6, edits=3)
            source = 'train'
        elif p < 0.55:
            idx = int(rng.integers(0, valid_pool.shape[0]))
            x0 = mutate_strategy(make_valid_strategy(
                valid_pool[idx]), rng, max_delta=6, edits=3)
            source = 'valid'
        elif p < 0.80:
            x0 = random_strategy(rng)
            source = 'random'
        else:
            x0 = seeds[int(rng.integers(0, len(seeds)))].copy()
            source = 'seed'

        x, metrics = local_search(train_pool, valid_pool, fresh_pool, x0, rng,
                                  iters=iters, neighborhood=neighborhood, max_delta=max_delta, verbose=False)
        sc = robust_score(metrics, x)
        survivors.append((sc, x.copy(), metrics, source))
        survivors.sort(key=lambda z: z[0], reverse=True)
        survivors = survivors[: max(keep_top * 4, 20)]

        if verbose:
            dt = time.time() - t0
            (wt, tt, lt), (wv, tv, lv), (wf, tf, lf), r10 = metrics
            print(f"[restart {r:3d}/{restarts}] src={source:<6s} | train={wt}/{tt}/{lt} | valid={wv}/{tv}/{lv} | fresh={wf}/{tf}/{lf} | sum={int(x.sum())} | vs10={r10} | x={x.tolist()} | {dt:.1f}s")

    picked = []
    out = []
    for sc, x, metrics, source in survivors:
        sx = int(x.sum())
        (_, _, _), (_, _, _), (_, _, _), r10 = metrics
        if sx <= TOTAL_CAP and r10 == 1 and is_distinct(x, picked, min_l1=16):
            picked.append(x)
            out.append((sc, x, metrics, source))
        if len(out) >= keep_top:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default=judge.name + '/top10k.npz')
    parser.add_argument('--restarts', type=int, default=100)
    parser.add_argument('--iters', type=int, default=1800)
    parser.add_argument('--neighborhood', type=int, default=20)
    parser.add_argument('--max_delta', type=int, default=12)
    parser.add_argument('--seed', type=int, default=202603140010)
    parser.add_argument('--keep_top', type=int, default=12)
    parser.add_argument('--save', type=str, default=judge.name +
                        '/robust_best_vs_top10k_le100.txt')
    args = parser.parse_args()

    print(f'NUMBA_OK={NUMBA_OK}')
    print(f'Loading {args.input} ...')
    data = np.load(args.input)
    pool = data['strategies'].astype(DTYPE)
    print(f'Pool size = {pool.shape[0]:,}')
    print('Hard constraints: sum<=100 and must beat [10]*10')

    results = global_search(
        pool=pool,
        restarts=args.restarts,
        iters=args.iters,
        neighborhood=args.neighborhood,
        max_delta=args.max_delta,
        seed=args.seed,
        keep_top=args.keep_top,
        verbose=True,
    )

    print('\n=== ROBUST CANDIDATES (sum<=100 and beat all-tens) ===')
    with open(args.save, 'a', encoding='utf8') as f:
        f.write(f'\n### {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for rank, (sc, x, metrics, source) in enumerate(results, 1):
            (wt, tt, lt), (wv, tv, lv), (wf, tf, lf), r10 = metrics
            sx = int(x.sum())
            print(f'[{rank}] src={source} | train={wt}/{tt}/{lt} | valid={wv}/{tv}/{lv} | fresh={wf}/{tf}/{lf} | sum={sx} | vs10={r10} | x={x.tolist()}')
            f.write(f'[{rank}] src={source} | train={wt}/{tt}/{lt} | valid={wv}/{tv}/{lv} | fresh={wf}/{tf}/{lf} | sum={sx} | vs10={r10} | x={" ".join(map(str, x.tolist()))}\n')

    print(f'Saved to {args.save}')


if __name__ == '__main__':
    main()

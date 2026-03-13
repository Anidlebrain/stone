# -*- coding: utf-8 -*-
"""
Round 11 multi-stage evolution (smart generator version).

核心改动：
1. 第11回合不再要求总和=100，而是每个位置在 [0,1000]
2. 随机生成不再“无脑纯随机”，而是混合多种策略族：
   - 安全型：总和 < 100 / =100，尽量避免自己先爆
   - 零陷阱型：大量 0，诱导对手高价抢堆后超标
   - 高价值偏置型：重点覆盖 7/8/9/10 号堆
   - 阶梯差值型：110/120/130/... 一类的差值设计
   - 中度超预算型：总和 > 100，但分布可控，不追求极端
   - 尖峰型：少数大点 + 其余很小，用来打穿特定对手
3. 批量生成 + 批量去重，显著加快初始池构造速度
4. 自动保留 judge.strategies 里的手工种子

说明：
- 本文件默认优先 import 同目录 judge.py。
- 如果同目录 judge.py 不是第11回合版本，则自动尝试导入 judge_round11.py。
"""

import os
import time
import argparse
import importlib
import numpy as np


# -----------------------------
# Judge import (robust fallback)
# -----------------------------

def load_judge_module():
    judge = importlib.import_module("judge")
    ok = getattr(judge, "round_no", None) == 11 and hasattr(judge, "calculate_match_result")
    if ok:
        return judge
    try:
        judge = importlib.import_module("judge_round11")
        return judge
    except Exception:
        return judge


judge = load_judge_module()

# ---- Optional speedup ----
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

K = 10
MAX_BID = 1000
START_COINS = 100

# 为了保证去重 key 稳定，统一使用 int16 存储
DTYPE = np.int16


# -----------------------------
# Utility
# -----------------------------

def clip_strategies(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int32)
    np.clip(arr, 0, MAX_BID, out=arr)
    return arr.astype(DTYPE, copy=False)


def unique_rows_preserve_first(arr: np.ndarray) -> np.ndarray:
    """批量内去重，保留首次出现顺序。"""
    if arr.shape[0] <= 1:
        return arr.astype(DTYPE, copy=False)
    arr16 = np.ascontiguousarray(arr.astype(DTYPE, copy=False))
    keys = arr16.view(np.dtype((np.void, arr16.dtype.itemsize * arr16.shape[1]))).ravel()
    _, idx = np.unique(keys, return_index=True)
    idx.sort()
    return arr16[idx]


def key_of(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v.astype(DTYPE, copy=False)).tobytes()


# -----------------------------
# Smart random generators
# -----------------------------

def gen_safe_under100(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    总和通常在 55~95：
    不容易自己先爆，适合打中等出价和盲目抬价对手。
    """
    alpha = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.55, 1.8, 2.1], dtype=np.float64)
    p = rng.dirichlet(alpha, size=n)
    totals = rng.integers(55, 96, size=n)
    x = np.floor(p * totals[:, None]).astype(np.int32)
    rem = totals - x.sum(axis=1)
    for i in range(n):
        if rem[i] > 0:
            idx = rng.choice(K, size=int(rem[i]), replace=True, p=alpha / alpha.sum())
            np.add.at(x[i], idx, 1)
    return clip_strategies(x)


def gen_exact100(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    总和=100：
    是很自然的一大类，会成为很多人的基线策略，必须覆盖。
    """
    alpha = np.array([0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.25, 1.4, 1.65, 1.9], dtype=np.float64)
    p = rng.dirichlet(alpha, size=n)
    x = rng.multinomial(100, alpha / alpha.sum(), size=n).astype(np.int32)
    mix_mask = rng.random(n) < 0.65
    if np.any(mix_mask):
        y = np.floor(p[mix_mask] * 100).astype(np.int32)
        rem = 100 - y.sum(axis=1)
        for i in range(y.shape[0]):
            if rem[i] > 0:
                idx = rng.choice(K, size=int(rem[i]), replace=True, p=alpha / alpha.sum())
                np.add.at(y[i], idx, 1)
        x[mix_mask] = y
    return clip_strategies(x)


def gen_zero_trap(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    大量 0 + 少数关键堆出价：
    诱导对面用较大代价拿到一些低/中价值堆，自己则集中拿高价值堆，
    或者让对面在错误的位置继续加价导致先爆。
    """
    x = np.zeros((n, K), dtype=np.int32)
    for i in range(n):
        hot = int(rng.integers(2, 5))
        tail_hot = rng.random() < 0.75
        pool = np.arange(5, 10) if tail_hot else np.arange(K)
        idx = rng.choice(pool, size=hot, replace=False)
        vals = rng.integers(18, 76, size=hot)
        x[i, idx] = vals

        # 额外设置 0/1/2/3 一类的诱导点
        tiny_cnt = int(rng.integers(0, 3))
        rest = np.setdiff1d(np.arange(K), idx, assume_unique=True)
        if tiny_cnt > 0 and rest.size > 0:
            tidx = rng.choice(rest, size=min(tiny_cnt, rest.size), replace=False)
            x[i, tidx] = rng.integers(0, 4, size=tidx.shape[0])

        # 偶尔做一个明显的“挡板”值
        if rng.random() < 0.40:
            j = int(rng.choice(idx))
            x[i, j] = int(rng.choice(np.array([85, 95, 100, 110, 120, 130, 140, 150], dtype=np.int32)))
    return clip_strategies(x)


def gen_tail_focus(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    偏向高价值堆(6~10)：
    这是第11回合非常自然的一类，因为后段堆价值高，且许多人会把预算留到后段。
    """
    x = rng.integers(0, 8, size=(n, K), dtype=np.int32)
    boost = rng.integers(8, 36, size=(n, 5), dtype=np.int32)
    x[:, 5:] += boost
    if n > 0:
        m = rng.integers(1, 3, size=n)
        for i in range(n):
            idx = rng.choice(np.arange(6, 10), size=int(m[i]), replace=False)
            x[i, idx] += rng.integers(20, 65, size=idx.shape[0])
    return clip_strategies(x)


def gen_ladder_diff(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    阶梯差值型：
    包含 110/120/130/140/150... 这类想法，但不会全局极端复制，
    而是只在若干关键堆上布置“挡板”。
    """
    base = rng.integers(0, 6, size=(n, K), dtype=np.int32)
    x = base.copy()
    for i in range(n):
        start = int(rng.integers(4, 8))
        cnt = int(rng.integers(2, 5))
        idx = np.arange(start, min(K, start + cnt))
        first = int(rng.choice(np.array([90, 100, 110, 120], dtype=np.int32)))
        step = int(rng.choice(np.array([10, 10, 10, 20], dtype=np.int32)))
        x[i, idx] = first + step * np.arange(idx.shape[0])
        if rng.random() < 0.35:
            j = int(rng.integers(0, 4))
            x[i, j] = int(rng.integers(0, 4))
    return clip_strategies(x)


def gen_over100_controlled(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    总和 105~170 左右的中度超预算型：
    不是瞎堆总和，而是让“可能赢下来的堆”的差值受控。
    这类策略会更敢拿关键堆，但不过分极端。
    """
    alpha = np.array([0.7, 0.75, 0.85, 0.95, 1.0, 1.1, 1.25, 1.55, 1.8, 2.0], dtype=np.float64)
    p = rng.dirichlet(alpha, size=n)
    totals = rng.integers(105, 171, size=n)
    x = np.floor(p * totals[:, None]).astype(np.int32)
    rem = totals - x.sum(axis=1)
    for i in range(n):
        if rem[i] > 0:
            idx = rng.choice(K, size=int(rem[i]), replace=True, p=alpha / alpha.sum())
            np.add.at(x[i], idx, 1)
        # 压低低价值堆的浪费
        x[i, :3] = np.minimum(x[i, :3], rng.integers(0, 12, size=3))
    return clip_strategies(x)


def gen_spikes(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    少数尖峰：
    用来碰撞那些“默认认为没人会在某堆放特别大”的对手。
    但这类人群通常应该更少，否则很容易自己先爆。
    """
    x = rng.integers(0, 5, size=(n, K), dtype=np.int32)
    for i in range(n):
        hot = int(rng.integers(1, 4))
        idx = rng.choice(K, size=hot, replace=False)
        vals = rng.choice(np.array([40, 50, 60, 70, 85, 100, 110, 120, 140, 160, 180], dtype=np.int32), size=hot)
        x[i, idx] = vals
    return clip_strategies(x)


def gen_manual_like(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    靠近常见人类手工思路：
    - 前面很小，后面逐步增大
    - 或者若干组均衡块
    这类很值得覆盖，因为真人提交里经常出现。
    """
    x = np.zeros((n, K), dtype=np.int32)
    modes = rng.integers(0, 3, size=n)
    for i in range(n):
        if modes[i] == 0:
            x[i, :4] = rng.integers(0, 6, size=4)
            x[i, 4:7] = rng.integers(6, 16, size=3)
            x[i, 7:] = rng.integers(15, 36, size=3)
        elif modes[i] == 1:
            x[i, :5] = rng.integers(0, 10, size=5)
            x[i, 5:] = rng.integers(10, 26, size=5)
        else:
            ladder = np.array([0, 1, 2, 4, 7, 11, 16, 22, 29, 37], dtype=np.int32)
            noise = rng.integers(-3, 4, size=K)
            x[i] = ladder + noise
    return clip_strategies(x)


def clip_strategy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.int16)
    np.clip(v, 0, MAX_BID, out=v)
    return v


def random_strategy_general(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 101, size=K, dtype=np.int16)


def random_strategy_focus_tail(rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(K, dtype=np.int16)
    m = int(rng.integers(2, 5))
    idx = rng.choice(np.arange(5, 10), size=m, replace=False)
    x[idx] = rng.integers(15, 101, size=m, dtype=np.int16)
    small_mask = (x == 0)
    x[small_mask] = rng.integers(0, 8, size=int(small_mask.sum()), dtype=np.int16)
    return x

def random_strategy_ladder(rng: np.random.Generator) -> np.ndarray:
    base = rng.integers(0, 9, size=K)
    slope = int(rng.integers(2, 9))
    x = base + slope * np.arange(K)
    noise = rng.integers(-4, 5, size=K)
    return clip_strategy((x + noise).astype(np.int16))


def random_strategy_blocks(rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(K, dtype=np.int16)
    cut = int(rng.integers(3, 8))
    x[:cut] = rng.integers(0, 12, size=cut, dtype=np.int16)
    x[cut:] = rng.integers(10, 70, size=K - cut, dtype=np.int16)
    if rng.random() < 0.35:
        hot = int(rng.integers(0, K))
        x[hot] = int(rng.integers(80, 181))
    return clip_strategy(x)


def random_strategy_spikes(rng: np.random.Generator) -> np.ndarray:
    x = rng.integers(0, 6, size=K, dtype=np.int16)
    m = int(rng.integers(1, 4))
    idx = rng.choice(K, size=m, replace=False)
    x[idx] = rng.integers(25, 201, size=m, dtype=np.int16)
    return x

def random_strategy_mixed(rng: np.random.Generator, n: int) -> np.ndarray:
    p = rng.random()
    if p < 0.22:
        return random_strategy_general(rng)
    if p < 0.47:
        return random_strategy_focus_tail(rng)
    if p < 0.67:
        return random_strategy_ladder(rng)
    if p < 0.85:
        return random_strategy_blocks(rng)
    return random_strategy_spikes(rng)

def generate_batch_mixed(rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, dict]:
    """按固定配比批量生成，避免逐条 random() 分支带来的开销。"""
    spec = [
        ("safe_under100", 0.16, gen_safe_under100),
        ("exact100", 0.14, gen_exact100),
        ("zero_trap", 0.12, gen_zero_trap),
        ("tail_focus", 0.11, gen_tail_focus),
        ("ladder_diff", 0.05, gen_ladder_diff),
        ("over100", 0.06, gen_over100_controlled),
        ("spikes", 0.03, gen_spikes),
        ("manual_like", 0.03, gen_manual_like),
        ("random", 0.3, random_strategy_mixed)
    ]

    counts = {}
    remain = batch_size
    arrs = []
    for idx, (name, ratio, fn) in enumerate(spec):
        if idx == len(spec) - 1:
            cnt = remain
        else:
            cnt = int(batch_size * ratio)
            remain -= cnt
        counts[name] = cnt
        if cnt > 0:
            arrs.append(fn(rng, cnt))
    out = np.vstack(arrs) if arrs else np.empty((0, K), dtype=DTYPE)
    perm = rng.permutation(out.shape[0])
    return out[perm], counts


# -----------------------------
# Unique generation
# -----------------------------

def generate_unique_strategies(N: int, seed: int, log_every: int, batch_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t0 = time.time()

    out = np.empty((N, K), dtype=DTYPE)
    seen = set()
    inserted = 0
    trials = 0
    family_total = {}

    additional_strategies = getattr(judge, "strategies", [])
    for v in additional_strategies:
        vv = clip_strategies(np.asarray(v, dtype=DTYPE).reshape(1, K))[0]
        k = key_of(vv)
        if k in seen:
            continue
        seen.add(k)
        if inserted < N:
            out[inserted] = vv
            inserted += 1

    while inserted < N:
        batch, counts = generate_batch_mixed(rng, min(batch_size, (N - inserted) * 2))
        for name, cnt in counts.items():
            family_total[name] = family_total.get(name, 0) + cnt

        batch = unique_rows_preserve_first(batch)
        trials += batch.shape[0]

        for row in batch:
            k = key_of(row)
            if k in seen:
                continue
            seen.add(k)
            out[inserted] = row
            inserted += 1
            if inserted >= N:
                break

        if log_every and (inserted % log_every == 0 or inserted == N):
            dt = time.time() - t0
            dup_rate = 1.0 - inserted / max(trials + len(additional_strategies), 1)
            print(f"[gen] {inserted:,}/{N:,} unique | batch={batch_size:,} | approx_dup={dup_rate:.4f} | {dt:.1f}s")

    dt = time.time() - t0
    print(f"[gen] DONE {inserted:,}/{N:,} unique | {dt:.1f}s")
    print("[gen] family mix used:")
    for name in ["safe_under100", "exact100", "zero_trap", "tail_focus", "ladder_diff", "over100", "spikes", "manual_like", "random"]:
        print(f"  - {name:<13} {family_total.get(name, 0):,}")
    return out


# -----------------------------
# Match + tournament
# -----------------------------
if NUMBA_OK:
    @njit(cache=True)
    def match_result(a, b):
        return judge.calculate_match_result(a, b)

    @njit(cache=True)
    def run_one_round(strats, order, wins):
        n = order.shape[0]
        for j in range(0, n - 1, 2):
            i1 = order[j]
            i2 = order[j + 1]
            r = match_result(strats[i1], strats[i2])
            if r == 1:
                wins[i1] += 1
            elif r == -1:
                wins[i2] += 1

    @njit(cache=True)
    def full_round_robin(strats, wins):
        n = strats.shape[0]
        for i in range(n):
            ai = strats[i]
            for j in range(i + 1, n):
                r = match_result(ai, strats[j])
                if r == 1:
                    wins[i] += 1
                elif r == -1:
                    wins[j] += 1
else:
    def match_result(a, b):
        return judge.calculate_match_result(a, b)

    def run_one_round(strats, order, wins):
        n = len(order)
        for j in range(0, n - 1, 2):
            i1 = order[j]
            i2 = order[j + 1]
            r = match_result(strats[i1], strats[i2])
            if r == 1:
                wins[i1] += 1
            elif r == -1:
                wins[i2] += 1

    def full_round_robin(strats, wins):
        n = strats.shape[0]
        for i in range(n):
            ai = strats[i]
            for j in range(i + 1, n):
                r = match_result(ai, strats[j])
                if r == 1:
                    wins[i] += 1
                elif r == -1:
                    wins[j] += 1


def evolve_tournament(strats: np.ndarray, rounds: int, seed: int, log_every_rounds: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = strats.shape[0]
    wins = np.zeros(n, dtype=np.int32)
    order = np.arange(n, dtype=np.int32)

    t0 = time.time()
    for r in range(1, rounds + 1):
        rng.shuffle(order)
        run_one_round(strats, order, wins)
        if log_every_rounds and (r == 1 or r % log_every_rounds == 0):
            dt = time.time() - t0
            print(f"[tournament] round {r}/{rounds} | best={wins.max()} | mean={wins.mean():.3f} | {dt:.1f}s")

    dt = time.time() - t0
    print(f"[tournament] DONE rounds={rounds} | {dt:.1f}s | best={wins.max()} | mean={wins.mean():.3f}")
    return wins


def evolve_full(strats: np.ndarray) -> np.ndarray:
    n = strats.shape[0]
    wins = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    full_round_robin(strats, wins)
    dt = time.time() - t0
    print(f"[full] DONE n={n:,} | games={n*(n-1)//2:,} | {dt:.1f}s | best={wins.max()} | mean={wins.mean():.3f}")
    return wins


def top_n_indices(wins: np.ndarray, n: int) -> np.ndarray:
    if n >= wins.shape[0]:
        return np.argsort(-wins)
    idx = np.argpartition(-wins, n - 1)[:n]
    return idx[np.argsort(-wins[idx])]


def save_stage(out_dir: str, name: str, strats: np.ndarray, wins: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(path, strategies=strats.astype(DTYPE), wins=wins.astype(np.int32))
    print(f"[save] {name} -> {path}  (n={strats.shape[0]:,})")


def stage_run(in_strats, out_size, mode, rounds, seed, log_every_rounds, out_dir, stage_name):
    print(f"\n[stage] {stage_name}: in={in_strats.shape[0]:,} -> out={out_size:,} | mode={mode} | rounds={rounds if mode=='tournament' else '-'}")
    if mode == "full":
        wins = evolve_full(in_strats)
    elif mode == "tournament":
        wins = evolve_tournament(in_strats, rounds=rounds, seed=seed, log_every_rounds=log_every_rounds)
    else:
        raise ValueError("mode must be 'tournament' or 'full'")

    idx = top_n_indices(wins, out_size)
    out_strats = in_strats[idx].copy()
    out_wins = wins[idx].copy()
    save_stage(out_dir, stage_name, out_strats, out_wins)
    return out_strats


# -----------------------------
# Args / main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="第11回合沙堆养蛊多级筛选（智能随机生成版）")

    # Generation
    p.add_argument("--seed_gen", type=int, default=20260313)
    p.add_argument("--n0", type=int, default=5000000,
                   help="initial unique strategies count")
    p.add_argument("--gen_log_every", type=int, default=50000,
                   help="generation log frequency")
    p.add_argument("--batch_size", type=int, default=8192,
                   help="batched random generation size")

    # Output directory
    p.add_argument("--out_dir", type=str, default=getattr(judge, "name", "results_round11"))

    # Stage sizes
    p.add_argument("--n1", type=int, default=500000)
    p.add_argument("--n2", type=int, default=10000)
    p.add_argument("--n3", type=int, default=1000)
    p.add_argument("--n4", type=int, default=100)

    # Modes per stage
    p.add_argument("--mode1", type=str, default="tournament", choices=["tournament", "full"])
    p.add_argument("--rounds1", type=int, default=1000)
    p.add_argument("--seed1", type=int, default=101)

    p.add_argument("--mode2", type=str, default="tournament", choices=["tournament", "full"])
    p.add_argument("--rounds2", type=int, default=10000)
    p.add_argument("--seed2", type=int, default=202)

    p.add_argument("--mode3", type=str, default="full", choices=["tournament", "full"])
    p.add_argument("--rounds3", type=int, default=120)
    p.add_argument("--seed3", type=int, default=303)

    p.add_argument("--mode4", type=str, default="full", choices=["tournament", "full"])
    p.add_argument("--rounds4", type=int, default=300)
    p.add_argument("--seed4", type=int, default=404)

    # Logging
    p.add_argument("--log_every_rounds", type=int, default=10,
                   help="tournament log every X rounds")

    return p.parse_args()


def main():
    args = parse_args()
    print(f"NUMBA_OK={NUMBA_OK}")
    print(f"Judge module: {judge.__name__}")
    print(f"Output dir: {args.out_dir}")

    print("\n== Generate unique strategies ==")
    strats0 = generate_unique_strategies(
        args.n0,
        seed=args.seed_gen,
        log_every=args.gen_log_every,
        batch_size=args.batch_size,
    )

    strats1 = stage_run(
        in_strats=strats0,
        out_size=args.n1,
        mode=args.mode1,
        rounds=args.rounds1,
        seed=args.seed1,
        log_every_rounds=args.log_every_rounds,
        out_dir=args.out_dir,
        stage_name="top100k",
    )

    strats2 = stage_run(
        in_strats=strats1,
        out_size=args.n2,
        mode=args.mode2,
        rounds=args.rounds2,
        seed=args.seed2,
        log_every_rounds=args.log_every_rounds,
        out_dir=args.out_dir,
        stage_name="top10k",
    )

    strats3 = stage_run(
        in_strats=strats2,
        out_size=args.n3,
        mode=args.mode3,
        rounds=args.rounds3,
        seed=args.seed3,
        log_every_rounds=args.log_every_rounds,
        out_dir=args.out_dir,
        stage_name="top1k",
    )

    _ = stage_run(
        in_strats=strats3,
        out_size=args.n4,
        mode=args.mode4,
        rounds=args.rounds4,
        seed=args.seed4,
        log_every_rounds=args.log_every_rounds,
        out_dir=args.out_dir,
        stage_name="top100",
    )

    print("\n== Preview top100 first 20 ==")
    top100_path = os.path.join(args.out_dir, "top100.npz")
    data = np.load(top100_path)
    s = data["strategies"]
    w = data["wins"]
    for i in range(min(20, s.shape[0])):
        print(f"{i+1:>2d}  win={int(w[i])}  sum={int(s[i].sum())}  {s[i].tolist()}")

    print("\nDONE.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Multi-stage evolution for round 10.

Round 10 validity rule:
- 10 piles are split into 5 contiguous groups by placing 4 boards.
- In each group, max(pile) - min(pile) <= 1.
- A strategy is valid iff there exists at least one such partition.

This file only handles generation / evolution under the new legality constraint.
Match judging still delegates to judge.calculate_match_result(a, b).
"""

import os
import time
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
GROUPS = 5
MIN_GROUP_LEN = 1
BASE = 101
BASE_POW = [BASE ** i for i in range(K)]


def encode_vec_int(v: np.ndarray) -> int:
    s = 0
    for i in range(K):
        s += int(v[i]) * BASE_POW[i]
    return s


def valid_group_segment(v: np.ndarray, l: int, r: int) -> bool:
    seg = v[l:r]
    return int(seg.max()) - int(seg.min()) <= 1


def exists_round10_partition(v: np.ndarray) -> bool:
    """Whether v can be split into 5 contiguous non-empty groups with range <= 1 in each group."""
    ok = np.zeros((K + 1, GROUPS + 1), dtype=np.bool_)
    ok[0, 0] = True
    for i in range(1, K + 1):
        for g in range(1, GROUPS + 1):
            for j in range(g - 1, i):
                if ok[j, g - 1] and valid_group_segment(v, j, i):
                    ok[i, g] = True
                    break
    return bool(ok[K, GROUPS])


def is_legal_round10(v: np.ndarray) -> bool:
    """Round10 submission legality: non-negative ints, sum=100, and has a valid 5-group partition."""
    vv = np.asarray(v)
    if vv.shape[0] != K:
        return False
    if (vv < 0).any():
        return False
    if int(vv.sum()) != TOTAL:
        return False
    return exists_round10_partition(vv)


def sample_partition_lengths(rng: np.random.Generator) -> np.ndarray:
    """Random positive lengths summing to 10, exactly 5 groups."""
    cuts = np.sort(rng.choice(np.arange(1, K), size=GROUPS - 1, replace=False))
    pts = np.concatenate(([0], cuts, [K]))
    return np.diff(pts).astype(np.int16)


def random_valid_round10_strategy(rng: np.random.Generator, max_tries: int = 2000) -> np.ndarray:
    """
    Direct constructive generator.
    For each of 5 contiguous groups, values are either base or base+1.
    This guarantees each group's max-min <= 1.
    """
    for _ in range(max_tries):
        lengths = sample_partition_lengths(rng)
        group_base = rng.integers(0, 21, size=GROUPS)  # loose range; total fixed later by +1 counts
        x = np.empty(K, dtype=np.int16)
        pos = 0
        for gi in range(GROUPS):
            m = int(lengths[gi])
            b = int(group_base[gi])
            add1 = int(rng.integers(0, m + 1))
            seg = np.full(m, b, dtype=np.int16)
            if add1 > 0:
                idx = rng.choice(m, size=add1, replace=False)
                seg[idx] += 1
            x[pos:pos + m] = seg
            pos += m

        s = int(x.sum())
        diff = TOTAL - s
        if diff == 0 and is_legal_round10(x):
            return x

        # Adjust by moving one coin at a time inside random groups while keeping segment range <= 1.
        y = x.copy()
        for _adj in range(400):
            s2 = int(y.sum())
            if is_legal_round10(y):
                return y

            # pick a partition induced by lengths
            pos = 0
            changed = False
            for gi in rng.permutation(GROUPS):
                l = int(np.sum(lengths[:gi]))
                r = l + int(lengths[gi])
                seg = y[l:r]
                mn = int(seg.min())
                mx = int(seg.max())
                if s2 < TOTAL:
                    # can increase any min element if range stays <=1
                    cand = np.where(seg == mn)[0]
                    if cand.size > 0 and mx - mn <= 1:
                        k = int(cand[rng.integers(0, cand.size)])
                        seg[k] += 1
                        changed = True
                        break
                else:
                    # can decrease any max element if range stays <=1
                    cand = np.where(seg == mx)[0]
                    if cand.size > 0 and mx - mn <= 1:
                        k = int(cand[rng.integers(0, cand.size)])
                        # IMPORTANT: avoid generating negative coin counts
                        if seg[k] > 0:
                            seg[k] -= 1
                            changed = True
                            break
            if not changed:
                break
        if is_legal_round10(y):
            return y
    raise RuntimeError("failed to generate a valid round10 strategy")


def repair_to_round10(x: np.ndarray, rng: np.random.Generator | None = None, max_tries: int = 200) -> np.ndarray:
    """
    Soft repair:
    1) fix sum to 100
    2) if still illegal, fall back to a fresh valid sample
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.asarray(x, dtype=np.int16).copy()
    y[y < 0] = 0

    s = int(y.sum())
    if s < TOTAL:
        y[K - 1] += (TOTAL - s)
    elif s > TOTAL:
        extra = s - TOTAL
        for i in range(K - 1, -1, -1):
            take = min(int(y[i]), extra)
            y[i] -= take
            extra -= take
            if extra == 0:
                break

    if is_legal_round10(y):
        return y

    # local random repair attempts
    for _ in range(max_tries):
        z = y.copy()
        i = int(rng.integers(0, K))
        j = int(rng.integers(0, K))
        while j == i:
            j = int(rng.integers(0, K))
        if z[i] > 0:
            z[i] -= 1
            z[j] += 1
        if is_legal_round10(z):
            return z

    return random_valid_round10_strategy(rng)


def random_composition_sum100(rng: np.random.Generator) -> np.ndarray:
    return random_valid_round10_strategy(rng)


def generate_unique_strategies(N: int, seed: int, log_every: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((N, K), dtype=np.int16)

    additional_strategies = []
    if hasattr(judge, "strategies"):
        for v in judge.strategies:
            vv = np.asarray(v, dtype=np.int16)
            vv = repair_to_round10(vv, rng)
            if is_legal_round10(vv):
                additional_strategies.append(vv)

    seen = set()
    i = 0
    for v in additional_strategies:
        if i >= N:
            break
        code = encode_vec_int(v)
        if code not in seen:
            out[i] = v
            seen.add(code)
            i += 1

    trials = 0
    t0 = time.time()
    while i < N:
        v = random_composition_sum100(rng)
        code = encode_vec_int(v)
        trials += 1
        if code in seen:
            continue
        seen.add(code)
        out[i] = v
        i += 1
        if log_every and i % log_every == 0:
            dt = time.time() - t0
            dup_rate = 1.0 - i / max(trials + len(additional_strategies), 1)
            print(f"[gen] {i:,}/{N:,} valid unique | trials={trials:,} | dup={dup_rate:.4f} | {dt:.1f}s")

    dt = time.time() - t0
    print(f"[gen] DONE {N:,} valid unique | trials={trials:,} | {dt:.1f}s")
    return out


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
            for j in range(i + 1, n):
                r = match_result(strats[i], strats[j])
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
        if log_every_rounds and (r % log_every_rounds == 0):
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
        idx = np.argsort(-wins)
        return idx
    idx = np.argpartition(-wins, n - 1)[:n]
    idx = idx[np.argsort(-wins[idx])]
    return idx


def save_stage(out_dir: str, name: str, strats: np.ndarray, wins: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(path, strategies=strats.astype(np.int16), wins=wins.astype(np.int32))
    print(f"[save] {name} -> {path}  (n={strats.shape[0]:,})")

def stage_run(
    in_strats: np.ndarray,
    out_size: int,
    mode: str,
    rounds: int,
    seed: int,
    log_every_rounds: int,
    out_dir: str,
    stage_name: str,
):
    print(f"\n[stage] {stage_name}: in={in_strats.shape[0]:,} -> out={out_size:,} | mode={mode} | rounds={rounds if mode=='tournament' else '-'}")
    if mode == "full":
        wins = evolve_full(in_strats)
    elif mode == "tournament":
        wins = evolve_tournament(
            in_strats, rounds=rounds, seed=seed, log_every_rounds=log_every_rounds)
    else:
        raise ValueError("mode must be 'tournament' or 'full'")

    idx = top_n_indices(wins, out_size)
    out_strats = in_strats[idx].copy()
    out_wins = wins[idx].copy()

    save_stage(out_dir, stage_name, out_strats, out_wins)
    return out_strats


def parse_args():
    p = argparse.ArgumentParser(description="沙堆养蛊多级筛选")

    # Generation
    p.add_argument("--seed_gen", type=int, default=20260307)
    p.add_argument("--n0", type=int, default=3_000_000,
                   help="initial unique strategies count (default 1,000,000)")
    p.add_argument("--gen_log_every", type=int, default=50_000,
                   help="generation log frequency")

    # Output directory
    p.add_argument("--out_dir", type=str, default=judge.name)

    # Stage sizes (fixed per your request but still editable)
    p.add_argument("--n1", type=int, default=300_000)
    p.add_argument("--n2", type=int, default=10_000)
    p.add_argument("--n3", type=int, default=1_000)
    p.add_argument("--n4", type=int, default=100)

    # Modes per stage
    # stage1: 1e6 -> 1e5
    p.add_argument("--mode1", type=str, default="tournament",
                   choices=["tournament", "full"])
    p.add_argument("--rounds1", type=int, default=1000)
    p.add_argument("--seed1", type=int, default=101)

    # stage2: 1e5 -> 1e4
    p.add_argument("--mode2", type=str, default="tournament",
                   choices=["tournament", "full"])
    p.add_argument("--rounds2", type=int, default=10000)
    p.add_argument("--seed2", type=int, default=202)

    # stage3: 1e4 -> 1e3 (这里常用 full)
    p.add_argument("--mode3", type=str, default="full",
                   choices=["tournament", "full"])
    p.add_argument("--rounds3", type=int, default=120)
    p.add_argument("--seed3", type=int, default=303)

    # stage4: 1e3 -> 100 (这里也可以 full / 或 tournament 很多轮)
    p.add_argument("--mode4", type=str, default="full",
                   choices=["tournament", "full"])
    p.add_argument("--rounds4", type=int, default=300)
    p.add_argument("--seed4", type=int, default=404)

    # Logging
    p.add_argument("--log_every_rounds", type=int, default=10,
                   help="tournament log every X rounds")

    return p.parse_args()


def main():
    args = parse_args()
    print(f"NUMBA_OK={NUMBA_OK}")
    print(f"Output dir: {args.out_dir}")

    # 0) generate unique 1,000,000
    print("\n== Generate unique strategies ==")
    strats0 = generate_unique_strategies(
        args.n0, seed=args.seed_gen, log_every=args.gen_log_every)
    # 可选：保存初始100万（你没要求保存，就不保存；想要可以自己加）

    # 1) 1e6 -> 1e5
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

    # 2) 1e5 -> 1e4
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

    # 3) 1e4 -> 1e3
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

    # 4) 1e3 -> 100
    strats4 = stage_run(
        in_strats=strats3,
        out_size=args.n4,
        mode=args.mode4,
        rounds=args.rounds4,
        seed=args.seed4,
        log_every_rounds=args.log_every_rounds,
        out_dir=args.out_dir,
        stage_name="top100",
    )

    # 额外：打印 top100 前 20 预览（不刷屏）
    print("\n== Preview top100 first 20 ==")
    # 载入保存文件，确保一致
    top100_path = os.path.join(args.out_dir, "top100.npz")
    data = np.load(top100_path)
    s = data["strategies"]
    w = data["wins"]
    for i in range(min(20, s.shape[0])):
        print(f"{i+1:>2d}  win={int(w[i])}  {s[i].tolist()}")

    print("\nDONE.")


if __name__ == "__main__":
    main()

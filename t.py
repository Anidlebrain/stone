# -*- coding: utf-8 -*-
"""
Multi-stage evolution:
1,000,000 -> 100,000 -> 10,000 -> 1,000 -> 100

Features:
- Generate 1,000,000 UNIQUE strategies (10 ints sum to 100)
- Stage filtering with either:
  - tournament: random pairing for N rounds (each win +1)
  - full: all-pairs round robin (only practical for <= ~12k)
- Save stage results to local .npz
- Controlled logging (not too spammy)
"""

import os
import time
import math
import argparse
import numpy as np
import judge

# ---- Optional speedup ----
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

TOTAL = 100
K = 10
BOOST = 1.5
CLOSE_LOSS = 5.0

# ---- Unique encoding (collision-free) ----
# Each component in [0,100], encode base 101 into a Python int.
BASE = 101
BASE_POW = [BASE ** i for i in range(K)]


def encode_vec_int(v: np.ndarray) -> int:
    s = 0
    for i in range(K):
        s += int(v[i]) * BASE_POW[i]
    return s


def random_composition_sum100(rng: np.random.Generator) -> np.ndarray:
    """
    Random 10-dim integer vector sum=100 via Dirichlet + Multinomial.
    Not uniform over compositions, but good for evolutionary search / 养蛊.
    """
    p = rng.dirichlet(np.ones(K))
    v = rng.multinomial(TOTAL, p)
    return v.astype(np.int16)


def generate_unique_strategies(N: int, seed: int, log_every: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total_n = N
    out = np.empty((total_n, K), dtype=np.int16)

    additional_strategies = judge.strategies

    additional_count = len(additional_strategies)
    if additional_count > total_n:
        raise ValueError(f'judge.strategies has {additional_count} items, larger than n0={total_n}')
    out[:additional_count] = additional_strategies

    seen = set()
    for v in additional_strategies:
        code = encode_vec_int(v)
        seen.add(code)

    i = additional_count
    trials = 0
    t0 = time.time()

    while i < total_n:
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
            dup_rate = 1.0 - i / max(trials, 1)
            print(
                f"[gen] {i:,}/{total_n:,} unique | trials={trials:,} | dup={dup_rate:.4f} | {dt:.1f}s")

    dt = time.time() - t0
    dup_rate = 1.0 - total_n / max(trials, 1)
    print(
        f"[gen] DONE {total_n:,} unique | trials={trials:,} | dup={dup_rate:.4f} | {dt:.1f}s")
    return out


# -----------------------------
# Match + tournament (fast path via numba)
# -----------------------------
if NUMBA_OK:
    @njit(cache=True)
    def match_result(a, b):
        """
        Return 1 if a wins, -1 if b wins, 0 if tie.
        a, b: int16[10] - representing the 10 sandpile values for each player.
        """
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
        """
        All pairs i<j.
        Winner gets +1. Tie +0.
        """
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
        """
        Return 1 if a wins, -1 if b wins, 0 if tie.
        a, b: int16[10] - representing the 10 sandpile values for each player.
        """

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
            print(
                f"[tournament] round {r}/{rounds} | best={wins.max()} | mean={wins.mean():.3f} | {dt:.1f}s")

    dt = time.time() - t0
    print(
        f"[tournament] DONE rounds={rounds} | {dt:.1f}s | best={wins.max()} | mean={wins.mean():.3f}")
    return wins


def evolve_full(strats: np.ndarray) -> np.ndarray:
    n = strats.shape[0]
    wins = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    full_round_robin(strats, wins)
    dt = time.time() - t0
    # 这里的 best/mean 有意义（总对局数固定），给个简短日志
    print(
        f"[full] DONE n={n:,} | games={n*(n-1)//2:,} | {dt:.1f}s | best={wins.max()} | mean={wins.mean():.3f}")
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
    # 保存: strategies(int16), wins(int32)
    np.savez_compressed(path, strategies=strats.astype(
        np.int16), wins=wins.astype(np.int32))
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
    p.add_argument("--n0", type=int, default=5_000_000,
                   help="initial unique strategies count (default 1,000,000)")
    p.add_argument("--gen_log_every", type=int, default=50_000,
                   help="generation log frequency")

    # Output directory
    p.add_argument("--out_dir", type=str, default=judge.name)

    # Stage sizes (fixed per your request but still editable)
    p.add_argument("--n1", type=int, default=500_000)
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
    print(f"Round: {judge.round_no}")
    print(f"Output dir: {args.out_dir}")
    print(f"min_values from judge: {judge.min_values.tolist()}")
    print(f"must_beat count from judge: {int(judge.must_beat.shape[0])}")

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

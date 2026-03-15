# -*- coding: utf-8 -*-
"""
Multi-stage evolution for round12.

Features:
- Generate UNIQUE strategies (10 ints, each >=0, sum <= 100)
- Stage filtering with either:
  - tournament: random pairing for N rounds (each win +1)
  - full: all-pairs round robin
- Save stage results to local .npz
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
BASE = 101
BASE_POW = [BASE ** i for i in range(K)]


def encode_vec_int(v: np.ndarray) -> int:
    s = 0
    for i in range(K):
        s += int(v[i]) * BASE_POW[i]
    return s


def random_strategy_sum_le_100(rng: np.random.Generator) -> np.ndarray:
    """Random 10-dim integer vector with sum <= 100."""
    used = int(rng.integers(0, TOTAL + 1))
    p = rng.dirichlet(np.ones(K))
    v = rng.multinomial(used, p)
    return v.astype(np.int16)


def generate_unique_strategies(N: int, seed: int, log_every: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((N, K), dtype=np.int16)

    additional_strategies = [np.asarray(
        v, dtype=np.int16) for v in judge.strategies]
    additional_count = min(len(additional_strategies), N)

    seen = set()
    i = 0
    for v in additional_strategies[:additional_count]:
        code = encode_vec_int(v)
        if code in seen:
            continue
        seen.add(code)
        out[i] = v
        i += 1

    trials = 0
    t0 = time.time()
    while i < N:
        v = random_strategy_sum_le_100(rng)
        code = encode_vec_int(v)
        trials += 1
        if code in seen:
            continue
        seen.add(code)
        out[i] = v
        i += 1

        if log_every and i % log_every == 0:
            dt = time.time() - t0
            dup_rate = 1.0 - i / max(trials + additional_count, 1)
            print(
                f"[gen] {i:,}/{N:,} unique | trials={trials:,} | dup={dup_rate:.4f} | {dt:.1f}s")

    dt = time.time() - t0
    dup_rate = 1.0 - N / max(trials + additional_count, 1)
    print(
        f"[gen] DONE {N:,} unique | trials={trials:,} | dup={dup_rate:.4f} | {dt:.1f}s")
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
    np.savez_compressed(path, strategies=strats.astype(
        np.int16), wins=wins.astype(np.int32))
    print(f"[save] {name} -> {path}  (n={strats.shape[0]:,})")


def stage_run(in_strats, out_size, mode, rounds, seed, log_every_rounds, out_dir, stage_name):
    print(f"\n[stage] {stage_name}: in={in_strats.shape[0]:,} -> out={out_size:,} | mode={mode} | rounds={rounds if mode == 'tournament' else '-'}")
    if mode == 'full':
        wins = evolve_full(in_strats)
    elif mode == 'tournament':
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
    p = argparse.ArgumentParser(description='沙堆养蛊多级筛选（第12回合）')
    p.add_argument('--seed_gen', type=int, default=202603142106)
    p.add_argument('--n0', type=int, default=5_000_000)
    p.add_argument('--gen_log_every', type=int, default=50_000)
    p.add_argument('--out_dir', type=str, default=judge.name)
    p.add_argument('--n1', type=int, default=500_000)
    p.add_argument('--n2', type=int, default=10_000)
    p.add_argument('--n3', type=int, default=1_000)
    p.add_argument('--n4', type=int, default=100)
    p.add_argument('--mode1', type=str, default='tournament',
                   choices=['tournament', 'full'])
    p.add_argument('--rounds1', type=int, default=1000)
    p.add_argument('--seed1', type=int, default=101)
    p.add_argument('--mode2', type=str, default='tournament',
                   choices=['tournament', 'full'])
    p.add_argument('--rounds2', type=int, default=10000)
    p.add_argument('--seed2', type=int, default=202)
    p.add_argument('--mode3', type=str, default='full',
                   choices=['tournament', 'full'])
    p.add_argument('--rounds3', type=int, default=120)
    p.add_argument('--seed3', type=int, default=303)
    p.add_argument('--mode4', type=str, default='full',
                   choices=['tournament', 'full'])
    p.add_argument('--rounds4', type=int, default=300)
    p.add_argument('--seed4', type=int, default=404)
    p.add_argument('--log_every_rounds', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"NUMBA_OK={NUMBA_OK}")
    print(f"Output dir: {args.out_dir}")

    print("\n== Generate unique strategies ==")
    strats0 = generate_unique_strategies(
        args.n0, seed=args.seed_gen, log_every=args.gen_log_every)

    strats1 = stage_run(strats0, args.n1, args.mode1, args.rounds1,
                        args.seed1, args.log_every_rounds, args.out_dir, 'top100k')
    strats2 = stage_run(strats1, args.n2, args.mode2, args.rounds2,
                        args.seed2, args.log_every_rounds, args.out_dir, 'top10k')
    strats3 = stage_run(strats2, args.n3, args.mode3, args.rounds3,
                        args.seed3, args.log_every_rounds, args.out_dir, 'top1k')
    stage_run(strats3, args.n4, args.mode4, args.rounds4, args.seed4,
              args.log_every_rounds, args.out_dir, 'top100')

    print("\n== Preview top100 first 20 ==")
    top100_path = os.path.join(args.out_dir, 'top100.npz')
    data = np.load(top100_path)
    s = data['strategies']
    w = data['wins']
    for i in range(min(20, s.shape[0])):
        print(
            f"{i+1:>2d}  win={int(w[i])}  sum={int(s[i].sum())}  {s[i].tolist()}")

    print("\nDONE.")


if __name__ == '__main__':
    main()

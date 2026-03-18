# -*- coding: utf-8 -*-
"""
Round 15 multi-stage evolution.
Strategy format:
[buyA, buyB, buyC, insurance, p1, ..., p10]
"""

import os
import time
import argparse
import numpy as np
import judge15 as judge

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

TOTAL = judge.START_COINS
DIM = judge.STRATEGY_LEN
BASE = 101
BASE_POW = [BASE ** i for i in range(DIM)]


def encode_vec_int(v: np.ndarray) -> int:
    s = 0
    for i in range(DIM):
        s += int(v[i]) * BASE_POW[i]
    return s


def random_strategy(rng: np.random.Generator) -> np.ndarray:
    return judge.sample_random_strategy(rng, judge.min_values)


def generate_unique_strategies(n: int, seed: int, log_every: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n, DIM), dtype=np.int16)

    additional = judge.strategies
    if len(additional) > n:
        raise ValueError(f'judge.strategies has {len(additional)} items, larger than n0={n}')

    seen = set()
    for i, v in enumerate(additional):
        out[i] = v
        seen.add(encode_vec_int(v))

    i = len(additional)
    trials = 0
    t0 = time.time()
    while i < n:
        v = random_strategy(rng)
        code = encode_vec_int(v)
        trials += 1
        if code in seen:
            continue
        seen.add(code)
        out[i] = v
        i += 1
        if log_every and i % log_every == 0:
            dt = time.time() - t0
            dup_rate = 1.0 - i / max(trials + len(additional), 1)
            print(f'[gen] {i:,}/{n:,} unique | trials={trials:,} | dup={dup_rate:.4f} | {dt:.1f}s')

    dt = time.time() - t0
    dup_rate = 1.0 - n / max(trials + len(additional), 1)
    print(f'[gen] DONE {n:,} unique | trials={trials:,} | dup={dup_rate:.4f} | {dt:.1f}s')
    return out


if NUMBA_OK:
    MATCH_FUNC = judge.calculate_match_result_nb

    @njit(cache=True)
    def match_result(a, b):
        return MATCH_FUNC(a, b)

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
        n = order.shape[0]
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
            print(f'[mc] round {r}/{rounds} | elapsed={time.time() - t0:.1f}s')
    return wins


def evolve_full(strats: np.ndarray) -> np.ndarray:
    wins = np.zeros(strats.shape[0], dtype=np.int32)
    full_round_robin(strats, wins)
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
    path = os.path.join(out_dir, f'{name}.npz')
    np.savez_compressed(path, strategies=strats.astype(np.int16), wins=wins.astype(np.int32))
    print(f'[save] {name} -> {path} (n={strats.shape[0]:,})')


def stage_run(in_strats, out_size, mode, rounds, seed, log_every_rounds, out_dir, stage_name):
    print(f'\n[stage] {stage_name}: in={in_strats.shape[0]:,} -> out={out_size:,} | mode={mode} | rounds={rounds if mode == "tournament" else "-"}')
    if mode == 'full':
        wins = evolve_full(in_strats)
    elif mode == 'tournament':
        wins = evolve_tournament(in_strats, rounds=rounds, seed=seed, log_every_rounds=log_every_rounds)
    else:
        raise ValueError("mode must be 'tournament' or 'full'")

    idx = top_n_indices(wins, out_size)
    out_strats = in_strats[idx].copy()
    out_wins = wins[idx].copy()
    save_stage(out_dir, stage_name, out_strats, out_wins)
    return out_strats


def parse_args():
    p = argparse.ArgumentParser(description='第15回合 多级筛选')
    p.add_argument('--seed_gen', type=int, default=202603191430)
    p.add_argument('--n0', type=int, default=5_000_000)
    p.add_argument('--gen_log_every', type=int, default=50_000)
    p.add_argument('--out_dir', type=str, default=judge.name)
    p.add_argument('--n1', type=int, default=500_000)
    p.add_argument('--n2', type=int, default=10_000)
    p.add_argument('--n3', type=int, default=1_000)
    p.add_argument('--n4', type=int, default=100)
    p.add_argument('--mode1', type=str, default='tournament', choices=['tournament', 'full'])
    p.add_argument('--rounds1', type=int, default=1000)
    p.add_argument('--seed1', type=int, default=101)
    p.add_argument('--mode2', type=str, default='tournament', choices=['tournament', 'full'])
    p.add_argument('--rounds2', type=int, default=5000)
    p.add_argument('--seed2', type=int, default=202)
    p.add_argument('--mode3', type=str, default='full', choices=['tournament', 'full'])
    p.add_argument('--rounds3', type=int, default=120)
    p.add_argument('--seed3', type=int, default=303)
    p.add_argument('--mode4', type=str, default='full', choices=['tournament', 'full'])
    p.add_argument('--rounds4', type=int, default=300)
    p.add_argument('--seed4', type=int, default=404)
    p.add_argument('--log_every_rounds', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    print(f'NUMBA_OK={NUMBA_OK}')
    print(f'Round: {judge.round_no}')
    print(f'Output dir: {args.out_dir}')
    print(f'Skill costs: A={judge.SKILL_COST_A}, B={judge.SKILL_COST_B}, C={judge.SKILL_COST_C}')
    print(f'min_values from judge: {judge.min_values.tolist()}')
    print(f'must_beat count from judge: {int(judge.must_beat.shape[0])}')

    print('\n== Generate unique strategies ==')
    strats0 = generate_unique_strategies(args.n0, seed=args.seed_gen, log_every=args.gen_log_every)

    strats1 = stage_run(strats0, args.n1, args.mode1, args.rounds1, args.seed1, args.log_every_rounds, args.out_dir, 'top100k')
    strats2 = stage_run(strats1, args.n2, args.mode2, args.rounds2, args.seed2, args.log_every_rounds, args.out_dir, 'top10k')
    strats3 = stage_run(strats2, args.n3, args.mode3, args.rounds3, args.seed3, args.log_every_rounds, args.out_dir, 'top1k')
    stage_run(strats3, args.n4, args.mode4, args.rounds4, args.seed4, args.log_every_rounds, args.out_dir, 'top100')

    print('\n== Preview top100 first 20 ==')
    data = np.load(os.path.join(args.out_dir, 'top100.npz'))
    s = data['strategies']
    w = data['wins']
    for i in range(min(20, s.shape[0])):
        print(f'{i + 1:>2d} win={int(w[i])} {judge.format_strategy(s[i])}')

    print('\nDONE.')


if __name__ == '__main__':
    main()

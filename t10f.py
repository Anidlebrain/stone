# -*- coding: utf-8 -*-
"""
Multi-stage evolution for round 10.

Faster generation version.

Key idea for round 10 legality:
- 10 piles are split into 5 contiguous groups by placing 4 boards.
- In each group, max(pile) - min(pile) <= 1.
- Therefore, inside a group of length m with total sum s,
  the group must be exactly q/q+1 form where q = s // m and r = s % m.

So instead of:
- randomly making a 10-d vector,
- then repairing sum,
- then testing legality with DP,

we generate legal strategies directly:
1) sample one partition length pattern
2) sample a 5-part composition of 100 as group sums
3) decode each group sum into q/q+1 values

This makes initial unique-pool generation much faster.
"""

import os
import time
import argparse
import itertools
import numpy as np
import judge
from typing import Optional

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

TOTAL = 100
K = 10
GROUPS = 5
BASE = 101
BASE_POW = [BASE ** i for i in range(K)]

# all 126 contiguous 5-group partitions of length 10
PARTITIONS = []
for cuts in itertools.combinations(range(1, K), GROUPS - 1):
    pts = (0,) + cuts + (K,)
    PARTITIONS.append(tuple(pts[i + 1] - pts[i] for i in range(GROUPS)))
PARTITIONS = np.asarray(PARTITIONS, dtype=np.int16)


def encode_vec_int(v: np.ndarray) -> int:
    s = 0
    for i in range(K):
        s += int(v[i]) * BASE_POW[i]
    return s


def valid_group_segment(v: np.ndarray, l: int, r: int) -> bool:
    seg = v[l:r]
    return int(seg.max()) - int(seg.min()) <= 1


def exists_round10_partition(v: np.ndarray) -> bool:
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
    vv = np.asarray(v)
    return bool((vv >= 0).all() and int(vv.sum()) == TOTAL and exists_round10_partition(vv))


def sample_partition_lengths(rng: np.random.Generator) -> np.ndarray:
    idx = int(rng.integers(0, PARTITIONS.shape[0]))
    return PARTITIONS[idx].copy()


def random_group_sums(rng: np.random.Generator) -> np.ndarray:
    """Random 5-part nonnegative composition of 100."""
    p = rng.dirichlet(np.ones(GROUPS))
    sums = rng.multinomial(TOTAL, p).astype(np.int16)
    return sums


def decode_group_sum(length: int, total_sum: int, rng: np.random.Generator) -> np.ndarray:
    """For a group of given length and sum, construct q/q+1 pattern directly."""
    q, r = divmod(int(total_sum), int(length))
    seg = np.full(int(length), q, dtype=np.int16)
    if r > 0:
        idx = rng.choice(int(length), size=int(r), replace=False)
        seg[idx] += 1
    return seg


def random_valid_round10_strategy(rng: np.random.Generator) -> np.ndarray:
    lengths = sample_partition_lengths(rng)
    sums = random_group_sums(rng)
    x = np.empty(K, dtype=np.int16)
    pos = 0
    for gi in range(GROUPS):
        m = int(lengths[gi])
        s = int(sums[gi])
        seg = decode_group_sum(m, s, rng)
        x[pos:pos + m] = seg
        pos += m
    return x


def random_valid_round10_batch(batch_size: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((batch_size, K), dtype=np.int16)
    for b in range(batch_size):
        out[b] = random_valid_round10_strategy(rng)
    return out


def repair_to_round10(x: np.ndarray, rng: Optional[np.random.Generator] = None, max_tries: int = 200) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    y = np.asarray(x, dtype=np.int16).copy()
    y[y < 0] = 0

    s = int(y.sum())
    if s < TOTAL:
        y[K - 1] += TOTAL - s
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


def dedup_rows(batch: np.ndarray) -> np.ndarray:
    if batch.shape[0] <= 1:
        return batch
    view = np.ascontiguousarray(batch).view(
        np.dtype((np.void, batch.dtype.itemsize * batch.shape[1])))
    _, idx = np.unique(view, return_index=True)
    idx.sort()
    return batch[idx]


def generate_unique_strategies(N: int, seed: int, log_every: int, batch_size: int = 50000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((N, K), dtype=np.int16)

    additional_strategies = []
    if hasattr(judge, "strategies"):
        for v in judge.strategies:
            vv = repair_to_round10(np.asarray(v, dtype=np.int16), rng)
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
        remain = N - i
        cur_batch = min(batch_size, max(remain * 2, 10000))
        cand = random_valid_round10_batch(cur_batch, rng)
        trials += cur_batch

        # first remove duplicates inside this batch cheaply in numpy
        cand = dedup_rows(cand)

        for v in cand:
            code = encode_vec_int(v)
            if code in seen:
                continue
            seen.add(code)
            out[i] = v
            i += 1
            if log_every and i % log_every == 0:
                dt = time.time() - t0
                dup_rate = 1.0 - i / \
                    max(trials + len(additional_strategies), 1)
                print(
                    f"[gen] {i:,}/{N:,} valid unique | trials={trials:,} | dup={dup_rate:.4f} | {dt:.1f}s")
            if i >= N:
                break

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
    p = argparse.ArgumentParser(description="沙堆养蛊多级筛选 round10 fast")
    p.add_argument("--seed_gen", type=int, default=202603122000)
    p.add_argument("--n0", type=int, default=5_000_000)
    p.add_argument("--gen_log_every", type=int, default=50_000)
    p.add_argument("--gen_batch_size", type=int, default=50_000,
                   help="candidate batch size for fast unique generation")
    p.add_argument("--out_dir", type=str, default=judge.name)
    p.add_argument("--n1", type=int, default=500_000)
    p.add_argument("--n2", type=int, default=10_000)
    p.add_argument("--n3", type=int, default=1_000)
    p.add_argument("--n4", type=int, default=100)
    p.add_argument("--mode1", type=str, default="tournament",
                   choices=["tournament", "full"])
    p.add_argument("--rounds1", type=int, default=1000)
    p.add_argument("--seed1", type=int, default=101)
    p.add_argument("--mode2", type=str, default="tournament",
                   choices=["tournament", "full"])
    p.add_argument("--rounds2", type=int, default=10000)
    p.add_argument("--seed2", type=int, default=202)
    p.add_argument("--mode3", type=str, default="full",
                   choices=["tournament", "full"])
    p.add_argument("--rounds3", type=int, default=120)
    p.add_argument("--seed3", type=int, default=303)
    p.add_argument("--mode4", type=str, default="full",
                   choices=["tournament", "full"])
    p.add_argument("--rounds4", type=int, default=300)
    p.add_argument("--seed4", type=int, default=404)
    p.add_argument("--log_every_rounds", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"NUMBA_OK={NUMBA_OK}")
    print(f"Output dir: {args.out_dir}")
    print("\n== Generate unique strategies ==")
    strats0 = generate_unique_strategies(args.n0, seed=args.seed_gen,
                                         log_every=args.gen_log_every,
                                         batch_size=args.gen_batch_size)

    strats1 = stage_run(strats0, args.n1, args.mode1, args.rounds1, args.seed1,
                        args.log_every_rounds, args.out_dir, "top100k")
    strats2 = stage_run(strats1, args.n2, args.mode2, args.rounds2, args.seed2,
                        args.log_every_rounds, args.out_dir, "top10k")
    strats3 = stage_run(strats2, args.n3, args.mode3, args.rounds3, args.seed3,
                        args.log_every_rounds, args.out_dir, "top1k")
    strats4 = stage_run(strats3, args.n4, args.mode4, args.rounds4, args.seed4,
                        args.log_every_rounds, args.out_dir, "top100")

    print("\n== Preview top100 first 20 ==")
    top100_path = os.path.join(args.out_dir, "top100.npz")
    data = np.load(top100_path)
    s = data["strategies"]
    w = data["wins"]
    for i in range(min(20, s.shape[0])):
        print(f"{i+1:>2d}  win={int(w[i])}  {s[i].tolist()}")

    print("\nDONE.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Round 10 compact search.

Hidden structure of a legal round-10 strategy:
- The 10 piles are split into 5 contiguous non-empty groups.
- In each group, max-min <= 1.
- Therefore, if a group's length is m and its group-sum is S,
  then every pile in the group must be either q=floor(S/m) or q+1,
  where exactly r=S%m piles take q+1.

So a legal strategy can be parameterized by:
1) 5 group lengths (positive, sum to 10)
2) 5 group sums   (nonnegative, sum to 100)
3) for each group, which r positions take the +1

This file searches directly in that compact space, instead of mutating raw
10-dimensional vectors. Every decoded candidate is guaranteed legal:
- nonnegative
- sum=100
- round10 partition exists by construction
"""

import time
import argparse
import itertools
import math
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
MIN_EACH = 2
PILE10_MIN = 35
REST = TOTAL - K * MIN_EACH

# all 126 possible 5-group positive compositions of 10
ALL_LENGTHS = [tuple(x) for x in itertools.product(
    range(1, K + 1), repeat=GROUPS) if sum(x) == K]
# cache all masks by (length, count_of_high_positions)
MASKS_BY_LEN_AND_R = {
    (m, r): [tuple(c) for c in itertools.combinations(range(m), r)]
    for m in range(1, K + 1)
    for r in range(0, m + 1)
}


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


def valid_group_segment(v: np.ndarray, l: int, r: int) -> bool:
    seg = v[l:r]
    if seg.size == 0:
        return False
    return int(seg.max()) - int(seg.min()) <= 1


def exists_round10_partition(v: np.ndarray) -> bool:
    if v.shape[0] != K:
        return False
    if (v < MIN_EACH).any() or int(v.sum()) != TOTAL:
        return False
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
    return bool(
        v.shape[0] == K and
        (v >= MIN_EACH).all() and
        int(v[9]) >= PILE10_MIN and
        int(v.sum()) == TOTAL and
        exists_round10_partition(v)
    )


def random_positive_composition(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    cuts = np.sort(rng.choice(np.arange(1, total),
                   size=parts - 1, replace=False))
    pts = np.concatenate(([0], cuts, [total]))
    return np.diff(pts).astype(np.int16)


def random_nonnegative_composition(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    # stars and bars by multinomial-like sequential draw
    xs = np.zeros(parts, dtype=np.int16)
    remain = total
    for i in range(parts - 1):
        x = int(rng.integers(0, remain + 1))
        xs[i] = x
        remain -= x
    xs[-1] = remain
    rng.shuffle(xs)
    return xs


def choose_mask(length: int, r: int, rng: np.random.Generator) -> tuple:
    opts = MASKS_BY_LEN_AND_R[(length, r)]
    idx = int(rng.integers(0, len(opts)))
    return opts[idx]


def decode_state(lengths, sums, masks) -> np.ndarray:
    x = np.empty(K, dtype=np.int16)
    pos = 0
    for m, S, mask in zip(lengths, sums, masks):
        m = int(m)
        S = int(S)
        q = S // m
        r = S % m
        seg = np.full(m, q + MIN_EACH, dtype=np.int16)
        if r > 0:
            for idx in mask:
                seg[int(idx)] += 1
        x[pos:pos + m] = seg
        pos += m
    return x


class State:
    __slots__ = ("lengths", "sums", "masks")

    def __init__(self, lengths, sums, masks):
        self.lengths = tuple(int(v) for v in lengths)
        self.sums = tuple(int(v) for v in sums)
        self.masks = tuple(tuple(int(i) for i in mask) for mask in masks)

    def decode(self) -> np.ndarray:
        return decode_state(self.lengths, self.sums, self.masks)


def state_from_vector(v: np.ndarray, rng: Optional[np.random.Generator] = None) -> State:
    # If v is already legal, recover one valid partition via DP and exact masks.
    if rng is None:
        rng = np.random.default_rng()
    vv = np.asarray(v, dtype=np.int16)
    if not is_legal_round10(vv):
        return random_state(rng)

    ok = np.zeros((K + 1, GROUPS + 1), dtype=np.bool_)
    prev = np.full((K + 1, GROUPS + 1), -1, dtype=np.int16)
    ok[0, 0] = True
    for i in range(1, K + 1):
        for g in range(1, GROUPS + 1):
            for j in range(g - 1, i):
                if ok[j, g - 1] and valid_group_segment(vv, j, i):
                    ok[i, g] = True
                    prev[i, g] = j
                    break
    if not ok[K, GROUPS]:
        return random_state(rng)

    parts = []
    i, g = K, GROUPS
    while g > 0:
        j = int(prev[i, g])
        parts.append((j, i))
        i, g = j, g - 1
    parts.reverse()

    lengths = []
    sums = []
    masks = []
    for l, r in parts:
        seg = vv[l:r]
        m = r - l
        S = int(seg.sum())
        q = S // m
        hi = tuple(int(i) for i in np.where(seg == q + 1)[0])
        lengths.append(m)
        sums.append(S)
        masks.append(hi)
    return State(lengths, sums, masks)


def random_state(rng: np.random.Generator) -> State:
    while True:
        lengths = tuple(int(v)
                        for v in random_positive_composition(K, GROUPS, rng))
        sums = tuple(int(v)
                     for v in random_nonnegative_composition(REST, GROUPS, rng))
        masks = []
        for m, S in zip(lengths, sums):
            r = int(S % m)
            masks.append(choose_mask(m, r, rng))
        st = State(lengths, sums, masks)
        if is_legal_round10(st.decode()):
            return st


def repair_state(state: State, rng: np.random.Generator) -> State:
    lengths = np.array(state.lengths, dtype=np.int16)
    if (lengths <= 0).any() or int(lengths.sum()) != K:
        lengths = random_positive_composition(K, GROUPS, rng)

    sums = np.array(state.sums, dtype=np.int16)
    sums[sums < 0] = 0
    s = int(sums.sum())
    if s < REST:
        sums[-1] += REST - s
    elif s > REST:
        extra = s - REST
        for i in range(GROUPS - 1, -1, -1):
            take = min(int(sums[i]), extra)
            sums[i] -= take
            extra -= take
            if extra == 0:
                break

    masks = []
    for m, S, old_mask in zip(lengths, sums, state.masks):
        r = int(S % m)
        old_mask = tuple(int(i) for i in old_mask if 0 <= int(i) < int(m))
        if len(old_mask) == r and len(set(old_mask)) == r:
            masks.append(tuple(sorted(old_mask)))
        else:
            masks.append(choose_mask(int(m), r, rng))

    out = State(lengths, sums, masks)
    x = out.decode()
    if not is_legal_round10(x):
        return random_state(rng)
    return out


def mutate_state(state: State, rng: np.random.Generator, max_coin_step: int = 8) -> State:
    lengths = np.array(state.lengths, dtype=np.int16)
    sums = np.array(state.sums, dtype=np.int16)
    masks = [tuple(mask) for mask in state.masks]

    op = int(rng.integers(0, 4))

    if op == 0:
        # move 1 boundary by 1 between neighboring groups
        i = int(rng.integers(0, GROUPS - 1))
        direction = -1 if int(rng.integers(0, 2)) == 0 else 1
        if direction == -1 and lengths[i] > 1:
            lengths[i] -= 1
            lengths[i + 1] += 1
        elif direction == 1 and lengths[i + 1] > 1:
            lengths[i] += 1
            lengths[i + 1] -= 1
    elif op == 1:
        # transfer coins between groups
        i = int(rng.integers(0, GROUPS))
        j = int(rng.integers(0, GROUPS))
        while j == i:
            j = int(rng.integers(0, GROUPS))
        if sums[i] > 0:
            step = int(rng.integers(1, max_coin_step + 1))
            step = min(step, int(sums[i]))
            sums[i] -= step
            sums[j] += step
    elif op == 2:
        # re-draw one group's internal +1 positions
        i = int(rng.integers(0, GROUPS))
        r = int(sums[i] % lengths[i])
        masks[i] = choose_mask(int(lengths[i]), r, rng)
    else:
        # mixed small move: nudge two group sums and refresh masks
        i = int(rng.integers(0, GROUPS))
        j = int(rng.integers(0, GROUPS))
        while j == i:
            j = int(rng.integers(0, GROUPS))
        if sums[i] > 0:
            sums[i] -= 1
            sums[j] += 1
        k = int(rng.integers(0, GROUPS))
        r = int(sums[k] % lengths[k])
        masks[k] = choose_mask(int(lengths[k]), r, rng)

    # refresh masks whose group length/residue changed
    fixed_masks = []
    for m, S, old_mask in zip(lengths, sums, masks):
        r = int(S % m)
        old_mask = tuple(int(t) for t in old_mask if 0 <= int(t) < int(m))
        if len(old_mask) == r and len(set(old_mask)) == r:
            fixed_masks.append(tuple(sorted(old_mask)))
        else:
            fixed_masks.append(choose_mask(int(m), r, rng))
    return repair_state(State(lengths, sums, fixed_masks), rng)


def random_strategy(rng):
    return random_state(rng).decode()


def repair_round10(x, rng):
    return state_from_vector(x, rng).decode()


def jitter_strategy(base, max_delta=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    st = state_from_vector(np.asarray(base, dtype=np.int16), rng)
    best = st
    for _ in range(max(10, max_delta * 8)):
        cand = mutate_state(best, rng, max_coin_step=max(1, max_delta))
        best = cand
    return best.decode()


def structured_seeds(max_delta=3, keep_original=True, rng=None):
    n = 50
    if rng is None:
        rng = np.random.default_rng()
    seeds = []
    if hasattr(judge, "strategies"):
        for s in judge.strategies:
            ss = repair_round10(s, rng)
            if keep_original and is_legal_round10(ss):
                seeds.append(ss.copy())
            for _ in range(n):
                seeds.append(jitter_strategy(ss, max_delta=max_delta, rng=rng))
    return seeds


def score_tuple(w, t, l):
    return (w, t, -l)


def local_search(pool, x0, rng, iters=3000, neighborhood=40, max_step=8, verbose=False):
    st = state_from_vector(x0, rng)
    x = st.decode()
    w, t, l = evaluate_against_pool(x, pool)
    best_state = st
    best_x = x.copy()
    best_w, best_t, best_l = w, t, l
    stall = 0
    for it in range(1, iters + 1):
        improved = False
        cand_best_state = None
        cand_best_x = None
        cand_score = None
        cand_tuple = None
        for _ in range(neighborhood):
            st2 = mutate_state(st, rng, max_coin_step=max_step)
            x2 = st2.decode()
            wy, ty, ly = evaluate_against_pool(x2, pool)
            sc = score_tuple(wy, ty, ly)
            if cand_score is None or sc > cand_score:
                cand_score = sc
                cand_best_state = st2
                cand_best_x = x2
                cand_tuple = (wy, ty, ly)
        if cand_score > score_tuple(w, t, l):
            st = cand_best_state
            x = cand_best_x
            w, t, l = cand_tuple
            improved = True
            stall = 0
            if score_tuple(w, t, l) > score_tuple(best_w, best_t, best_l):
                best_state = st
                best_x = x.copy()
                best_w, best_t, best_l = w, t, l
        else:
            stall += 1
            st = mutate_state(best_state, rng, max_coin_step=max_step * 2)
            x = st.decode()
            w, t, l = evaluate_against_pool(x, pool)
        if verbose and (it % 200 == 0):
            print(
                f"    iter={it:5d} | local_best wins={best_w} ties={best_t} losses={best_l} | stall={stall}")
    return best_x, best_w, best_t, best_l


def global_search(pool, restarts=30, iters=3000, neighborhood=40, max_step=8, seed=1, verbose=True):
    rng = np.random.default_rng(seed)
    global_best = None
    global_w = -1
    global_t = -1
    global_l = 10**9
    seeds = structured_seeds(rng=rng)
    t0 = time.time()
    for r in range(1, restarts + 1):
        bucket = (r - 1) % 10
        if bucket < 4 and pool.shape[0] > 0:
            idx = int(rng.integers(0, pool.shape[0]))
            x0 = repair_round10(pool[idx].copy(), rng)
            x0 = mutate_state(state_from_vector(x0, rng),
                              rng, max_coin_step=5).decode()
        elif bucket < 8:
            x0 = random_strategy(rng)
        else:
            if len(seeds) == 0:
                x0 = random_strategy(rng)
            else:
                x0 = seeds[int(rng.integers(0, len(seeds)))].copy()
        x, w, t, l = local_search(pool=pool, x0=x0, rng=rng, iters=iters,
                                  neighborhood=neighborhood, max_step=max_step,
                                  verbose=False)
        if score_tuple(w, t, l) > score_tuple(global_w, global_t, global_l):
            global_best = x.copy()
            global_w, global_t, global_l = w, t, l
        if verbose:
            dt = time.time() - t0
            print(
                f"[restart {r:3d}/{restarts}] best_this={w}/{t}/{l} | global_best={global_w}/{global_t}/{global_l} | x={global_best.tolist()} | {dt:.1f}s")
    return global_best, global_w, global_t, global_l


def filter_legal_pool(pool):
    return pool
    # keep = [i for i in range(pool.shape[0]) if is_legal_round10(pool[i])]
    # if len(keep) == pool.shape[0]:
    #     return pool
    # if len(keep) == 0:
    #     raise ValueError("input pool has no legal round10 strategies")
    # return pool[np.array(keep, dtype=np.int32)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=judge.name + "/top10k.npz")
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--neighborhood", type=int, default=40)
    parser.add_argument("--max_step", type=int, default=10)
    parser.add_argument("--seed", type=int, default=202603122235)
    parser.add_argument("--save", type=str,
                        default=judge.name + "/best_vs_top10k.txt")
    args = parser.parse_args()

    print(f"NUMBA_OK={NUMBA_OK}")
    print(f"Total possible partitions(len only) = {len(ALL_LENGTHS)}")
    print(f"Loading {args.input} ...")
    data = np.load(args.input)
    pool = data["strategies"].astype(np.int16)
    raw_n = pool.shape[0]
    pool = filter_legal_pool(pool)
    print(
        f"Pool size = {pool.shape[0]:,} (legal / raw = {pool.shape[0]:,} / {raw_n:,})")

    best_x, best_w, best_t, best_l = global_search(pool=pool, restarts=args.restarts,
                                                   iters=args.iters, neighborhood=args.neighborhood,
                                                   max_step=args.max_step, seed=args.seed, verbose=True)
    print("\n=== FINAL BEST ===")
    print("strategy =", best_x.tolist())
    print(f"wins={best_w}, ties={best_t}, losses={best_l}")
    print(f"win_rate={best_w / pool.shape[0]:.4f}")

    with open(args.save, "a", encoding="utf8") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} round10-compact wins={best_w} ties={best_t} losses={best_l} win_rate={best_w / pool.shape[0]:.6f}\n")
        f.write("strategy: " + " ".join(map(str, best_x.tolist())) + "\n")
    print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Round 15 local/global search against top10k.npz
Strategy format:
[buyA, buyB, buyC, insurance, p1, ..., p10]
"""

import os
import time
import heapq
import argparse
import numpy as np
import judge

try:
    from numba import njit
    NUMBA_OK = False
except Exception:
    NUMBA_OK = False

TOTAL = judge.START_COINS
DIM = judge.STRATEGY_LEN
K = judge.K
BAD_LOSS = 10 ** 9

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

    @njit(cache=True)
    def evaluate_against_indices(x, pool, idxs):
        wins = 0
        ties = 0
        losses = 0
        for ii in range(idxs.shape[0]):
            i = idxs[ii]
            r = match_result(x, pool[i])
            if r == 1:
                wins += 1
            elif r == 0:
                ties += 1
            else:
                losses += 1
        return wins, ties, losses

    @njit(cache=True)
    def evaluate_must_beat(x, must_beat_pool):
        wins = 0
        ties = 0
        losses = 0
        for i in range(must_beat_pool.shape[0]):
            r = match_result(x, must_beat_pool[i])
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

    def evaluate_against_indices(x, pool, idxs):
        wins = 0
        ties = 0
        losses = 0
        for i in idxs:
            r = match_result(x, pool[i])
            if r == 1:
                wins += 1
            elif r == 0:
                ties += 1
            else:
                losses += 1
        return wins, ties, losses

    def evaluate_must_beat(x, must_beat_pool):
        wins = 0
        ties = 0
        losses = 0
        for i in range(must_beat_pool.shape[0]):
            r = match_result(x, must_beat_pool[i])
            if r == 1:
                wins += 1
            elif r == 0:
                ties += 1
            else:
                losses += 1
        return wins, ties, losses


def normalize_min_values(min_values=None):
    if min_values is None:
        arr = np.zeros(DIM, dtype=np.int16)
    else:
        arr = np.asarray(min_values, dtype=np.int16)
        if arr.shape != (DIM,):
            raise ValueError(f'min_values must have shape ({DIM},), got {arr.shape}')
    if np.any(arr < 0):
        raise ValueError('all min_values must be >= 0')
    return arr.astype(np.int16)


def min_values_to_text(min_values):
    return ','.join(map(str, np.asarray(min_values, dtype=np.int16).tolist()))


def parse_must_beat(text):
    if text is None:
        return np.zeros((0, DIM), dtype=np.int16)
    if isinstance(text, (list, tuple, np.ndarray)):
        arr = np.asarray(text, dtype=np.int16)
        if arr.size == 0:
            return np.zeros((0, DIM), dtype=np.int16)
        if arr.ndim == 1:
            if arr.shape[0] != DIM:
                raise ValueError(f'each must_beat strategy must have length {DIM}')
            arr = arr.reshape(1, DIM)
        if arr.ndim != 2 or arr.shape[1] != DIM:
            raise ValueError(f'must_beat array must have shape (n, {DIM})')
        return arr.astype(np.int16)
    raise ValueError('must_beat must be array-like for round 15')


def must_beat_to_text(must_beat):
    if must_beat is None or len(must_beat) == 0:
        return '(none)'
    return '; '.join(' '.join(map(str, row.tolist())) for row in must_beat)


def exact_sum_repair(values, total, mins):
    y = np.maximum(np.asarray(values, dtype=np.int16), np.asarray(mins, dtype=np.int16)).astype(np.int16)
    s = int(y.sum())
    if s < total:
        y[K - 1] += total - s
        return y.astype(np.int16)
    extra = s - total
    for i in range(K - 1, -1, -1):
        can_take = int(y[i] - mins[i])
        if can_take <= 0:
            continue
        take = min(can_take, extra)
        y[i] -= take
        extra -= take
        if extra == 0:
            break
    if extra != 0:
        raise ValueError('repair failed on pile allocation')
    return y.astype(np.int16)


def repair_strategy(x, mins):
    y = np.asarray(x, dtype=np.int16).copy()
    buy_a = 1 if int(y[0]) > 0 else 0
    buy_b = 1 if int(y[1]) > 0 else 0
    buy_c = 1 if int(y[2]) > 0 else 0

    skill_cost = judge.skill_cost_from_flags(buy_a, buy_b, buy_c)
    pile_mins = np.maximum(mins[4:14], 0).astype(np.int16)
    min_insurance_when_a = max(1, int(mins[3]))

    if buy_a == 1:
        max_ins = TOTAL - skill_cost - int(pile_mins.sum())
        if max_ins < min_insurance_when_a:
            buy_a = 0
            skill_cost = judge.skill_cost_from_flags(buy_a, buy_b, buy_c)
        
    if buy_a == 1:
        insurance = max(min_insurance_when_a, int(y[3]))
        max_ins = TOTAL - skill_cost - int(pile_mins.sum())
        insurance = min(insurance, max_ins)
    else:
        insurance = 0

    pile_budget = TOTAL - skill_cost - insurance
    if pile_budget < int(pile_mins.sum()):
        if buy_a == 1:
            insurance = max(min_insurance_when_a, TOTAL - skill_cost - int(pile_mins.sum()))
            if insurance < min_insurance_when_a:
                buy_a = 0
                insurance = 0
                skill_cost = judge.skill_cost_from_flags(buy_a, buy_b, buy_c)
            pile_budget = TOTAL - skill_cost - insurance
        if pile_budget < int(pile_mins.sum()):
            return judge.sample_random_strategy(np.random.default_rng(), mins)

    piles = exact_sum_repair(y[4:14], pile_budget, pile_mins)

    z = np.zeros(DIM, dtype=np.int16)
    z[0] = buy_a
    z[1] = buy_b
    z[2] = buy_c
    z[3] = insurance
    z[4:14] = piles
    if not judge.is_valid_strategy(z):
        return judge.sample_random_strategy(np.random.default_rng(), mins)
    return z.astype(np.int16)


def random_strategy(rng, mins):
    return judge.sample_random_strategy(rng, mins)


def mutate_strategy(x, rng, mins, max_step=8, moves=3):
    y = np.asarray(x, dtype=np.int16).copy()
    op = int(rng.integers(0, 5))

    if op == 0:
        # pile transfer
        for _ in range(int(rng.integers(1, moves + 1))):
            i = 4 + int(rng.integers(0, K))
            j = 4 + int(rng.integers(0, K - 1))
            if j >= i:
                j += 1
            available = int(y[i] - mins[i])
            if available <= 0:
                continue
            step = min(int(rng.integers(1, max_step + 1)), available)
            y[i] -= step
            y[j] += step
    elif op == 1:
        # toggle one skill
        idx = int(rng.integers(0, 3))
        y[idx] = 1 - int(y[idx] > 0)
        if idx == 0 and y[0] == 0:
            y[3] = 0
    elif op == 2:
        # adjust insurance when A is on
        if int(y[0]) == 0:
            y[0] = 1
        delta = int(rng.integers(-max_step, max_step + 1))
        if delta == 0:
            delta = 1
        y[3] = max(1, int(y[3]) + delta)
    elif op == 3:
        # resample piles around current skills
        base = random_strategy(rng, mins)
        y[4:14] = base[4:14]
    else:
        # light jitter on all coordinates then repair
        noise = rng.integers(-2, 3, size=DIM)
        y = y + noise.astype(np.int16)

    return repair_strategy(y, mins)


def jitter_strategy(base, rng, mins, max_delta=3):
    base = np.asarray(base, dtype=np.int16)
    noise = rng.integers(-max_delta, max_delta + 1, size=base.shape)
    return repair_strategy(base + noise, mins)


def structured_seeds(rng, mins, max_delta=3, keep_original=True, per_base=24):
    seeds = []
    for s in judge.strategies:
        base = repair_strategy(s, mins)
        if keep_original:
            seeds.append(base.copy())
        for _ in range(per_base):
            seeds.append(jitter_strategy(base, rng=rng, mins=mins, max_delta=max_delta))
    return seeds


def score_tuple(w, t, l, must_beat_w=0, must_beat_t=0, must_beat_l=0, must_beat_total=0):
    unmet = must_beat_total - must_beat_w
    return (must_beat_w, -must_beat_l, -unmet, w, t, -l)


def eval_candidate(x, pool, must_beat_pool, prescreen_indices=None, full_eval=False):
    if must_beat_pool is None or must_beat_pool.shape[0] == 0:
        mbw, mbt, mbl = 0, 0, 0
        mbtotal = 0
    else:
        mbw, mbt, mbl = evaluate_must_beat(x, must_beat_pool)
        mbtotal = int(must_beat_pool.shape[0])
        if mbw != mbtotal:
            return -1, 0, BAD_LOSS, mbw, mbt, mbl, mbtotal

    if full_eval or prescreen_indices is None or prescreen_indices.shape[0] == pool.shape[0]:
        w, t, l = evaluate_against_pool(x, pool)
    else:
        w, t, l = evaluate_against_indices(x, pool, prescreen_indices)
    return w, t, l, mbw, mbt, mbl, mbtotal


def pick_prescreen_indices(pool_size, prescreen_size, rng):
    if prescreen_size <= 0 or prescreen_size >= pool_size:
        return None
    idxs = rng.choice(pool_size, size=prescreen_size, replace=False)
    return np.asarray(idxs, dtype=np.int32)


def local_search(pool, x0, rng, mins, must_beat_pool=None, iters=3000,
                 neighborhood=40, max_step=8, prescreen_size=1024,
                 full_eval_top_k=4, refresh_prescreen_every=25, verbose=False):
    full_indices = None
    prescreen_indices = pick_prescreen_indices(pool.shape[0], prescreen_size, rng)

    x = x0.copy()
    w, t, l, mbw, mbt, mbl, mbtotal = eval_candidate(x, pool, must_beat_pool, prescreen_indices=full_indices, full_eval=True)
    best_local = x.copy()
    best_w, best_t, best_l = w, t, l
    best_mbw, best_mbt, best_mbl, best_mbtotal = mbw, mbt, mbl, mbtotal
    stall = 0

    for it in range(1, iters + 1):
        if refresh_prescreen_every > 0 and (it == 1 or it % refresh_prescreen_every == 0):
            prescreen_indices = pick_prescreen_indices(pool.shape[0], prescreen_size, rng)

        seen = set()
        prescreen_best = []
        target_unique = max(neighborhood, full_eval_top_k)
        tries = 0
        max_tries = max(neighborhood * 4, target_unique + 8)

        while len(seen) < target_unique and tries < max_tries:
            tries += 1
            y = mutate_strategy(x, rng, mins=mins, max_step=max_step, moves=3)
            key = tuple(int(v) for v in y)
            if key in seen:
                continue
            seen.add(key)

            wy, ty, ly, mbwy, mbty, mbly, mbtotaly = eval_candidate(
                y, pool, must_beat_pool, prescreen_indices=prescreen_indices, full_eval=False
            )
            prescore = score_tuple(wy, ty, ly, mbwy, mbty, mbly, mbtotaly)
            heapq.heappush(prescreen_best, (prescore, key, y))
            if len(prescreen_best) > full_eval_top_k:
                heapq.heappop(prescreen_best)

        if not prescreen_best:
            stall += 1
            x = mutate_strategy(best_local, rng, mins=mins, max_step=max_step * 2, moves=6)
            continue

        final_best = None
        final_score = None
        finalists = sorted(prescreen_best, key=lambda z: z[0], reverse=True)
        for _, _, y in finalists:
            wy, ty, ly, mbwy, mbty, mbly, mbtotaly = eval_candidate(
                y, pool, must_beat_pool, prescreen_indices=full_indices, full_eval=True
            )
            sc = score_tuple(wy, ty, ly, mbwy, mbty, mbly, mbtotaly)
            if final_score is None or sc > final_score:
                final_score = sc
                final_best = (y, wy, ty, ly, mbwy, mbty, mbly, mbtotaly)

        current_score = score_tuple(w, t, l, mbw, mbt, mbl, mbtotal)
        improved = final_score is not None and final_score > current_score

        if improved:
            x, w, t, l, mbw, mbt, mbl, mbtotal = final_best
            stall = 0
            if final_score > score_tuple(best_w, best_t, best_l, best_mbw, best_mbt, best_mbl, best_mbtotal):
                best_local = x.copy()
                best_w, best_t, best_l = w, t, l
                best_mbw, best_mbt, best_mbl, best_mbtotal = mbw, mbt, mbl, mbtotal
        else:
            stall += 1
            x = mutate_strategy(best_local, rng, mins=mins, max_step=max_step * 2, moves=6)
            w, t, l, mbw, mbt, mbl, mbtotal = eval_candidate(x, pool, must_beat_pool, prescreen_indices=full_indices, full_eval=True)

        if verbose and (it % 200 == 0):
            print(f'    iter={it:5d} | local_best must_beat={best_mbw}/{best_mbtotal} | wins={best_w} ties={best_t} losses={best_l} | stall={stall}')

    return best_local, best_w, best_t, best_l, best_mbw, best_mbt, best_mbl, best_mbtotal


def global_search(pool, restarts=30, iters=3000, neighborhood=40, max_step=8,
                  seed=1, verbose=True, min_values=None,
                  must_beat_pool=None, prescreen_size=1024,
                  full_eval_top_k=4, refresh_prescreen_every=25,
                  seed_jitters=24):
    mins = normalize_min_values(min_values=min_values)
    rng = np.random.default_rng(seed)

    global_best = None
    global_w = -1
    global_t = -1
    global_l = BAD_LOSS
    global_mbw = -1
    global_mbt = 0
    global_mbl = BAD_LOSS
    global_mbtotal = 0

    restart_results = []
    seeds = structured_seeds(rng=rng, mins=mins, per_base=seed_jitters)
    t0 = time.time()

    for r in range(1, restarts + 1):
        p = rng.random()
        if p < 0.4:
            idx = int(rng.integers(0, pool.shape[0]))
            x0 = repair_strategy(pool[idx].copy(), mins)
            x0 = mutate_strategy(x0, rng, mins=mins, max_step=5, moves=2)
            source = 'pool'
        elif p < 0.8:
            x0 = random_strategy(rng, mins=mins)
            source = 'random'
        else:
            x0 = seeds[int(rng.integers(0, len(seeds)))].copy()
            source = 'seed'

        x, w, t, l, mbw, mbt, mbl, mbtotal = local_search(
            pool=pool,
            x0=x0,
            rng=rng,
            mins=mins,
            must_beat_pool=must_beat_pool,
            iters=iters,
            neighborhood=neighborhood,
            max_step=max_step,
            prescreen_size=prescreen_size,
            full_eval_top_k=full_eval_top_k,
            refresh_prescreen_every=refresh_prescreen_every,
            verbose=False,
        )

        is_global_best = False
        if score_tuple(w, t, l, mbw, mbt, mbl, mbtotal) > score_tuple(global_w, global_t, global_l, global_mbw, global_mbt, global_mbl, global_mbtotal):
            global_best = x.copy()
            global_w, global_t, global_l = w, t, l
            global_mbw, global_mbt, global_mbl, global_mbtotal = mbw, mbt, mbl, mbtotal
            is_global_best = True

        restart_results.append({
            'restart': r,
            'source': source,
            'wins': w,
            'ties': t,
            'losses': l,
            'must_beat_wins': mbw,
            'must_beat_ties': mbt,
            'must_beat_losses': mbl,
            'must_beat_total': mbtotal,
            'must_beat_ok': (mbw == mbtotal),
            'skill_cost': judge.strategy_skill_cost(x),
            'used': judge.strategy_total_used(x),
            'strategy': x.copy(),
            'is_global_best': is_global_best,
        })

        if verbose:
            dt = time.time() - t0
            flag = ' <-- GLOBAL BEST' if is_global_best else ''
            print(
                f"[restart {r:3d}/{restarts}] src={source:<6} "
                f"must_beat={mbw}/{mbtotal} best_this={w}/{t}/{l} "
                f"| skill_cost={judge.strategy_skill_cost(x):2d} used={judge.strategy_total_used(x):3d} "
                f"| global_best must_beat={global_mbw}/{global_mbtotal} {global_w}/{global_t}/{global_l} "
                f"| x={x.tolist()} | {dt:.1f}s{flag}"
            )

    return global_best, global_w, global_t, global_l, global_mbw, global_mbt, global_mbl, global_mbtotal, restart_results, mins


def save_restart_results(path, results, best_x, best_w, best_t, best_l,
                         best_mbw, best_mbt, best_mbl, best_mbtotal,
                         pool_size, min_values, must_beat_pool, args):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf8') as f:
        f.write('=' * 80 + '\n')
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} round={judge.round_no} search "
            f"| min_values={min_values_to_text(min_values)} | pool={pool_size} "
            f"| must_beat={must_beat_to_text(must_beat_pool)} "
            f"| prescreen_size={args.prescreen_size} | full_eval_top_k={args.full_eval_top_k} "
            f"| refresh_prescreen_every={args.refresh_prescreen_every}\n"
        )
        f.write('-' * 80 + '\n')
        for item in results:
            mark = ' <-- GLOBAL BEST' if item['is_global_best'] else ''
            ok = 'OK' if item['must_beat_ok'] else 'NO'
            f.write(
                f"restart={item['restart']:>3d} src={item['source']:<6} "
                f"must_beat={item['must_beat_wins']}/{item['must_beat_total']}({ok}) "
                f"wins={item['wins']} ties={item['ties']} losses={item['losses']} "
                f"skill_cost={item['skill_cost']} used={item['used']} "
                f"strategy={' '.join(map(str, item['strategy'].tolist()))}{mark}\n"
            )
        f.write('-' * 80 + '\n')
        final_ok = 'OK' if best_mbw == best_mbtotal else 'NO'
        f.write(
            f"FINAL BEST must_beat={best_mbw}/{best_mbtotal}({final_ok}) wins={best_w} ties={best_t} losses={best_l} "
            f"skill_cost={judge.strategy_skill_cost(best_x)} used={judge.strategy_total_used(best_x)}\n"
        )
        f.write('BEST strategy: ' + ' '.join(map(str, best_x.tolist())) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=judge.name + '/top10k.npz')
    parser.add_argument('--restarts', type=int, default=10)
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--neighborhood', type=int, default=40)
    parser.add_argument('--max_step', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260316)
    parser.add_argument('--save', type=str, default=judge.name + '/best_vs_top10k.txt')
    parser.add_argument('--prescreen_size', type=int, default=1024)
    parser.add_argument('--full_eval_top_k', type=int, default=4)
    parser.add_argument('--refresh_prescreen_every', type=int, default=25)
    parser.add_argument('--seed_jitters', type=int, default=24)
    args = parser.parse_args()

    mins = normalize_min_values(judge.min_values)
    must_beat_pool = parse_must_beat(judge.must_beat)

    print(f'NUMBA_OK={NUMBA_OK}')
    print(f'Round = {judge.round_no}')
    print(f'Loading {args.input} ...')
    print(f'Skill costs: A={judge.SKILL_COST_A}, B={judge.SKILL_COST_B}, C={judge.SKILL_COST_C}')
    print(f'min_values = {mins.tolist()}')
    print(f'prescreen_size = {args.prescreen_size}')
    print(f'full_eval_top_k = {args.full_eval_top_k}')
    print(f'refresh_prescreen_every = {args.refresh_prescreen_every}')
    print(f'must_beat count = {must_beat_pool.shape[0]}')

    data = np.load(args.input)
    pool = data['strategies'].astype(np.int16)
    print(f'Pool size = {pool.shape[0]:,}')

    best_x, best_w, best_t, best_l, best_mbw, best_mbt, best_mbl, best_mbtotal, restart_results, mins = global_search(
        pool=pool,
        restarts=args.restarts,
        iters=args.iters,
        neighborhood=args.neighborhood,
        max_step=args.max_step,
        seed=args.seed,
        verbose=True,
        min_values=mins,
        must_beat_pool=must_beat_pool,
        prescreen_size=args.prescreen_size,
        full_eval_top_k=args.full_eval_top_k,
        refresh_prescreen_every=args.refresh_prescreen_every,
        seed_jitters=args.seed_jitters,
    )

    print('\n=== FINAL BEST ===')
    print('min_values =', mins.tolist())
    print('strategy =', best_x.tolist())
    print('format =', judge.format_strategy(best_x))
    print(f'skill_cost={judge.strategy_skill_cost(best_x)}, used={judge.strategy_total_used(best_x)}')
    print(f'must_beat={best_mbw}/{best_mbtotal}')
    print(f'wins={best_w}, ties={best_t}, losses={best_l}')
    print(f'win_rate={best_w / pool.shape[0]:.4f}')
    if best_mbtotal > 0 and best_mbw != best_mbtotal:
        print('WARNING: current best result did NOT beat all required strategies.')

    save_restart_results(args.save, restart_results, best_x, best_w, best_t, best_l,
                         best_mbw, best_mbt, best_mbl, best_mbtotal,
                         pool.shape[0], mins, must_beat_pool, args)
    print(f'Saved to {args.save}')


if __name__ == '__main__':
    main()

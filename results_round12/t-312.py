# -*- coding: utf-8 -*-
"""
Search best strategies against top10k.npz for round12.

- Supports sum <= 100
- Supports per-pile minimum values, e.g. [0,1,2,2,...]
- Saves every restart result
- Highlights global best separately
- Supports required-opponent constraints via --must_beat
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
    def evaluate_must_beat(x, must_beat_pool):
        beat_wins = 0
        beat_ties = 0
        beat_losses = 0
        for i in range(must_beat_pool.shape[0]):
            r = match_result(x, must_beat_pool[i])
            if r == 1:
                beat_wins += 1
            elif r == 0:
                beat_ties += 1
            else:
                beat_losses += 1
        return beat_wins, beat_ties, beat_losses
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

    def evaluate_must_beat(x, must_beat_pool):
        beat_wins = 0
        beat_ties = 0
        beat_losses = 0
        for i in range(must_beat_pool.shape[0]):
            r = match_result(x, must_beat_pool[i])
            if r == 1:
                beat_wins += 1
            elif r == 0:
                beat_ties += 1
            else:
                beat_losses += 1
        return beat_wins, beat_ties, beat_losses


def normalize_min_values(min_values=None, min_value=None):
    if min_values is None:
        if min_value is None:
            arr = np.zeros(K, dtype=np.int16)
        else:
            arr = np.full(K, int(min_value), dtype=np.int16)
    elif isinstance(min_values, str):
        text = min_values.strip()
        if not text:
            arr = np.zeros(K, dtype=np.int16)
        else:
            parts = [p.strip() for p in text.split(',') if p.strip() != '']
            if len(parts) != K:
                raise ValueError(
                    f'min_values must contain exactly {K} integers, got {len(parts)}')
            arr = np.array([int(p) for p in parts], dtype=np.int16)
    else:
        arr = np.asarray(min_values, dtype=np.int16)
        if arr.shape != (K,):
            raise ValueError(
                f'min_values must have shape ({K},), got {arr.shape}')

    if np.any(arr < 0):
        raise ValueError('all min_values must be >= 0')
    if int(arr.sum()) > TOTAL:
        raise ValueError(
            f'invalid min_values: sum(min_values)={int(arr.sum())} > TOTAL={TOTAL}')
    return arr


def min_values_to_text(min_values):
    return ','.join(map(str, np.asarray(min_values, dtype=np.int16).tolist()))


def parse_must_beat(text):
    if text is None:
        return np.zeros((0, K), dtype=np.int16)
    if isinstance(text, (list, tuple, np.ndarray)):
        arr = np.asarray(text, dtype=np.int16)
        if arr.size == 0:
            return np.zeros((0, K), dtype=np.int16)
        if arr.ndim == 1:
            if arr.shape[0] != K:
                raise ValueError(
                    f'each must_beat strategy must have length {K}')
            arr = arr.reshape(1, K)
        if arr.ndim != 2 or arr.shape[1] != K:
            raise ValueError(f'must_beat array must have shape (n, {K})')
        return arr.astype(np.int16)

    text = str(text).strip()
    if not text:
        return np.zeros((0, K), dtype=np.int16)

    rows = []
    for idx, item in enumerate(text.split(';'), start=1):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(',') if p.strip() != '']
        if len(parts) != K:
            raise ValueError(
                f'must_beat strategy #{idx} must contain exactly {K} integers, got {len(parts)}')
        arr = np.array([int(p) for p in parts], dtype=np.int16)
        rows.append(arr)

    if not rows:
        return np.zeros((0, K), dtype=np.int16)
    return np.stack(rows).astype(np.int16)


def must_beat_to_text(must_beat):
    if must_beat is None or len(must_beat) == 0:
        return '(none)'
    return '; '.join(','.join(map(str, row.tolist())) for row in must_beat)


def random_strategy(rng, min_values=None, min_value=None):
    mins = normalize_min_values(min_values=min_values, min_value=min_value)
    base = mins.copy()
    remain = TOTAL - int(base.sum())
    used_extra = int(rng.integers(0, remain + 1))
    if used_extra == 0:
        return base.copy()
    p = rng.dirichlet(np.ones(K))
    extra = rng.multinomial(used_extra, p).astype(np.int16)
    return (base + extra).astype(np.int16)


def repair_strategy(x, min_values=None, min_value=None):
    mins = normalize_min_values(min_values=min_values, min_value=min_value)
    y = np.asarray(x, dtype=np.int16).copy()
    y = np.maximum(y, mins).astype(np.int16)
    s = int(y.sum())
    if s <= TOTAL:
        return y

    extra = s - TOTAL
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
        raise ValueError(
            'repair failed: cannot satisfy sum<=TOTAL with given min_values')
    return y.astype(np.int16)


def mutate_strategy(x, rng, max_step=8, moves=3, min_values=None, min_value=None):
    mins = normalize_min_values(min_values=min_values, min_value=min_value)
    y = np.asarray(x, dtype=np.int16).copy()
    m = int(rng.integers(1, moves + 1))

    for _ in range(m):
        mode = int(rng.integers(0, 3))
        total_now = int(y.sum())

        if mode == 0:
            i = int(rng.integers(0, K))
            j = int(rng.integers(0, K))
            while j == i:
                j = int(rng.integers(0, K))
            available = int(y[i] - mins[i])
            if available <= 0:
                continue
            step = min(int(rng.integers(1, max_step + 1)), available)
            y[i] -= step
            y[j] += step

        elif mode == 1:
            i = int(rng.integers(0, K))
            available = int(y[i] - mins[i])
            if available <= 0:
                continue
            step = min(int(rng.integers(1, max_step + 1)), available)
            y[i] -= step

        else:
            spare = TOTAL - total_now
            if spare <= 0:
                continue
            j = int(rng.integers(0, K))
            step = min(int(rng.integers(1, max_step + 1)), spare)
            y[j] += step

    return y.astype(np.int16)


def jitter_strategy(base, max_delta=3, rng=None, min_values=None, min_value=None):
    if rng is None:
        rng = np.random.default_rng()
    base = np.asarray(base, dtype=np.int16)
    noise = rng.integers(-max_delta, max_delta + 1, size=base.shape)
    new_x = base + noise
    return repair_strategy(new_x, min_values=min_values, min_value=min_value)


def structured_seeds(max_delta=3, keep_original=True, rng=None, min_values=None, min_value=None):
    n = 100
    if rng is None:
        rng = np.random.default_rng()
    seeds = []
    for s in judge.strategies:
        base = repair_strategy(s, min_values=min_values, min_value=min_value)
        if keep_original:
            seeds.append(base.copy())
        for _ in range(n):
            seeds.append(jitter_strategy(
                base, max_delta=max_delta, rng=rng, min_values=min_values, min_value=min_value))
    return seeds


def score_tuple(w, t, l, must_beat_w=0, must_beat_t=0, must_beat_l=0, must_beat_total=0):
    unmet = must_beat_total - must_beat_w
    return (must_beat_w, -must_beat_l, -unmet, w, t, -l)


def eval_all(x, pool, must_beat_pool=None):
    w, t, l = evaluate_against_pool(x, pool)
    if must_beat_pool is None or must_beat_pool.shape[0] == 0:
        mbw, mbt, mbl = 0, 0, 0
        mbtotal = 0
    else:
        mbw, mbt, mbl = evaluate_must_beat(x, must_beat_pool)
        mbtotal = int(must_beat_pool.shape[0])
    return w, t, l, mbw, mbt, mbl, mbtotal


def local_search(pool, x0, rng, iters=3000, neighborhood=40, max_step=8,
                 verbose=False, min_values=None, min_value=None, must_beat_pool=None):
    x = x0.copy()
    w, t, l, mbw, mbt, mbl, mbtotal = eval_all(x, pool, must_beat_pool)
    best_local = x.copy()
    best_w, best_t, best_l = w, t, l
    best_mbw, best_mbt, best_mbl, best_mbtotal = mbw, mbt, mbl, mbtotal
    stall = 0

    for it in range(1, iters + 1):
        improved = False
        cand_best = None
        cand_score = None

        for _ in range(neighborhood):
            y = mutate_strategy(x, rng, max_step=max_step, moves=3,
                                min_values=min_values, min_value=min_value)
            wy, ty, ly, mbwy, mbty, mbly, mbtotaly = eval_all(
                y, pool, must_beat_pool)
            sc = score_tuple(wy, ty, ly, mbwy, mbty, mbly, mbtotaly)
            if cand_score is None or sc > cand_score:
                cand_score = sc
                cand_best = (y, wy, ty, ly, mbwy, mbty, mbly, mbtotaly)

        if cand_score > score_tuple(w, t, l, mbw, mbt, mbl, mbtotal):
            x, w, t, l, mbw, mbt, mbl, mbtotal = cand_best
            improved = True
            stall = 0
            if score_tuple(w, t, l, mbw, mbt, mbl, mbtotal) > score_tuple(
                best_w, best_t, best_l, best_mbw, best_mbt, best_mbl, best_mbtotal
            ):
                best_local = x.copy()
                best_w, best_t, best_l = w, t, l
                best_mbw, best_mbt, best_mbl, best_mbtotal = mbw, mbt, mbl, mbtotal
        else:
            stall += 1

        if not improved:
            x = mutate_strategy(best_local, rng, max_step=max_step * 2, moves=6,
                                min_values=min_values, min_value=min_value)
            w, t, l, mbw, mbt, mbl, mbtotal = eval_all(x, pool, must_beat_pool)

        if verbose and (it % 200 == 0):
            print(
                f'    iter={it:5d} | local_best must_beat={best_mbw}/{best_mbtotal} '
                f'| wins={best_w} ties={best_t} losses={best_l} | stall={stall}')

    return best_local, best_w, best_t, best_l, best_mbw, best_mbt, best_mbl, best_mbtotal


def global_search(pool, restarts=30, iters=3000, neighborhood=40, max_step=8,
                  seed=1, verbose=True, min_values=None, min_value=None,
                  must_beat_pool=None):
    mins = normalize_min_values(min_values=min_values, min_value=min_value)
    rng = np.random.default_rng(seed)
    global_best = None
    global_w = -1
    global_t = -1
    global_l = 10**9
    global_mbw = -1
    global_mbt = 0
    global_mbl = 10**9
    global_mbtotal = 0
    restart_results = []
    seeds = structured_seeds(min_values=mins)
    t0 = time.time()

    for r in range(1, restarts + 1):
        p = rng.random()
        if p < 0.4:
            idx = int(rng.integers(0, pool.shape[0]))
            x0 = pool[idx].copy()
            x0 = repair_strategy(x0, min_values=mins)
            x0 = mutate_strategy(x0, rng, max_step=5, moves=2, min_values=mins)
            source = 'pool'
        elif p < 0.8:
            x0 = random_strategy(rng, min_values=mins)
            source = 'random'
        else:
            x0 = seeds[int(rng.integers(0, len(seeds)))].copy()
            source = 'seed'

        x, w, t, l, mbw, mbt, mbl, mbtotal = local_search(
            pool, x0, rng, iters=iters, neighborhood=neighborhood,
            max_step=max_step, verbose=False, min_values=mins,
            must_beat_pool=must_beat_pool
        )
        is_global_best = False
        if score_tuple(w, t, l, mbw, mbt, mbl, mbtotal) > score_tuple(
            global_w, global_t, global_l, global_mbw, global_mbt, global_mbl, global_mbtotal
        ):
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
            'used': int(x.sum()),
            'saved': int(TOTAL - int(x.sum())),
            'strategy': x.copy(),
            'is_global_best': is_global_best,
        })

        if verbose:
            dt = time.time() - t0
            flag = '  <-- GLOBAL BEST' if is_global_best else ''
            print(
                f"[restart {r:3d}/{restarts}] src={source:<6} "
                f"must_beat={mbw}/{mbtotal} best_this={w}/{t}/{l} "
                f"| used={int(x.sum()):3d} saved={TOTAL - int(x.sum()):3d} "
                f"| global_best must_beat={global_mbw}/{global_mbtotal} {global_w}/{global_t}/{global_l} "
                f"| x={x.tolist()} | {dt:.1f}s{flag}"
            )

    return (
        global_best, global_w, global_t, global_l,
        global_mbw, global_mbt, global_mbl, global_mbtotal,
        restart_results, mins
    )


def save_restart_results(path, results, best_x, best_w, best_t, best_l,
                         best_mbw, best_mbt, best_mbl, best_mbtotal,
                         pool_size, min_values, must_beat_pool):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf8') as f:
        f.write('=' * 80 + '\n')
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} round12 search "
            f"| min_values={min_values_to_text(min_values)} "
            f"| pool={pool_size} | must_beat={must_beat_to_text(must_beat_pool)}\n")
        f.write('-' * 80 + '\n')
        for item in results:
            mark = '  <-- GLOBAL BEST' if item['is_global_best'] else ''
            ok = 'OK' if item['must_beat_ok'] else 'NO'
            f.write(
                f"restart={item['restart']:>3d} src={item['source']:<6} "
                f"must_beat={item['must_beat_wins']}/{item['must_beat_total']}({ok}) "
                f"wins={item['wins']} ties={item['ties']} losses={item['losses']} "
                f"used={item['used']} saved={item['saved']} "
                f"strategy={' '.join(map(str, item['strategy'].tolist()))}{mark}\n"
            )
        f.write('-' * 80 + '\n')
        final_ok = 'OK' if best_mbw == best_mbtotal else 'NO'
        f.write(
            f"FINAL BEST must_beat={best_mbw}/{best_mbtotal}({final_ok}) "
            f"wins={best_w} ties={best_t} losses={best_l} "
            f"used={int(best_x.sum())} saved={TOTAL - int(best_x.sum())}\n")
        f.write('BEST strategy: ' + ' '.join(map(str, best_x.tolist())) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default=judge.name + '/top10k.npz')
    parser.add_argument('--restarts', type=int, default=20)
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--neighborhood', type=int, default=40)
    parser.add_argument('--max_step', type=int, default=8)
    parser.add_argument('--seed', type=int, default=202603082229)
    parser.add_argument('--save', type=str,
                        default=judge.name + '/best_vs_top10k.txt')
    parser.add_argument('--min_value', type=int, default=None,
                        help='uniform minimum coins for each pile')
    parser.add_argument('--min_values', type=str, default='0,1,2,2,2,2,2,2,2,2',
                        help='comma separated per-pile minimums, e.g. 0,1,2,2,2,2,2,2,2,2')
    parser.add_argument('--must_beat', type=str, default='0,0,0,0,0,0,0,0,0,0;10,10,10,10,10,10,10,10,10,10;1,1,11,11,1,21,24,28,1,1;0,3,3,3,7,10,13,17,19,22;0,0,0,0,16,22,27,0,0,35',
                        help='semicolon-separated strategies that the result must beat, '
                             'e.g. "10,10,10,10,10,10,10,10,10,10;0,0,0,0,24,0,24,24,24,4"')
    args = parser.parse_args()

    mins = normalize_min_values(
        min_values=args.min_values, min_value=args.min_value)
    must_beat_pool = parse_must_beat(args.must_beat)

    print(f'NUMBA_OK={NUMBA_OK}')
    print(f'Loading {args.input} ...')
    print(f'min_values = {mins.tolist()}')
    if must_beat_pool.shape[0] > 0:
        print(f'must_beat count = {must_beat_pool.shape[0]}')
        for i, row in enumerate(must_beat_pool, start=1):
            print(f'  must_beat[{i}] = {row.tolist()}')
    else:
        print('must_beat count = 0')

    data = np.load(args.input)
    pool = data['strategies'].astype(np.int16)
    print(f'Pool size = {pool.shape[0]:,}')

    (
        best_x, best_w, best_t, best_l,
        best_mbw, best_mbt, best_mbl, best_mbtotal,
        restart_results, mins
    ) = global_search(
        pool=pool,
        restarts=args.restarts,
        iters=args.iters,
        neighborhood=args.neighborhood,
        max_step=args.max_step,
        seed=args.seed,
        verbose=True,
        min_values=mins,
        must_beat_pool=must_beat_pool,
    )

    print('\n=== FINAL BEST ===')
    print('min_values =', mins.tolist())
    print('strategy =', best_x.tolist())
    print(f'used={int(best_x.sum())}, saved={TOTAL - int(best_x.sum())}')
    print(f'must_beat={best_mbw}/{best_mbtotal}')
    print(f'wins={best_w}, ties={best_t}, losses={best_l}')
    print(f'win_rate={best_w / pool.shape[0]:.4f}')
    if best_mbtotal > 0 and best_mbw != best_mbtotal:
        print('WARNING: current best result did NOT beat all required strategies.')

    save_restart_results(
        args.save, restart_results, best_x,
        best_w, best_t, best_l,
        best_mbw, best_mbt, best_mbl, best_mbtotal,
        pool.shape[0], mins, must_beat_pool
    )
    print(f'Saved to {args.save}')


if __name__ == '__main__':
    main()

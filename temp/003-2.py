# sand_round3_custom_pool.py
# Python 3.9+
import argparse, json, math, os, random, heapq, time
from dataclasses import dataclass
from typing import List, Tuple, Optional

TOTAL = 100
N = 10

# ---------- Round3 scoring ----------
def score_pair_round3(a: List[int], b: List[int], delta_trigger: int = 10) -> Tuple[int, int]:
    trig_a = None
    trig_b = None
    for i, (ai, bi) in enumerate(zip(a, b)):
        if trig_a is None and ai - bi >= delta_trigger:
            trig_a = i
        if trig_b is None and bi - ai >= delta_trigger:
            trig_b = i
        if trig_a is not None and trig_b is not None:
            break

    sa = sb = 0
    for i, (ai, bi) in enumerate(zip(a, b)):
        v = i + 1
        if ai > bi:
            sa += (2 * v) if (trig_a is not None and i > trig_a) else v
        elif bi > ai:
            sb += (2 * v) if (trig_b is not None and i > trig_b) else v
    return sa, sb

# ---------- helpers ----------
def _fix_sum_100(x: List[int], rng: random.Random) -> List[int]:
    diff = TOTAL - sum(x)
    while diff != 0:
        j = rng.randrange(N)
        if diff > 0:
            x[j] += 1
            diff -= 1
        else:
            if x[j] > 0:
                x[j] -= 1
                diff += 1
    return x

def alloc_from_weights(weights: List[float], rng: random.Random) -> List[int]:
    s = sum(weights)
    x = [int(TOTAL * w / s) for w in weights]
    return _fix_sum_100(x, rng)

def random_alloc(rng: random.Random) -> List[int]:
    w = [rng.random() for _ in range(N)]
    return alloc_from_weights(w, rng)

def mutate_transfer(x: List[int], rng: random.Random, step: int = 6) -> List[int]:
    y = x[:]
    i, j = rng.sample(range(N), 2)
    amt = rng.randint(1, step)
    if y[i] >= amt:
        y[i] -= amt
        y[j] += amt
    return y

def mutate_swap(x: List[int], rng: random.Random) -> List[int]:
    y = x[:]
    i, j = rng.sample(range(N), 2)
    y[i], y[j] = y[j], y[i]
    return y

def mutate_restart(rng: random.Random) -> List[int]:
    return random_alloc(rng)

# ---------- load elite pool as opponents ----------
def load_elite_allocs(path: str) -> List[List[int]]:
    if not path or (not os.path.exists(path)):
        return []
    allocs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # allow either {"alloc":[...]} or raw [...]
            if isinstance(rec, dict) and "alloc" in rec:
                a = rec["alloc"]
            else:
                a = rec
            if len(a) != 10 or sum(a) != 100:
                raise ValueError(f"Bad elite row: {a}")
            allocs.append(a)
    if not allocs:
        raise ValueError("elite_pool file exists but no valid rows found.")
    return allocs

# ---------- opponent generators you requested ----------
def gen_uniform10_fluct(rng: random.Random) -> List[int]:
    # around 10 with noise (your "全10的上限波动的")
    x = [max(0, int(rng.gauss(10, 2))) for _ in range(N)]
    if sum(x) == 0:
        x = [10] * N
    # rescale
    s = sum(x)
    x = [int(TOTAL * xi / s) for xi in x]
    return _fix_sum_100(x, rng)

def gen_concentrated_high_with_trigger(rng: random.Random) -> List[int]:
    # focus 8/9/10, and also put a "trigger bomb" in one of 1..6 (index 0..5)
    # idea: reserve t in [14..24] to a low pile; remaining concentrate at 8/9/10
    t_idx = rng.randrange(0, 6)
    t_val = rng.randint(14, 24)

    remain = TOTAL - t_val
    weights = [rng.uniform(0.05, 0.2) for _ in range(N)]
    for i in (7, 8, 9):
        weights[i] = rng.uniform(1.0, 2.0)
    # allocate remaining by weights, then force trigger pile to t_val
    base = alloc_from_weights(weights, rng)
    # scale base to sum=remain
    s = sum(base)
    base = [int(remain * bi / s) for bi in base]
    base = _fix_sum_100(base, rng)
    base[t_idx] += t_val
    return _fix_sum_100(base, rng)

def gen_concentrated_mid_with_trigger(rng: random.Random) -> List[int]:
    # focus 6/7/8 (index 5/6/7), and put a trigger bomb in one of 1..5 (index 0..4)
    t_idx = rng.randrange(0, 5)
    t_val = rng.randint(14, 24)

    remain = TOTAL - t_val
    weights = [rng.uniform(0.05, 0.2) for _ in range(N)]
    for i in (5, 6, 7):
        weights[i] = rng.uniform(1.0, 2.0)
    base = alloc_from_weights(weights, rng)
    s = sum(base)
    base = [int(remain * bi / s) for bi in base]
    base = _fix_sum_100(base, rng)
    base[t_idx] += t_val
    return _fix_sum_100(base, rng)

# ---------- your weighted opponent pool ----------
@dataclass
class PoolConfig:
    # weights you gave
    w_elite: float = 0.4
    w_random: float = 0.2
    w_high_trig: float = 0.4
    w_mid_trig: float = 0.3
    w_uniform10: float = 0.4

    # sample size per group each evaluation (controls stability vs speed)
    k_each: int = 80

    # round3 trigger threshold
    delta_trigger: int = 10

def evaluate_vs_pool(
    a: List[int],
    elite_opps: List[List[int]],
    rng: random.Random,
    cfg: PoolConfig,
) -> Tuple[float, float]:
    """
    returns (weighted_wins, weighted_mean_margin)
    weighted_wins counts only wins (tie/loss=0), weighted by group weights.
    mean_margin is weighted average of (sa-sb).
    """
    # helper to accumulate
    total_w = 0.0
    win_w = 0.0
    margin_sum = 0.0

    def add_one(b: List[int], w: float):
        nonlocal total_w, win_w, margin_sum
        sa, sb = score_pair_round3(a, b, delta_trigger=cfg.delta_trigger)
        total_w += w
        margin_sum += w * (sa - sb)
        if sa > sb:
            win_w += w

    k = cfg.k_each

    # 1) elite_pool opponents (sample with replacement)
    if elite_opps:
        for _ in range(k):
            b = elite_opps[rng.randrange(len(elite_opps))]
            add_one(b, cfg.w_elite)

    # 2) random
    for _ in range(k):
        add_one(random_alloc(rng), cfg.w_random)

    # 3) high(8/9/10) + low trigger
    for _ in range(k):
        add_one(gen_concentrated_high_with_trigger(rng), cfg.w_high_trig)

    # 4) mid(6/7/8) + low trigger
    for _ in range(k):
        add_one(gen_concentrated_mid_with_trigger(rng), cfg.w_mid_trig)

    # 5) uniformish around 10
    for _ in range(k):
        add_one(gen_uniform10_fluct(rng), cfg.w_uniform10)

    mean_margin = (margin_sum / total_w) if total_w > 0 else 0.0
    return win_w, mean_margin

# ---------- breeding (random sampling keep topK) ----------
def keep_topk(
    out_path: str,
    elite_opps: List[List[int]],
    sample_n: int,
    topk: int,
    seed: int,
    cfg: PoolConfig,
    log_every: int = 20000,
) -> None:
    rng = random.Random(seed)
    heap = []  # min-heap (key, alloc)
    t0 = time.time()

    for i in range(1, sample_n + 1):
        a = random_alloc(rng)

        ev_rng = random.Random(rng.getrandbits(64))
        wins, mean_margin = evaluate_vs_pool(a, elite_opps, ev_rng, cfg)
        key = (wins, mean_margin)

        if len(heap) < topk:
            heapq.heappush(heap, (key, a))
        else:
            if key > heap[0][0]:
                heapq.heapreplace(heap, (key, a))

        if log_every and i % log_every == 0:
            best_key = max(heap, key=lambda z: z[0])[0]
            dt = time.time() - t0
            print(f"[breed] {i}/{sample_n} heap={len(heap)} best={best_key} {dt:.1f}s")

    heap.sort(key=lambda z: z[0], reverse=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for key, a in heap:
            f.write(json.dumps({"wins_w": key[0], "mean_margin": key[1], "alloc": a}, ensure_ascii=False) + "\n")
    print(f"[breed] saved top{topk} -> {out_path}")

# ---------- anneal ----------
def load_pool_allocs(path: str) -> List[List[int]]:
    allocs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if isinstance(rec, dict) and "alloc" in rec:
                a = rec["alloc"]
            else:
                a = rec
            if len(a) == 10 and sum(a) == 100:
                allocs.append(a)
    return allocs

def anneal(
    elite_pool_path: str,
    seed: int,
    steps: int,
    cfg: PoolConfig,
    start_from_top: int = 30,
    step_move: int = 6,
    log_every: int = 5000,
) -> Tuple[List[int], Tuple[float, float]]:
    elite_pool_allocs = load_pool_allocs(elite_pool_path)
    if not elite_pool_allocs:
        raise ValueError("elite_pool.jsonl is empty/unreadable for allocs.")

    elite_opps = elite_pool_allocs  # IMPORTANT: elite_pool used as opponents too

    rng = random.Random(seed)
    m = min(start_from_top, len(elite_pool_allocs))
    cur = elite_pool_allocs[rng.randrange(m)][:]

    def eval_key(x: List[int]) -> Tuple[float, float]:
        ev_rng = random.Random(rng.getrandbits(64))
        return evaluate_vs_pool(x, elite_opps, ev_rng, cfg)

    cur_key = eval_key(cur)
    best = cur[:]
    best_key = cur_key

    # temperature schedule
    temp0, temp1 = 1.0, 0.02

    t0 = time.time()
    for s in range(1, steps + 1):
        T = temp0 * (temp1 / temp0) ** (s / steps)

        # mixed mutations: local move + occasional swap/restart
        r = rng.random()
        if r < 0.86:
            cand = mutate_transfer(cur, rng, step=step_move)
        elif r < 0.98:
            cand = mutate_swap(cur, rng)
        else:
            cand = mutate_restart(rng)

        cand_key = eval_key(cand)

        def scalar(k):
            # wins dominates hard; mean margin tie-break
            return k[0] * 1000.0 + k[1]

        dc = scalar(cand_key) - scalar(cur_key)
        if dc >= 0 or rng.random() < math.exp(dc / max(1e-9, T)):
            cur, cur_key = cand, cand_key

        if cand_key > best_key:
            best, best_key = cand[:], cand_key

        if log_every and s % log_every == 0:
            dt = time.time() - t0
            print(f"[anneal] {s}/{steps} | T={T:.4f} | cur={cur_key} | best={best_key} | {dt:.1f}s")

    return best, best_key

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("breed")
    pb.add_argument("--elite", type=str, required=True, help="elite_pool.jsonl used as opponents")
    pb.add_argument("--out", type=str, default="elite_pool2.jsonl")
    pb.add_argument("--sample", type=int, default=1_000_000)
    pb.add_argument("--topk", type=int, default=1000)
    pb.add_argument("--seed", type=int, default=1)

    pa = sub.add_parser("anneal")
    pa.add_argument("--elite", type=str, required=True, help="elite_pool.jsonl used as opponents and as start pool")
    pa.add_argument("--seed", type=int, default=2)
    pa.add_argument("--steps", type=int, default=200_000)
    pa.add_argument("--start_from_top", type=int, default=30)
    pa.add_argument("--step_move", type=int, default=6)

    # pool config
    for sp in (pb, pa):
        sp.add_argument("--k_each", type=int, default=80)
        sp.add_argument("--w_elite", type=float, default=0.4)
        sp.add_argument("--w_random", type=float, default=0.2)
        sp.add_argument("--w_high_trig", type=float, default=0.4)
        sp.add_argument("--w_mid_trig", type=float, default=0.3)
        sp.add_argument("--w_uniform10", type=float, default=0.4)
        sp.add_argument("--delta_trigger", type=int, default=10)

    args = p.parse_args()
    cfg = PoolConfig(
        w_elite=args.w_elite,
        w_random=args.w_random,
        w_high_trig=args.w_high_trig,
        w_mid_trig=args.w_mid_trig,
        w_uniform10=args.w_uniform10,
        k_each=args.k_each,
        delta_trigger=args.delta_trigger,
    )

    elite_opps = load_elite_allocs(args.elite)

    if args.cmd == "breed":
        keep_topk(
            out_path=args.out,
            elite_opps=elite_opps,
            sample_n=args.sample,
            topk=args.topk,
            seed=args.seed,
            cfg=cfg,
            log_every=20000
        )
    else:
        best, best_key = anneal(
            elite_pool_path=args.elite,
            seed=args.seed,
            steps=args.steps,
            cfg=cfg,
            start_from_top=args.start_from_top,
            step_move=args.step_move,
            log_every=5000
        )
        print("\n===== FINAL BEST =====")
        print("alloc:", best, "sum=", sum(best))
        print("key (wins_w, mean_margin):", best_key)

if __name__ == "__main__":
    main()
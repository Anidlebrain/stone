# sand_pile_round3_breeding.py
# Python 3.9+
import argparse, json, math, os, random, statistics as stats, heapq, time
from dataclasses import dataclass
from typing import List, Tuple, Optional

TOTAL = 100
N = 10

# ---------------- Round3 scoring ----------------
def score_pair_round3(a: List[int], b: List[int], delta_trigger: int = 10) -> Tuple[int, int]:
    """
    Round3:
    - Trigger for a: first i with a[i]-b[i] >= 10
    - After trigger, any later won pile (i > trigger) worth 2*(i+1)
    - Pile at trigger index itself is NOT doubled (matches your example).
    - Symmetric for b.
    - Tie => nobody gets that pile.
    """
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

def duel_result(a: List[int], b: List[int]) -> int:
    sa, sb = score_pair_round3(a, b)
    return 1 if sa > sb else (-1 if sb > sa else 0)

# ---------------- Allocation helpers ----------------
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

def random_alloc_dirichletish(rng: random.Random) -> List[int]:
    w = [rng.random() for _ in range(N)]
    s = sum(w)
    x = [int(TOTAL * wi / s) for wi in w]
    return _fix_sum_100(x, rng)

def mutate_transfer(x: List[int], rng: random.Random, step: int = 6) -> List[int]:
    y = x[:]
    i, j = rng.sample(range(N), 2)
    amt = rng.randint(1, step)
    if y[i] >= amt:
        y[i] -= amt
        y[j] += amt
    return y

# ---------------- Opponent generators (your 3 + optional random) ----------------
def gen_concentrated_high(rng: random.Random) -> List[int]:
    # focus on 8/9/10 (index 7/8/9)
    weights = []
    for i in range(N):
        if i >= 7:
            weights.append(rng.uniform(1.0, 1.8))
        else:
            weights.append(rng.uniform(0.05, 0.25))
    return _weights_to_alloc(weights, rng)

def gen_concentrated_mid(rng: random.Random) -> List[int]:
    # focus on 6/7/8 (index 5/6/7)
    weights = []
    for i in range(N):
        if 5 <= i <= 7:
            weights.append(rng.uniform(1.0, 1.8))
        else:
            weights.append(rng.uniform(0.05, 0.25))
    return _weights_to_alloc(weights, rng)

def gen_uniformish(rng: random.Random) -> List[int]:
    # around 10 with noise
    x = [max(0, int(rng.gauss(10, 2))) for _ in range(N)]
    if sum(x) == 0:
        x = [10] * N
    return _rescale_to_100(x, rng)

def gen_random(rng: random.Random) -> List[int]:
    return random_alloc_dirichletish(rng)

def _weights_to_alloc(weights: List[float], rng: random.Random) -> List[int]:
    s = sum(weights)
    x = [int(TOTAL * w / s) for w in weights]
    return _fix_sum_100(x, rng)

def _rescale_to_100(x: List[int], rng: random.Random) -> List[int]:
    s = sum(x)
    x = [int(TOTAL * xi / s) for xi in x]
    return _fix_sum_100(x, rng)

# ---------------- Mixed evaluation to avoid overfit ----------------
@dataclass
class EvalConfig:
    # weights for mixing opponent pools
    w_real: float = 1.0
    w_uniform: float = 0.6
    w_mid: float = 0.6
    w_high: float = 0.6
    w_random: float = 0.2

    # how many generated opponents per eval
    gen_each: int = 60

    # objective weights (tie-break order still wins first then mean margin)
    # we compute "weighted wins" and "weighted mean margin"
    # but we still sort by (wins, mean_margin)
    pass

def make_generated_opponents(rng: random.Random, cfg: EvalConfig) -> List[List[int]]:
    opp = []
    for _ in range(cfg.gen_each): opp.append(gen_uniformish(rng))
    for _ in range(cfg.gen_each): opp.append(gen_concentrated_mid(rng))
    for _ in range(cfg.gen_each): opp.append(gen_concentrated_high(rng))
    for _ in range(cfg.gen_each): opp.append(gen_random(rng))
    return opp

def load_real_opponents(path: Optional[str]) -> List[List[int]]:
    if not path:
        return []
    if not os.path.exists(path):
        return []
    real = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            arr = json.loads(line)
            if len(arr) != 10 or sum(arr) != 100:
                raise ValueError(f"Bad opponent row: {arr}")
            real.append(arr)
    return real

def evaluate_mixed(a: List[int], real_opps: List[List[int]], rng: random.Random, cfg: EvalConfig) -> Tuple[float, float]:
    """
    Returns (weighted_wins, weighted_mean_margin)
    Weighted wins: sum(weight * win_indicator) where win_indicator=1 for win, 0 for loss/tie
    Weighted mean margin: weighted average of (sa-sb)
    """
    total_w = 0.0
    win_w = 0.0
    margin_sum = 0.0

    # Real opponents
    if real_opps:
        w = cfg.w_real
        for b in real_opps:
            sa, sb = score_pair_round3(a, b)
            total_w += w
            margin_sum += w * (sa - sb)
            if sa > sb:
                win_w += w

    # Generated opponents (fresh each eval => less overfit)
    gens = make_generated_opponents(rng, cfg)

    # apply pool weights by type blocks (uniform, mid, high, random) each size=gen_each
    blocks = [
        (cfg.w_uniform, gens[0:cfg.gen_each]),
        (cfg.w_mid, gens[cfg.gen_each:2*cfg.gen_each]),
        (cfg.w_high, gens[2*cfg.gen_each:3*cfg.gen_each]),
        (cfg.w_random, gens[3*cfg.gen_each:4*cfg.gen_each]),
    ]
    for w, bs in blocks:
        for b in bs:
            sa, sb = score_pair_round3(a, b)
            total_w += w
            margin_sum += w * (sa - sb)
            if sa > sb:
                win_w += w

    mean_margin = margin_sum / total_w if total_w > 0 else 0.0
    return win_w, mean_margin

# ---------------- Breeding phase: sample many, keep TopK ----------------
def keep_topk_random(
    out_path: str,
    real_opps: List[List[int]],
    sample_n: int,
    topk: int,
    seed: int,
    cfg: EvalConfig,
    log_every: int = 20000,
) -> None:
    rng = random.Random(seed)
    heap = []  # min-heap of (key, alloc)
    t0 = time.time()

    for i in range(1, sample_n + 1):
        a = random_alloc_dirichletish(rng)

        # IMPORTANT: use a separate RNG stream for evaluation randomness
        ev_rng = random.Random(rng.getrandbits(64))
        wins, mean_margin = evaluate_mixed(a, real_opps, ev_rng, cfg)

        key = (wins, mean_margin)  # lexicographic
        if len(heap) < topk:
            heapq.heappush(heap, (key, a))
        else:
            if key > heap[0][0]:
                heapq.heapreplace(heap, (key, a))

        if log_every and i % log_every == 0:
            best_key = max(heap, key=lambda z: z[0])[0]
            dt = time.time() - t0
            print(f"[breed] {i}/{sample_n} | heap={len(heap)} | best={best_key} | {dt:.1f}s")

    # sort descending
    heap.sort(key=lambda z: z[0], reverse=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for key, a in heap:
            rec = {"wins_w": key[0], "mean_margin": key[1], "alloc": a}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[breed] saved top{topk} -> {out_path}")

# ---------------- Anneal phase: start from elite pool, optimize ----------------
def load_pool(path: str) -> List[Tuple[Tuple[float, float], List[int]]]:
    pool = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["wins_w"], rec["mean_margin"])
            pool.append((key, rec["alloc"]))
    pool.sort(key=lambda z: z[0], reverse=True)
    return pool

def anneal_from_pool(
    pool: List[Tuple[Tuple[float, float], List[int]]],
    real_opps: List[List[int]],
    seed: int,
    cfg: EvalConfig,
    steps: int = 200000,
    start_from_top: int = 20,
    step_move: int = 6,
    temp0: float = 1.0,
    temp1: float = 0.02,
    log_every: int = 5000,
) -> Tuple[List[int], Tuple[float, float]]:
    rng = random.Random(seed)
    # pick a starting elite among top M
    m = min(start_from_top, len(pool))
    start_idx = rng.randrange(m) if m > 0 else 0
    cur = pool[start_idx][1][:] if pool else random_alloc_dirichletish(rng)

    def eval_key(x: List[int]) -> Tuple[float, float]:
        ev_rng = random.Random(rng.getrandbits(64))
        return evaluate_mixed(x, real_opps, ev_rng, cfg)

    cur_key = eval_key(cur)
    best = cur[:]
    best_key = cur_key

    t0 = time.time()
    for s in range(1, steps + 1):
        # temperature schedule
        t = temp0 * (temp1 / temp0) ** (s / steps)

        cand = mutate_transfer(cur, rng, step=step_move)
        cand_key = eval_key(cand)

        # acceptance: compare scalarized score but keep lexicographic best
        # scalarization: wins dominates; mean margin as tie breaker
        def scalar(k):
            return k[0] * 1000.0 + k[1]
        dc = scalar(cand_key) - scalar(cur_key)

        accept = False
        if dc >= 0:
            accept = True
        else:
            # SA probability
            if rng.random() < math.exp(dc / max(1e-9, t)):
                accept = True

        if accept:
            cur, cur_key = cand, cand_key

        if cand_key > best_key:
            best, best_key = cand[:], cand_key

        if log_every and s % log_every == 0:
            dt = time.time() - t0
            print(f"[anneal] {s}/{steps} | T={t:.4f} | cur={cur_key} | best={best_key} | {dt:.1f}s")

    return best, best_key

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_breed = sub.add_parser("breed", help="random sample and keep topK")
    p_breed.add_argument("--real", type=str, default="", help="real opponents jsonl (each line is a 10-int array sum=100)")
    p_breed.add_argument("--out", type=str, default="elite_pool.jsonl")
    p_breed.add_argument("--sample", type=int, default=1_000_000)
    p_breed.add_argument("--topk", type=int, default=1000)
    p_breed.add_argument("--seed", type=int, default=1)

    p_anneal = sub.add_parser("anneal", help="anneal starting from elite pool")
    p_anneal.add_argument("--real", type=str, default="")
    p_anneal.add_argument("--pool", type=str, default="elite_pool.jsonl")
    p_anneal.add_argument("--seed", type=int, default=2)
    p_anneal.add_argument("--steps", type=int, default=200_000)
    p_anneal.add_argument("--start_from_top", type=int, default=20)
    p_anneal.add_argument("--step_move", type=int, default=6)

    # mix weights / gen sizes
    for sp in (p_breed, p_anneal):
        sp.add_argument("--gen_each", type=int, default=60)
        sp.add_argument("--w_real", type=float, default=1.0)
        sp.add_argument("--w_uniform", type=float, default=0.6)
        sp.add_argument("--w_mid", type=float, default=0.6)
        sp.add_argument("--w_high", type=float, default=0.6)
        sp.add_argument("--w_random", type=float, default=0.2)

    args = p.parse_args()
    cfg = EvalConfig(
        w_real=args.w_real,
        w_uniform=args.w_uniform,
        w_mid=args.w_mid,
        w_high=args.w_high,
        w_random=args.w_random,
        gen_each=args.gen_each
    )
    real_opps = load_real_opponents(args.real)

    if args.cmd == "breed":
        keep_topk_random(
            out_path=args.out,
            real_opps=real_opps,
            sample_n=args.sample,
            topk=args.topk,
            seed=args.seed,
            cfg=cfg,
            log_every=20000
        )
    elif args.cmd == "anneal":
        pool = load_pool(args.pool)
        best, best_key = anneal_from_pool(
            pool=pool,
            real_opps=real_opps,
            seed=args.seed,
            cfg=cfg,
            steps=args.steps,
            start_from_top=args.start_from_top,
            step_move=args.step_move,
            log_every=5000
        )
        print("\n===== FINAL BEST =====")
        print("alloc:", best, "sum=", sum(best))
        print("key (wins_w, mean_margin):", best_key)
    else:
        raise RuntimeError("Unknown cmd")

if __name__ == "__main__":
    main()